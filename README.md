# 叶哥相机（DSHCam）

一台**自己手写**的 Android 相机：**全离线、不联网、不调云服务**，把"怎么拍好"的判断全放在端上跑。

> 个人自用项目，Android 端 Kotlin + CameraX，2026-10 的 v1.0。
> 特色是三个自己写的"智能"：**自动构图**（拍完自动裁一版更好看的）、**实时构图提示**（带方向和百分比，且不会一直催）、**自动曝光补偿**。

| | |
| --- | --- |
| 语言 / 构建 | Kotlin，Gradle 8.10.2，JDK 17，AGP（见 `build.gradle.kts`） |
| SDK | compileSdk 35 / minSdk 26 / targetSdk 35 |
| 相机 | CameraX 1.4.1 + ML Kit 人脸检测 16.1.7 |
| 端上模型 | u2netp（显著目标检测，4.4 MB，ONNX Runtime 1.20.0，CPU） |
| 测试 | 58 个 JVM 单元测试（不需要真机、不需要模拟器） |
| APK | debug 约 158 MB（含 ONNX Runtime 4 个 ABI + 4.4 MB 模型；只打 arm64 约 60 MB） |
| 网络权限 | **无**（Manifest 里只有 CAMERA，没有 INTERNET） |

---

## 1. 有什么功能

| 功能 | 说明 | 主要文件 |
| --- | --- | --- |
| 预览 / 拍照 | CameraX，前后置切换、点按对焦、双指缩放、闪光灯三档 | `CameraController.kt` |
| 风格 LUT（9 档） | 原图 / 经典 / 鲜艳 / 清新 / 影院 / 复古 / 怀旧 / 淡雅 / 黑白；预览实时生效，成片保存时补一次调色 | `LutEffect.kt`、`res/raw/lut_*.png` |
| 网格 + 画幅 | 3×3 三分线开关；4:3 / 16:9 / 1:1（1:1 拍完居中裁） | `OverlayView.kt`、`CameraController.kt` |
| 水平仪 + 构图分 | 重力/磁力算倾角，规则引擎给 0–100 的构图分与逐条建议 | `SensorPose.kt`、`Rules.kt` |
| 实时构图提示 | "镜头往左移一点（约 13%）"这种带方向和**百分比**的提示；主体锁定后不再乱跳 | `AutoFrame.kt`、`SubjectLock.kt` |
| 自动构图 | 按下快门后自己算裁剪框，达标就另存一张 `..._auto.jpg`（保留原图） | `AutoFrame.kt`、`CameraController.kt` |
| 主体模型 | u2netp 在端上找"主体在哪"，喂给构图提示与自动构图 | `SubjectModel.kt`、`SaliencyMath.kt` |
| 自动曝光 | 画面持续偏暗/过曝时自动走 1/3 EV 挡 | `AutoExposure.kt` |
| 合焦辅助 | 峰值对焦（边缘描色）/ 斑马纹（过曝斜纹），三档 | `FocusAssist.kt` |
| 诊断钩子 | 一串 `adb am start --ez ...` 参数，用来在真机上复现/量测 | `MainActivity.kt` |

界面从上到下：胶囊行（网格 / 画幅 / 风格 / 自动EV / 合焦 / 引导 / 构图）→ 取景区（叠网格、水平仪、峰值、提示卡）→ 曝光滑杆 → 切换 / 快门 / 闪光灯。

---

## 2. 代码结构

```
app/src/main/java/cn/yege/dshcam/
├── MainActivity.kt        界面装配、帧循环、引导决策、诊断钩子
├── CameraController.kt    CameraX 绑定、拍照、EXIF 处理、MediaStore 写入、自动构图落盘
├── LutEffect.kt           CameraEffect + 自写 GL SurfaceProcessor（EGL14/GLES20）做 3D LUT
├── OverlayView.kt         网格 / 水平仪 / 峰值 / 引导箭头的绘制
├── FaceAnalyzer.kt        ML Kit 人脸 + 亮度&梯度网格 + 主体锚点后台线程（YUV→320×320 ARGB）
├── AutoFrame.kt           显著性、构图评分、裁剪搜索、引导文案（纯 Kotlin）
├── SubjectModel.kt        ONNX Runtime 加载 u2netp、推理、输出热图
├── SaliencyMath.kt        输入预处理（NCHW/ImageNet 归一化）、掩膜转网格、旋转映射（纯 Kotlin）
├── SubjectLock.kt         主体位置锁定 + 迟滞（纯 Kotlin）
├── Rules.kt               v1 规则打分（三分/水平/主体大小/曝光/地平线）
├── AutoExposure.kt        曝光补偿决策
├── FocusAssist.kt         峰值 / 斑马纹的判定
└── SensorPose.kt          倾角

app/src/test/java/cn/yege/dshcam/   58 个 JVM 单测（AutoFrame / SaliencyMath / SubjectLock / …）
scripts/build_apk.ps1               PowerShell 一键构建（会自己写 local.properties）
scripts/install_apk.ps1             PowerShell 一键 adb install -r
```

设计上刻意把**纯逻辑**（`AutoFrame` / `SaliencyMath` / `SubjectLock` / `Rules` / `AutoExposure` / `FocusAssist`）写成不 `import android.*` 的类，所以能在普通 JVM 上跑单测 —— 这是这个项目里最省事的一条经验：**图像算法的正确性 90% 可以在没有手机的情况下用网格化的数字验证**。

---

## 3. 三个"智能"是怎么做的

### 3.1 自动构图（拍照后自动裁一版）

1. **找主体**
   - 主路线：u2netp 端上推理 → 320×320 热图 → 降到 64×48 网格；网格峰值 ≥ 0.75 认为"画面里有主体"（0.75 来自实测：拍鼠标 0.996、拍空鼠标垫 0.573）。
   - 兜底路线：自写的**中心-周边对比**显著性（Itti 1998 的简化版，半径 2 和 6 两个尺度的"原图 − 盒式模糊"，取亮度与两个色彩对立通道；加了 `1e-4` 噪声地板，否则纯色画面会被浮点残差放大成"到处是主体"）。
   - 两条路线都不信饱和度、也不信"离平均色远"—— 纯色/渐晕背景会给每格铺一层均匀底座，把质心拽到画面中间，这是最初"怎么裁都不加分"的根因。
2. **评分**：`0.55*三分法 + 0.25*重要格子保留率 + 0.20*面积占比 − 裁剪惩罚`，在 21 档尺寸 × 滑窗里找最优框。
3. **动剪刀门槛**：最优框必须比"最大可用框"（通常就是整图）高 **0.05** 才真裁，否则保持原图 —— 宁可不裁，也不要为了 1% 的分数感裁掉画面。
4. **方向正确**：竖拍（EXIF 6/8）先把显著性网格**转正**、在显示坐标系里算框，再 `mapCropToSensor` 映射回传感器坐标去裁 —— 否则构图目标会从"左三分点"歪成"上三分点"。

### 3.2 实时构图提示：为什么不会"永远在催"

早期版本的问题很具体：**永远在说"向右一点、靠近一点"，没有"度"，也不收敛**。原因和解法：

| 问题 | 解法 |
| --- | --- |
| 没找到主体也硬给方向（背景质心离最近三分点永远 >5%） | 先判"有没有可信主体"（强度 <0.06 或扩散 >0.40 就是没有）→ 文案换成"没找到明显主体，对准要拍的东西"，且**不画箭头** |
| 只在 4 个三分点里选目标、没有"还差多少" | 文案带**百分比**（`约 13%`），容差 **6%**（已经站好位后放宽到 11%，避免手抖来回切） |
| 模型每 1.5 秒在不同物体上跳 → 提示横跳 | `SubjectLock`：连续两个锚点位置互差 ≤0.08 才锁定、锁定后半步融合跟踪、连续两次对不上才改锁/解锁；锁定失败就**回退到规则网格**（规则路径更稳） |
| 规则强度骑在门限上抖（0.039 ↔ 0.069） | 强度/扩散做 `0.6*旧 + 0.4*新` 平滑；已锁定的可信主体用更松的门限（0.045） |
| 人脸场景本该优先 | 检测到人脸直接用人脸中心当主体（强度按脸大小），最高优先级 |

**真机实测对照**（同一台手机、同一个桌面场景）：改前 8 秒内出现 5 种不同提示；改后 25 秒全程只有一句「镜头压低一点（约 15%）」，锁定位漂移 <0.05。

### 3.3 主体锚点管线

```
CameraX 分析帧 (YUV_420_888)
  └─ FaceAnalyzer.analyze()
       ├─ 亮度/梯度网格（每帧，规则路径用）
       └─ maybeRunAnchor()：想开引导 + 模型就绪 + 不忙 + 距上次 ≥1.5s
            └─ sampleArgb() 取 320×320 ARGB（按显示方向反查传感器像素，BT.601）
                 └─ 单线程池 → SubjectModel.salienceGridFromArgb()
                      → AutoFrame.subjectCenter() → SubjectAnchor{x,y,strength,spread,peak,atMs,hasSubject}
                         └─ MainActivity 帧循环：新鲜（<3s）→ SubjectLock → 引导文案
```

推理一次 0.56–1.0 s（4096×3072 取景帧 → 320×320 → CPU），所以不给它每帧跑；锚点当"1.5 秒一次的判卷"，帧间的方向/百分比由平滑和容差兜住。

---

## 4. 编译

```powershell
# 依赖：JDK 17、Android SDK（platform 35 + build-tools）、Gradle 8.10.2（或用仓库里的 ./gradlew）
$env:JAVA_HOME   = 'C:\path\to\jdk17'
$env:ANDROID_HOME = 'C:\path\to\Android\Sdk'      # 或者写 local.properties：sdk.dir=...

# 单元测试 + 出 debug APK（仓库自带 wrapper）
.\gradlew.bat --offline :app:testDebugUnitTest :app:assembleDebug

# 或者用脚本（会自己写 local.properties，默认输出到 dist\）
powershell -File .\scripts\build_apk.ps1 -AndroidSdk $env:ANDROID_HOME -JdkHome $env:JAVA_HOME
```

注意：

- **`local.properties` 不在仓库里**（它指向本机 SDK 路径）。第一次构建要么设 `ANDROID_HOME`，要么自己建一行 `sdk.dir=...`。
- 第一次构建需要联网拉依赖（AndroidX / CameraX / ML Kit / ONNX Runtime）；之后加 `--offline` 就能离线构建。
- `u2netp.onnx` 随仓库一起提供，不需要下载。
- 想瘦身：在 `app/build.gradle.kts` 里加 `ndk { abiFilters += "arm64-v8a" }`，APK 能从 158 MB 掉到 60 MB 上下（ONNX Runtime 的 4 个 ABI 占大头）。

装机：

```powershell
powershell -File .\scripts\install_apk.ps1 -Apk .\app\build\outputs\apk\debug\app-debug.apk
# 或直接
adb install -r -d .\app\build\outputs\apk\debug\app-debug.apk
```

成片位置：`/sdcard/Pictures/DSHCam/DSHCam_<时间戳>.jpg`；自动构图版是同一目录下的 `..._auto.jpg`。

---

## 5. 真机诊断钩子

`MainActivity` 在 `onCreate` 里读一串 intent extra（**只读一次**，所以重复用要先 `adb shell am force-stop cn.yege.dshcam`）：

| 参数 | 作用 |
| --- | --- |
| `--ez capture true` | 启动后自动拍一张 |
| `--ez forcetrim true` | 强制走一遍裁剪/另存链路（用来验证落盘，不依赖算法是否达标） |
| `--ez selftest true` | 一组自检 |
| `--ez autoev true` / `--ei evtest <n>` | 自动曝光自检 / 指定 EV 档 |
| `--ei autoframe <n>` | 指定自动构图模式 |
| `--ez grid true|false`、`--ei ratio <n>` | 网格开关、画幅档 |
| `--ei style <n>` / `--ei style2 <n>` | 风格档（0 原图 … 7 淡雅 / 8 黑白） |
| `--ei assist <n>` | 合焦辅助：0 关 / 1 峰值 / 2 斑马 / 3 全开 |
| `--ei guide <n>` / `--ei autoframe <n>` | 构图引导 / 自动构图：0 关 / 1 开 |
| `--es modeltest <绝对路径>` | 对指定图片跑一次主体模型自检，打印峰值 / 耗时 / 主体位置 |
| `--ez guidedbg true` | 打开引导诊断日志（每个锚点一行 + 每秒一行），默认关 |

例：`adb shell am start -n cn.yege.dshcam/.MainActivity --ez capture true --ez guidedbg true`

日志用 `adb logcat -s CameraController:V DSHCam:V` 看（`CameraController` 是拍照/构图链路，`DSHCam` 是主体模型）。

> 真机小坑：App **读不了** `/sdcard/Android/data/<包名>/files/`（HyperOS 上 `run-as ls` 会被拒），调试要喂图片给 `modeltest` 时用
> `adb push x.jpg /data/local/tmp/` + `adb shell run-as cn.yege.dshcam cp /data/local/tmp/x.jpg files/x.jpg`，再传 `/data/data/cn.yege.dshcam/files/x.jpg`。

---

## 6. 单元测试（58 个）

```
AutoFrameTest      21   显著性 / 评分 / 裁剪搜索 / 引导文案 / EXIF 转正与映射
SaliencyMathTest    7   NCHW 与 ImageNet 归一化、掩膜→网格、显示↔传感器坐标
SubjectLockTest     6   锁定的迟滞：两个锚点才锁、单次抖动不改锁、两次空锚点解锁…
AutoExposureTest    8   曝光补偿决策
FocusAssistTest     8   峰值 / 斑马纹判定
RulesTest           8   v1 规则打分
```

```powershell
.\gradlew.bat --offline :app:testDebugUnitTest
```

几条测试专门守着踩过的坑：纯色画面不能凭空出现"主体"（噪声地板）；主体已经压在三分点上时不能为了 1% 的分数动剪刀；EXIF 6 的裁剪框映射（竖拍时"显示宽度"等于"传感器高度"，容易写反）。

---

## 7. 已知局限

- **杂乱场景里模型会挑错主体**：桌面上一堆颜色鲜艳的杂物时，u2netp 会盯着最亮/最花的那块，锁定逻辑只能让它"别乱跳"，不能让它"挑对人想拍的东西"。
- 盒式模糊的边缘钳位会让纯色画面在画面边缘产生一点点假响应（不足以触发裁剪，但值得记着）。
- 地平线检测没做（`FrameFacts.horizonY` 恒为 `null`），所以规则里"地平线放在三分线"这条暂时不生效。
- 只跑了 arm64 之外的 ABI 没裁，APK 因此偏大。
- 竖拍取景时 `screenOrientation` 固定 portrait，横拍体验没做。

## 8. 隐私

- Manifest 里**没有 `INTERNET` 权限**，App 不联网、不上传照片、没有统计 SDK。
- 主体模型、LUT、规则全在本地；照片只写进系统相册（MediaStore）。
- `android:allowBackup="false"`。

## 9. 许可

- 本项目代码：**MIT**（见 LICENSE）。
- 第三方素材与依赖：见 **THIRD-PARTY.md**（u2netp 是 Apache-2.0；LUT 名称与素材来源请注意其归属）。

---

## English (short)

**DSHCam** — a hand-written, fully offline Android camera (Kotlin + CameraX). No `INTERNET`
permission, no cloud calls: everything (including the 4.4 MB u2netp salient-object model running
on ONNX Runtime) is on-device.

Highlights: 9 LUT looks rendered through a custom GL surface processor; a rule-based composition
score; **auto-framing** that crops a better version after the shutter (`..._auto.jpg`); **live
composition guidance** that reports a direction *and a percentage* and stays quiet when there is
no clear subject (a lock-with-hysteresis layer keeps the model's per-1.5 s subject estimate from
jittering the hints); auto exposure in 1/3 EV steps; focus peaking / zebra.

All image logic is written as plain Kotlin (no `android.*`) so 58 JVM unit tests cover it without a
device. Build: JDK 17 + Android SDK 35, `./gradlew :app:assembleDebug` (add `local.properties`
with `sdk.dir` or set `ANDROID_HOME`). Licensed MIT; third-party notices in THIRD-PARTY.md.
