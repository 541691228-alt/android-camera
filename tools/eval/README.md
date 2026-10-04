# 构图算法离线评测

拍完照自动裁一刀（`AutoFrame.bestCrop`）到底有没有用？这组脚本给一套不需要真机、不需要模拟器的
离线基准：拿 DUTS-TE 里 200 张带主体掩膜的图，把像素或显著度网格喂给**端上同一份 Kotlin 算法代码**，
导出 TSV，再由 Python 算指标。

跑的是 app 主源集里那几个类本身（`AutoFrame`、`SaliencyMath`），不是另写一遍 Python 版算法 ——
否则测的是复刻品，不是线上代码。

## 跑一遍

```bash
# 1) 准备数据：下 DUTS-TE，抽样 200 张，产出 work/manifest.tsv（image<TAB>mask）
python tools/eval/prepare_data.py --limit 200

# 2) 显著度从哪来，二选一
python tools/eval/pack_pixels.py --manifest work/manifest.tsv --out-dir work   # 规则算法：原始像素
python tools/eval/model_grid.py --manifest work/manifest.tsv --out-dir work   # u2netp：64x48 显著度网格

# 3) 跑 Kotlin 侧算法（必须先 cd 到仓库根，否则 gradle 找不到 build）
export JAVA_HOME=/path/to/jdk17 ANDROID_HOME=/path/to/android-sdk
export EVAL_MANIFEST=work/pixels_manifest.tsv
export EVAL_PIXELS=work/pixels.bin            # 用 EVAL_GRID 时改成 work/grids.bin
export EVAL_OUT=work/harness_src.tsv
./gradlew :app:testDebugUnitTest --tests 'cn.yege.dshcam.EvalHarnessTest' --rerun

# 4) 算指标、出 markdown/json 报告
python tools/eval/run_eval.py --manifest work/manifest.tsv --harness work/harness_src.tsv --label rule-src

# 5) 自检（不用真机）
python tools/eval/metrics.py             # metrics self-test OK
python tools/eval/test_metrics.py        # 12 个用例
```

Windows PowerShell 把 `export A=B` 换成 `$env:A='B'`。`--rerun` 不能省：不加的话 gradle 会
判定测试 UP-TO-DATE，直接跳过、不产出文件。

### EVAL_* 环境变量

| 变量 | 作用 |
| --- | --- |
| `EVAL_MANIFEST` | 清单：`index<TAB>image<TAB>w<TAB>h`，多余列忽略，可带表头 |
| `EVAL_PIXELS` | 原始像素，每张 64x48 个大端 int32（12288 字节） |
| `EVAL_GRID` | 显著度网格，每张 64x48 个大端 float32（12288 字节）；**给了它就忽略 `EVAL_PIXELS`** |
| `EVAL_ASPECT` | `src`（默认，保持原图比例只变焦）或正数（`1`、`0.75`）；写错直接抛异常，不悄悄退回 src |
| `EVAL_OUT` | 输出 TSV 路径 |

不设 `EVAL_*` 时这个测试自己 `Assume` 跳过，常规 `gradlew test` 里只记一条 skipped。

**为什么像素要外部喂**：Android 模块的单测拿 `android.jar` 当编译期 JDK，`java.desktop` 整个包不在，
`javax.imageio` / `java.awt` 全都用不了，Kotlin 侧读不了 JPEG。所以解码、缩放放 Python，
Kotlin 只管跑算法。

## 指标

| 指标 | 含义 |
| --- | --- |
| 保留率 `kept_ratio` | 裁完框里还剩多少主体掩膜像素，1.0 = 一个像素没切 |
| 密度 `density` | 框内主体像素占比，越高说明框得越紧 |
| 居中裁剪 `center` | 同样尺寸、位置居中的框（最朴素的基线） |
| 最优位置 `optimal` | 同尺寸下平移到掩膜最优位置的上界 |
| `capture_ratio` | Σ(算法−居中)/Σ(最优−居中)，只在最优高于居中的图上算 |
| 三分点距离 | 主体质心到最近三分交点的距离（归一化，交点为 0，正中心约 0.2357 最远） |
| 摆正中比例 | 主体质心离裁剪框中心 5% 以内 |
| 切到主体比例 | 框内留下不到 99% 的主体像素 |

保留率单看会骗人：整图不裁拿 1.0000，但密度只有 0.1413。所以两组指标必须并列看 ——
保留率回答"有没有把主体切掉"，三分点距离才回答"主体摆得正不正"。

## DUTS-TE 200 张实测（2026-10-05）

显著度两条来源：**模型** = 线上优先用的 u2netp 输出；**规则** = 模型没加载时的 `salienceFromArgb` 兜底。
比例三档：`src` 是线上真实口径（`CameraController.kt` 传原图比例 + minKeep 0.60），1:1 和 3:4 是"换比例重构"的假设口径。

保留率与胜率：

| 显著度 | 目标比例 | 算法 | 居中 | 最优 | 胜/平/败 | 提升捕获 | 不裁比例 | 平均裁剪面积 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 模型 | src | 0.9777 | 0.9751 | 0.9968 | 14.0/53.5/32.5 | +39.7% | 30.0% | 0.752 |
| 模型 | 1:1 | 0.9292 | 0.9109 | 0.9803 | 30.5/27.0/42.5 | +45.8% | 1.0% | 0.640 |
| 模型 | 3:4 | 0.8775 | 0.8642 | 0.9518 | 28.5/24.5/47.0 | +20.8% | 5.5% | 0.571 |
| 规则 | src | 0.9872 | 0.9880 | 0.9973 | 6.0/76.5/17.5 | +30.0% | 66.5% | 0.884 |
| 规则 | 1:1 | 0.9119 | 0.9301 | 0.9814 | 23.0/37.0/40.0 | −13.4% | 3.5% | 0.706 |
| 规则 | 3:4 | 0.8464 | 0.8712 | 0.9517 | 22.5/29.0/48.5 | −21.8% | 8.0% | 0.615 |

构图质量（算法 / 居中）：

| 显著度 | 目标比例 | 三分点距离 | 摆正中 | 切到主体 |
| --- | --- | --- | --- | --- |
| 模型 | src | **0.0982** / 0.1543 | 3.0% / 13.0% | 31.0% / 22.0% |
| 模型 | 1:1 | **0.1072** / 0.1680 | 3.0% / 11.5% | 53.0% / 49.5% |
| 模型 | 3:4 | **0.1216** / 0.1711 | 2.0% / 9.5% | 66.5% / 56.5% |
| 规则 | src | 0.1412 / 0.1557 | 12.0% / 16.0% | 17.0% / 10.0% |
| 规则 | 1:1 | 0.1499 / 0.1652 | 4.5% / 13.5% | 46.0% / 41.5% |
| 规则 | 3:4 | 0.1788 / 0.1694 | 4.0% / 12.0% | 59.0% / 53.0% |

复现命令（六份报告都在 `work/`，`harness_*.tsv` 是原始逐图输出，`report_*.{md,json}` 是汇总）。
模型口径换比例只要改 `EVAL_ASPECT` 和输出名：

```bash
export EVAL_MANIFEST=work/grids_manifest.tsv
export EVAL_GRID=work/grids.bin
export EVAL_ASPECT=src EVAL_OUT=work/harness_model_src.tsv
./gradlew :app:testDebugUnitTest --tests 'cn.yege.dshcam.EvalHarnessTest' --rerun
python tools/eval/run_eval.py --manifest work/manifest.tsv \
    --harness work/harness_model_src.tsv --label model-src --aspect src
# 1:1 把 EVAL_ASPECT/--aspect 换成 1、输出名换成 harness_model_1to1；3:4 换成 0.75
```

## 结论

1. **模型确实让这套算法值钱了。** 线上优先走的 u2netp 显著度下，三个比例里算法保留率都高于居中裁剪
   （src 0.9777 vs 0.9751，1:1 0.9292 vs 0.9109，3:4 0.8775 vs 0.8642），而且差距主要来自构图：
   三分点距离比居中裁剪小 29%~36%（0.0982 vs 0.1543），摆正中的比例从 13% 掉到 3%。
   它平均只留 75% 的画面，却保住了 97.8% 的主体 —— 这就是"裁紧了，主体还在，而且更靠三分点"。
2. **规则兜底显著度是这条链路的短板。** 换成 `salienceFromArgb`，src 口径基本打平（0.9872 vs 0.9880），
   换比例时反而比居中裁剪差（1:1 −13.4%、3:4 −21.8%）；而且 66.5% 的图直接不裁了 ——
   规则图的峰值太弥散，`bestCrop` 那道"bestScore ≥ base + 0.05 才动剪刀"的门槛（`AutoFrame.kt:589`）
   过不去，算法等于没运行。**要提升自动构图，先提升兜底显著度，而不是改搜框逻辑。**
3. **代价是切到主体边缘。** 模型口径下"切到主体"从 22% 升到 31%（换比例时 53%/66%）。
   注意判据是"框内留下 < 99% 的主体像素"，切掉 1% 边缘也算，所以这个数字偏高；
   但换比例时确实有真切掉的风险，值得单独看一眼。
4. **胜率别单看。** model-src 败率 32.5% 高于胜率 14.0%，均值却仍占优：赢的图平均多留 11.5 个
   保留率点，输的平均只少留 4.2 个。只看命中率会误判这套算法"偏保守、动得少"。
5. **换比例（1:1 / 3:4）不如保持原比例稳。** 同样模型口径，src 的提升捕获 39.7%，
   3:4 只有 20.8%，切到主体的比例也高得多。产品上如果要给"方构图/竖构图"选项，
   最好让用户看到结果再确认。

## 局限（别把上面的数字用过头）

- DUTS-TE 是**显著性**数据集，掩膜标注的是"主体在哪"，不是"这张照片构图好不好"。
  它能证明"没把主体裁掉、主体摆得更靠三分点"，**证明不了照片更好看**。
- 200 张、单一数据集、没有跨数据集交叉验证；种子的抽样偏差没量化。
- 网格只有 64x48，框的宽高比与请求值最多差约 1%（网格量化）。算法、居中基线、最优上界都在
  同一套网格上取框，所以三方对比公平，但绝对值别当精确值读。
- harness 不处理 EXIF 方向（DUTS 图没有方向标记），也不走线上 `GRID_PEAK_GATE = 0.75` 那道
  "没主体就不另存"的闸；这批评测每张都有主体，那道闸不影响结论。
- 掩膜本身有标注噪声（边缘像素、细小主体），"切到主体"的 1% 容差也因此偏敏感。
- `pack_pixels.py` 用 `PIL` 双线性缩放；端上用 `Bitmap.createScaledBitmap(..., true)`。
  大比例缩小时两者接近，但不是逐位一致。
