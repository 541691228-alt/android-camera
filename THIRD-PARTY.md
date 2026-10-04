# 第三方素材与依赖

本仓库（`android-camera` / 叶哥相机 DSHCam）自己的代码用 MIT（见 LICENSE）。
下面这些不是本项目的原创内容，各自的许可如下。

## 1. 主体模型：`app/src/main/assets/u2netp.onnx`（4.4 MB）

| 项目 | 说明 |
| --- | --- |
| 模型 | **U-2-Net（u2netp 轻量版）**，显著目标检测（Salient Object Detection） |
| 作者 | Xuebin Qin 等，<https://github.com/xuebinqin/U-2-Net> |
| 许可 | **Apache License 2.0** |
| 本仓库里的文件 | 由公开镜像的 ONNX 权重原样复制（未再训练、未改结构）：`BritishWerewolf/U-2-Netp` → `onnx/model.onnx`，亦可从 `Heliosoph/u2net-onnx` → `u2netp.onnx` 获取；文件大小 4,574,861 字节 |
| 用法 | App 内把取景图缩到 320×320、按 ImageNet mean/std 归一化后跑推理，取第 0 个输出（融合掩膜）当作"主体在哪"的热图 |

> 若你要再分发/商用，请自行核对上游仓库的许可与署名要求（Apache-2.0 要求保留版权与许可声明）。

## 2. LUT 调色素材：`app/src/main/res/raw/lut_*.png`（7 张，各约 30 KB）

| 项目 | 说明 |
| --- | --- |
| 内容 | 8-bit 3D LUT 转成的 PNG 条带（`lut_astia` / `lut_velvia` / `lut_eterna` / `lut_classic_chrome` / `lut_classic_neg` / `lut_pro_neg_std` / `lut_nostalgic_neg`） |
| 来源 | 由公开的 `.cube` LUT 文件用脚本转成 PNG（转换脚本 `cube_to_lut_png.py` 在本地工具目录，未随仓库上传） |
| 名称 | 沿用富士胶片（FUJIFILM）胶片模拟的公开名称，**仅用于个人学习/自用**；相关商标归富士胶片所有，本项目与富士胶片无任何关联 |
| 建议 | 如果要公开分发或商用，建议换成自己生成/授权的 LUT，或把 `res/raw` 里的 LUT 一并删掉（删掉后 App 只剩"原图"一档，其余代码不受影响） |

## 3. 依赖库

| 依赖 | 版本 | 许可 |
| --- | --- | --- |
| AndroidX core-ktx / appcompat / activity-ktx / lifecycle-runtime-ktx | 1.13.1 / 1.7.0 / 1.9.3 / 2.8.7 | Apache-2.0 |
| CameraX（core / camera2 / lifecycle / view / mlkit-vision） | 1.4.1 | Apache-2.0 |
| Google ML Kit face-detection | 16.1.7 | Google ML Kit 服务条款（免费，闭源） |
| Google ML Kit image-labeling | 17.0.9 | 同上（当前版本未实际使用，预留） |
| ONNX Runtime Android | 1.20.0 | MIT |
| AndroidX ExifInterface | 1.3.7 | Apache-2.0 |
| JUnit | 4.13.2 | Eclipse Public License 1.0 |
