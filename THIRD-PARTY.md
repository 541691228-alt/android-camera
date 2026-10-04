# 第三方素材与依赖

本仓库（`android-camera` / AI相机 DSHCam）自己的代码用 MIT（见 LICENSE）。
下面这些不是本项目的原创内容，各自的许可如下。

仓库里不带任何图片资源。早期版本里有过 7 张胶片模拟的 3D LUT PNG（`res/raw/lut_*.png`），
名字沿用富士胶片的公开名称、只适合个人自用，已经全部移出仓库；代码里的 LUT 读表通路保留着，
要用的话自己产一份 LUT 放进 `res/raw` 即可。

## 1. 主体模型：`app/src/main/assets/u2netp.onnx`（4.4 MB）

| 项目 | 说明 |
| --- | --- |
| 模型 | **U-2-Net（u2netp 轻量版）**，显著目标检测（Salient Object Detection） |
| 作者 | Xuebin Qin 等，<https://github.com/xuebinqin/U-2-Net> |
| 许可 | **Apache License 2.0** |
| 本仓库里的文件 | 由公开镜像的 ONNX 权重原样复制（未再训练、未改结构）：`BritishWerewolf/U-2-Netp` → `onnx/model.onnx`，亦可从 `Heliosoph/u2net-onnx` → `u2netp.onnx` 获取；文件大小 4,574,861 字节 |
| 用法 | App 内把取景图缩到 320×320、按 ImageNet mean/std 归一化后跑推理，取第 0 个输出（融合掩膜）当作"主体在哪"的热图 |

> 若你要再分发/商用，请自行核对上游仓库的许可与署名要求（Apache-2.0 要求保留版权与许可声明）。

## 2. 依赖库

| 依赖 | 版本 | 许可 |
| --- | --- | --- |
| AndroidX core-ktx / appcompat / activity-ktx / lifecycle-runtime-ktx | 1.13.1 / 1.7.0 / 1.9.3 / 2.8.7 | Apache-2.0 |
| CameraX（core / camera2 / lifecycle / view / mlkit-vision） | 1.4.1 | Apache-2.0 |
| Google ML Kit face-detection | 16.1.7 | Google ML Kit 服务条款（免费，闭源） |
| Google ML Kit image-labeling | 17.0.9 | 同上（当前版本未实际使用，预留） |
| ONNX Runtime Android | 1.20.0 | MIT |
| AndroidX ExifInterface | 1.3.7 | Apache-2.0 |
| JUnit | 4.13.2 | Eclipse Public License 1.0 |
