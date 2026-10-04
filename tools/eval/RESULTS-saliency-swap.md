# 换显著度模型重跑：u2netp / u2net 全量 / RMBG-1.4

2026-10-05。起因是那 21 张「主体被整个切掉」的极端个案（11137 张里 0.19%）。当时的判断是：这不是搜框逻辑的 bug，而是端上有点图模型（u2netp）和数据集标注「选的根本不是同一个物体」。所以这一轮把点图模型换掉，看这些个案会不会消失、整体指标会不会翻盘。

**一句话结论：换模型有用，但只在 DUTS-TE 上翻盘。** DUTS 的坏图 5→0、配对均值 −0.0059→−0.0019、提升捕获从 −11.4% 变 **+9.0%**（唯一一次转正）；DUT-OMRON 也基本拉平（t 从 −2.02 到 −0.37）；**ECSSD 依旧系统性为负**（−0.0119，t=−9.64）。代价是 RMBG 要 1024 输入、176 MB、单张 355~620 ms，端上放不下 —— 结论只能用来改评测口径或指导后续选型，不能直接上手机。

## 实验设置

- **三个模型**（都在 PC 上用 onnxruntime 1.30.0 CPU 跑，8 线程）：

  | 模型 | 输入 | 输出 | 权重 | 单张耗时（本机） |
  | --- | --- | --- | --- | --- |
  | u2netp（**线上基线**） | 320×320 | `d0` 融合图，已 sigmoid | 4.6 MB（`app/src/main/assets/u2netp.onnx`，同权重 PC 复跑） | 未单独测（远小于 u2net 全量） |
  | u2net 全量 | 320×320 | 7 个输出 1959..1965，各 (1,1,320,320)，均已 sigmoid | 176 MB | 228 ms（4 线程） |
  | RMBG-1.4 | 1024×1024 | `output` (1,1,1024,1024)，已 sigmoid | 176 MB | 355 ms（8 线程，含前后处理） |
  | BiRefNet_lite（只做小样本） | 1024×1024 | `output_image` 是 logits（−14.7~101.5，需 sigmoid） | 224 MB | 4156 ms → 放弃全量 |

- **口径**：所有模型都走「PIL BILINEAR 拉伸到模型输入边 → 归一化 → 掩膜 min/max 线性拉伸到 0..1 → 盒式平均降到 64×48 网格 → 交给同一份 `AutoFrame.bestCrop()`」，harness 用 `src` 口径（原图长宽比），与线上一致。三个模型的对比是同一份代码、同一份 manifest、同一套 21 列 harness 输出。
- **数据集**：DUTS-TE 5019 / ECSSD 995 / DUT-OMRON 5123，共 11137 张，全部跑满（不是抽样）。
- **RMBG 的预处理是用 IoU 现场定的**，不是猜的：51 张图上 `x/255`（unit）中位 IoU 0.6275、`(x/255−0.5)/0.5`（half）0.5233、**官方口径（x/255 后 mean 0.5 / std 1）0.8692** → 取官方口径。u2netp 与 u2net 全量都是 ImageNet mean/std。
- **回归保护**：`model_grid.py` 扩了多模型参数后，用 u2netp 在 200 张清单上重跑，`grids.bin` 的 sha256 与改动前逐字节一致（`A4BEDE78…B32EE`），保证默认口径没被改动。

## 主表（预测保留率 / 居中保留率，src 口径）

| 数据集 | 模型 | 预测 | 居中 | 同尺寸上界 | 提升捕获 | 不裁比例 | 切主体 预测/居中 | 三分距 预测/居中 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DUTS-TE 5019 | u2netp | 0.9727 | 0.9785 | 0.9961 | −11.4% | 31.6% | 31.1% / 23.6% | 0.0951 / 0.1467 |
| | u2net 全量 | 0.9745 | 0.9794 | 0.9963 | −6.4% | 31.8% | 30.3% / 23.7% | 0.0926 / 0.1462 |
| | **RMBG** | **0.9846** | 0.9865 | 0.9981 | **+9.0%** | 39.5% | **25.7%** / 18.9% | 0.0889 / 0.1463 |
| ECSSD 995 | u2netp | 0.9703 | 0.9839 | 0.9959 | −72.1% | 35.1% | 40.0% / 25.1% | 0.0907 / 0.1473 |
| | u2net 全量 | 0.9728 | 0.9850 | 0.9968 | −66.5% | 35.3% | 39.0% / 24.5% | 0.0898 / 0.1477 |
| | RMBG | 0.9769 | **0.9888** | 0.9973 | −91.2% | 42.6% | 35.2% / 20.9% | 0.0950 / 0.1478 |
| DUT-OMRON 5123 | u2netp | 0.9704 | 0.9727 | 0.9965 | +8.9% | 37.4% | 28.7% / 23.4% | 0.1026 / 0.1514 |
| | u2net 全量 | 0.9733 | 0.9745 | 0.9971 | +11.0% | 37.8% | 27.2% / 22.3% | 0.1012 / 0.1511 |
| | RMBG | **0.9830** | 0.9833 | 0.9981 | **+17.1%** | 52.7% | **21.1%** / 16.4% | 0.1107 / 0.1500 |

「提升捕获」= 算法相对居中裁剪吃掉了多少可提升空间（`(预测−居中)/(上界−居中)`），只在两边同尺寸配对的假设下有意义；ECSSD 那一行上界和居中只差 0.0085，这个比率会被放大成 −91%，跨数据集不要横着比，看配对差值。

**注意一个混淆**：RMBG 的显著度图更平滑，`bestCrop` 判定「裁剪还不如保持最大框」的次数明显变多 —— 不裁比例 DUTS 31.6%→39.5%、OMRON 37.4%→**52.7%**。框变大，两侧的绝对保留率都会跟着涨（OMRON 居中 0.9727→0.9833），所以**绝对保留率跨模型不可比**，可比的是同一张图上的配对差值。

## 配对统计（算法保留率 − 居中保留率，n = 全量）

| 数据集 | 模型 | 均值 | 配对 t | 算法更好/打平/居中更好 | σ | 保留率=0 的图 |
| --- | --- | --- | --- | --- | --- | --- |
| DUTS-TE | u2netp | −0.00587 | −7.03 | 637 / 2794 / 1588 | 0.05917 | 2 |
| | u2net 全量 | −0.00488 | −5.80 | 674 / 2840 / 1505 | 0.0596 | **0** |
| | **RMBG** | **−0.00186** | **−2.94** | 570 / 3103 / 1346 | 0.04481 | **0** |
| ECSSD | u2netp | −0.01364 | −9.96 | 113 / 483 / 399 | 0.04318 | 0 |
| | u2net 全量 | −0.01227 | −8.99 | 115 / 484 / 396 | 0.04306 | 0 |
| | RMBG | −0.01191 | −9.64 | 99 / 545 / 351 | 0.03897 | 0 |
| DUT-OMRON | u2netp | −0.00227 | −2.02 | 701 / 3012 / 1410 | 0.08046 | 7 |
| | u2net 全量 | −0.00117 | −1.07 | 683 / 3068 / 1372 | — | 8 |
| | **RMBG** | **−0.00031** | **−0.37** | 516 / 3521 / 1086 | 0.06030 | 4 |

t 值（越接近 0 越好）三个数据集一致地按 u2netp → u2net 全量 → RMBG 单调改善。ECSSD 的均值也改善了一点点，但 t 反而更负 —— 因为方差从 0.0432 掉到 0.0390，同样的负偏被更紧的分布放大了，不是变差。

## 极端个案（保留率 < 0.2）

| 数据集 | u2netp | u2net 全量 | RMBG |
| --- | --- | --- | --- |
| DUTS-TE | 5 张（其中 2 张 =0） | 5 张（0 张 =0） | **0 张** |
| ECSSD | 0 张 | 0 张 | 0 张 |
| DUT-OMRON | 16 张（7 张 =0） | 15 张（8 张 =0） | 6 张（4 张 =0） |
| 差值合计 | −54.7 | −42.7 | **−22.7** |

RMBG 下 DUT-OMRON 剩下的 6 张（保留率 / 居中 / 框内质量占比）：

```
 0.0000  0.0000   0.8213   sun_anewdfmxopooqxfy.jpg   ← 居中裁剪也 0，主体本来就在边上
 0.0000  0.8031   1.0000   sun_bczhwhayjmthmbpk.jpg
 0.0000  0.3956   0.9345   sun_bkcwjiggefhprnao.jpg
 0.0000  0.7675   0.9743   sun_bqnmtlgvsydqtugt.jpg
 0.0031  0.5081   0.9786   sun_aoecovdxkchyctlh.jpg
 0.0728  0.6753   0.7293   sun_acdxxpsnivhovjvv.jpg   ← 唯一一个「框内质量占比」明显偏低的
```

去掉这 6 张后，OMRON 的均值从 −0.00031 变成 **+0.00029**（转正）；DUTS 去掉坏图后仍是 −0.00186（本来就没有坏图）；ECSSD 没有坏图、负值跟尾巴无关。

## 健全性：模型掩膜与标注的 IoU（51 张 = 30 随机 + 21 张旧坏图）

| 集合 | u2netp | u2net 全量 | RMBG |
| --- | --- | --- | --- |
| 30 张随机（中位） | 0.8926 | 0.8874 | **0.9273** |
| 21 张旧坏图（中位） | 0.0000（21/21 <0.5） | 0.0000（18/21 <0.5） | 0.0697（17/21 <0.5） |
| 21 张旧坏图（均值） | 0.0046 | 0.1299 | **0.2338** |

这组数字是这次实验里最有信息量的：随机图上三个模型都跟标注对得上（RMBG 最好），但**在那 21 张旧坏图上，RMBG 也只把均值从 0.005 拉到 0.234，中位数仍然是 0** —— 也就是说一半以上，模型和标注选的依旧不是同一个物体。这从「换模型能不能修好极端个案」的角度给出了明确答案：**能缓解，不能消除；根因在标注口径（单主体掩膜只标一个小物体），不在模型强弱。**

## 复现命令

```powershell
$py='C:\Users\Administrator\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe'
$ev='D:\spider\tools\android-camera\tools\eval'
$env:JAVA_HOME='D:\spider\tools\toolchain\jdk'; $env:ANDROID_HOME='D:\spider\tools\toolchain\sdk'

# 1) 产网格（这里以 RMBG 为例；u2netp/u2net 见 --side/--model）
& $py "$ev\model_grid.py" --manifest <manifest.tsv> --out-dir <out> `
    --model <rmbg14.onnx> --side 1024 --norm imagenet --mean 0.5,0.5,0.5 --std 1,1,1 `
    --threads 8 --progress 1000

# 2) 跑算法（必须 --rerun；汇总行在 app\build\test-results\testDebugUnitTest\TEST-cn.yege.dshcam.EvalHarnessTest.xml 里）
$env:EVAL_MANIFEST='<out>\grids_manifest.tsv'; $env:EVAL_GRID='<out>\grids.bin'
$env:EVAL_OUT='<out>\harness_rmbg-src.tsv'; $env:EVAL_ASPECT='src'
Remove-Item Env:EVAL_GATE,Env:EVAL_GATE_REL,Env:EVAL_ANCHOR -ErrorAction SilentlyContinue
& 'D:\spider\tools\toolchain\gradle-8.10.2\bin\gradle.bat' -p 'D:\spider\tools\android-camera' `
    --offline --console=plain :app:testDebugUnitTest --tests 'cn.yege.dshcam.EvalHarnessTest' --rerun

# 3) 出报告
& $py "$ev\run_eval.py" --manifest <manifest.tsv> --harness $env:EVAL_OUT --label rmbg-src --aspect src `
    --out-json <03-rmbg.json> --out-md <03-rmbg.md>
& $py "$ev\compare_reports.py" --reports '<cmp>\*.json' --labels u2netp-src u2net-src rmbg-src `
    --title '三方对比' --out-md <compare_models.md>
& $py "$ev\tail_scan.py" --label RMBG --manifest <manifest.tsv> --grids-manifest '<out>\grids_manifest.tsv' `
    --grids-bin '<out>\grids.bin' --harness $env:EVAL_OUT
```

本轮产物全在 `D:\spider\.cache\saliency-swap\`：`models\`（三个 onnx）、`rmbg\<ds>\`（网格 + harness）、`cmp\<ds>\0{1,2,3}-*.json`、`compare_models_<ds>.md`、`logs\`（harness/tail/paired 日志）。`iou_check.py`、`probe_onnx.py` 也在该目录下，用来验输入输出规格与 IoU。
`tools/eval/paired.py` 是本轮临时写的配对统计脚本，留在 `.cache` 里，没入库。

## 结论与建议

1. **换模型确实能把 DUTS-TE 从「系统性变差」拉到「基本打平偏好」**：配对均值 −0.0059 → −0.0019，提升捕获首次转正（+9.0%），5 张坏图全消失。
2. **DUT-OMRON 已经打平**（−0.0003，t=−0.37，去掉尾巴还稍微为正）。**ECSSD 修不动**：它的主体大而居中，居中裁剪本来就是强候选，算法差距稳定在 −0.012 左右。
3. **代价决定了这条路暂时只能用在评测侧**：RMBG 1024×1024、176 MB、355~620 ms/张，端上（u2netp 只有 4.6 MB）差着两个数量级。要用在端上，得蒸馏成 512 级别的模型，或者换一个 320 输入但训练数据更好的 backbone —— 这轮没做。
4. **不要加「框内显著度太低就不裁」这类守卫**（上一轮已证）：RMBG 下 DUTS/ECSSD 根本没有坏图，剩下的 6 张 OMRON 坏图里框内质量占比中位 0.90，仍然分不开。
5. **极端个案的根因是标注口径**，不是模型强弱（21 张旧坏图上 RMBG 的 IoU 中位仍是 0）。想真的解决，得换多主体标注的数据集（如 COCO 类）或改评测口径，而不是继续换点图模型。
6. **评分权重没跟着换模型调**：`bestCrop` 现在是 `0.55×三分距 + 0.25×保留率 + 0.20×面积 − 切主体罚项`。RMBG 的图更平滑、不裁比例涨到 52.7%，说明这套权重跟新模型的显著度分布并不匹配；如果真要用新模型，这 4 个系数（以及 `highThreshold=0.6×max`）需要重新标定。

## 局限

- 三个模型都是 PC 上的 CPU 推理，与手机端（NNAPI/GPU、量化）的数值不完全一致；线上 u2netp 的表现是按同口径的 PC 复现评估的。
- DUT-OMRON 的「居中保留率」在 RMBG 下涨到 0.9833、上界 0.9981，指标接近天花板，剩下的差异虽然统计显著但工程意义有限。
- 只换了点图模型，`model_grid.py` 的「掩膜 → 64×48 网格」这一步没变（整幅 min/max 拉伸 + 盒式平均），更细的网格分辨率没试。
