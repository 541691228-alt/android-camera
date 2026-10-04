#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 u2netp 主体模型把清单里的图片跑成 64x48 显著度网格，打包给 Kotlin harness。

为什么要有这一步：线上自动构图优先用的是 u2netp 的输出（SubjectModel），规则算法
（AutoFrame.salienceFromArgb）只是模型不可用时的兜底。两套显著图的性格完全不同
（模型给的是"主体掩膜"，规则给的是"中心-周边对比"），同一个 bestCrop 喂不同的图，
结果可能不一样 —— 所以评测要能分别跑这两条路。

预处理/后处理逐行照着 app/src/main/java/cn/yege/dshcam/SaliencyMath.kt 复刻：
- argbToNchw：拉伸到 320x320（双线性），RGB 顺序，除以 255 后按 ImageNet 的
  mean=(0.485,0.456,0.406) / std=(0.229,0.224,0.225) 标准化，通道整块在前（NCHW）。
- maskToGrid：先按整幅最小/最大值线性拉伸到 0..1（模型输出值域每次都不一样），
  再按格子面积做盒式平均降到 64x48。块边界用整数除法（和 Kotlin 的 Int 除法一致），
  所以 320/48 不是整除非均匀切块这点也对得上。
- peak：网格最大值，线上拿它和 GRID_PEAK_GATE=0.75 比，判"画面里到底有没有主体"。

用法：
    python model_grid.py --manifest work/manifest.tsv --out-dir work \\
        --model ../app/src/main/assets/u2netp.onnx
产出：
    work/grids.bin            200 张连续拼接的 64x48 float32（大端，每张 12288 字节）
    work/grids_manifest.tsv   表头 index<TAB>image<TAB>w<TAB>h<TAB>peak
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SIDE = 320
GRID_W = 64
GRID_H = 48
GRID_PEAK_GATE = 0.75
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(im: Image.Image) -> np.ndarray:
    """PIL 图 → (1,3,320,320) float32，与 SaliencyMath.argbToNchw 对齐。"""
    small = im.convert("RGB").resize((SIDE, SIDE), Image.Resampling.BILINEAR)
    a = np.asarray(small, dtype=np.float32) / 255.0
    a = (a - MEAN) / STD
    return np.ascontiguousarray(np.transpose(a, (2, 0, 1))[None, ...], dtype=np.float32)


def mask_to_grid(mask: np.ndarray, cols: int = GRID_W, rows: int = GRID_H) -> np.ndarray:
    """(320,320) 原始掩膜 → (rows,cols) 0..1 网格，与 SaliencyMath.maskToGrid 对齐。"""
    mn = float(mask.min())
    mx = float(mask.max())
    rng = mx - mn
    stretch = rng > 1e-6
    out = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        y0 = r * SIDE // rows
        y1 = max(y0 + 1, (r + 1) * SIDE // rows)
        for c in range(cols):
            x0 = c * SIDE // cols
            x1 = max(x0 + 1, (c + 1) * SIDE // cols)
            blk = mask[y0:y1, x0:x1]
            out[r, c] = float(((blk - mn) / rng).mean()) if stretch else 0.0
    return out


def read_manifest(path: str):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip().lstrip("\ufeff")
            if not line or line.startswith("#"):
                continue
            if lineno == 1 and line.split("\t")[0].strip().lower() in ("image", "index"):
                continue
            parts = line.split("\t")
            if parts[0].strip():
                rows.append([p.strip() for p in parts])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join("work", "manifest.tsv"))
    ap.add_argument("--out-dir", default="work")
    ap.add_argument("--model", default=os.path.join("app", "src", "main", "assets", "u2netp.onnx"))
    args = ap.parse_args()

    try:
        import onnxruntime as ort
    except ImportError:
        print("需要 onnxruntime：python -m pip install onnxruntime")
        return 1

    if not os.path.isfile(args.model):
        print("模型不存在: %s" % os.path.abspath(args.model))
        return 1

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    bin_path = os.path.join(out_dir, "grids.bin")
    man_path = os.path.join(out_dir, "grids_manifest.tsv")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(args.model, opts, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]

    rows = read_manifest(args.manifest)
    if not rows:
        print("清单里没有可用行: %s" % args.manifest)
        return 1

    peaks = []
    n_ok = 0
    n_fail = 0
    with open(bin_path, "wb") as fb, open(man_path, "w", encoding="utf-8", newline="\n") as fm:
        fm.write("index\timage\tw\th\tpeak\n")
        for parts in rows:
            path = parts[0]
            if not os.path.isfile(path):
                print("跳过（文件不存在）: %s" % path)
                n_fail += 1
                continue
            try:
                with Image.open(path) as im:
                    w, h = im.size
                    nchw = preprocess(im)
                # 取第 0 个输出：和 Kotlin 侧 res.get(0) 同一口径（u2netp 的 d0 融合图）
                raw = sess.run([out_names[0]], {in_name: nchw})[0]
                mask = np.asarray(raw, dtype=np.float32).reshape(SIDE, SIDE)
            except Exception as exc:
                print("跳过（推理失败 %s）: %s" % (type(exc).__name__, path))
                n_fail += 1
                continue

            grid = mask_to_grid(mask)
            fb.write(grid.astype(">f4").tobytes())
            peak = float(grid.max())
            peaks.append(peak)
            fm.write("%d\t%s\t%d\t%d\t%.4f\n" % (n_ok, path, w, h, peak))
            n_ok += 1

    if peaks:
        arr = np.asarray(peaks)
        below = int((arr < GRID_PEAK_GATE).sum())
        print(
            "grids=%d skipped=%d bytes=%d\n"
            "  峰值 min=%.3f 中位=%.3f max=%.3f，低于门槛 %.2f 的有 %d 张（%.1f%%）\n  %s\n  %s"
            % (
                n_ok,
                n_fail,
                n_ok * GRID_W * GRID_H * 4,
                arr.min(),
                float(np.median(arr)),
                arr.max(),
                GRID_PEAK_GATE,
                below,
                100.0 * below / len(arr),
                bin_path,
                man_path,
            )
        )
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
