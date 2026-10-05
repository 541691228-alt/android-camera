#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用主体模型把清单里的图片跑成 64x48 显著度网格，打包给 Kotlin harness。

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

换别的显著度模型（2026-10-05 加，全部默认值＝ u2netp 现状，不改默认行为）：
    --side 1024                预处理拉伸边长（u2netp 是 320）
    --norm unit|half|raw       imagenet(默认) 之外的值域；--mean/--std 可改常量
    --layout nhwc              默认 nchw，输出布局
    --channel-order bgr        默认 rgb
    --output-index 1           取第几个输出（默认 0，＝ Kotlin 侧 res.get(0)）
    --output-name d0           直接按名字取输出，优先于 --output-index
    --mask-channel 0           输出带通道维时取哪个通道
    --mask-resize N            输出边长不等于 --side 时插值到 NxN（默认 0＝按 --side）
    --threads N                onnxruntime 线程数（默认 2，＝加这个选项之前的行为）
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
IMAGENET_MEAN = "0.485,0.456,0.406"
IMAGENET_STD = "0.229,0.224,0.225"


def parse_triple(text: str, default):
    if not text:
        return default
    vals = [float(x) for x in text.replace(" ", "").split(",") if x]
    if len(vals) != 3:
        raise SystemExit("要 3 个逗号分隔的数，收到: %r" % text)
    return np.array(vals, dtype=np.float32)


def preprocess(im: Image.Image, side: int = SIDE, norm: str = "imagenet",
               mean=None, std=None, layout: str = "nchw", order: str = "rgb") -> np.ndarray:
    """PIL 图 → (1,3,side,side) float32，与 SaliencyMath.argbToNchw 对齐。
    side/norm/layout/order 都是 2026-10-05 加的开关，默认值与加之前逐字节一致。"""
    mk = mean if mean is not None else MEAN
    sk = std if std is not None else STD
    small = im.convert("RGB").resize((side, side), Image.Resampling.BILINEAR)
    a = np.asarray(small, dtype=np.float32)
    if norm == "raw":
        pass
    elif norm == "unit":
        a = a / 255.0
    elif norm == "half":
        a = a / 127.5 - 1.0
    else:  # imagenet
        a = a / 255.0
        a = (a - mk) / sk
    if order == "bgr":
        a = a[:, :, ::-1]
    if layout == "nhwc":
        return np.ascontiguousarray(a[None, ...], dtype=np.float32)
    return np.ascontiguousarray(np.transpose(a, (2, 0, 1))[None, ...], dtype=np.float32)


def _block_edges(dim: int, n: int) -> np.ndarray:
    """复刻 mask_to_grid 的块边界：e[i]=i*dim//n，再强制严格递增（与逐格循环一致）。"""
    e = np.empty(n + 1, dtype=np.int64)
    e[0] = 0
    for i in range(n):
        e[i + 1] = max(e[i] + 1, (i + 1) * dim // n)
    if e[n] > dim:
        e[n] = dim
    return e


def mask_to_grid(mask: np.ndarray, cols: int = GRID_W, rows: int = GRID_H,
                 side: int = None, fast: bool = None) -> np.ndarray:
    """(side,side) 原始掩膜 → (rows,cols) 0..1 网格，与 SaliencyMath.maskToGrid 对齐。
    side 默认取掩膜自己的边长（默认路径下即 320，与加参数之前一致）。
    fast：默认 64x48（cols*rows<=4096）走逐格循环——保证默认口径逐位不变；
    更高分辨率用 reduceat 矢量化（float32 同精度，加和次序不同，末位可能有差）。"""
    if side is None:
        side = int(mask.shape[0])
    mn = float(mask.min())
    mx = float(mask.max())
    rng = mx - mn
    stretch = rng > 1e-6
    if fast is None:
        fast = cols * rows > 4096
    if fast:
        ye = _block_edges(side, rows)
        xe = _block_edges(side, cols)
        if stretch:
            n = (mask - np.float32(mn)) / np.float32(rng)
        else:
            n = np.zeros(mask.shape, dtype=np.float32)
        acc = np.add.reduceat(n, ye[:-1], axis=0)
        acc = acc / np.diff(ye)[:, None].astype(np.float32)
        acc = np.add.reduceat(acc, xe[:-1], axis=1)
        acc = acc / np.diff(xe)[None, :].astype(np.float32)
        return np.ascontiguousarray(acc, dtype=np.float32)
    out = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        y0 = r * side // rows
        y1 = max(y0 + 1, (r + 1) * side // rows)
        for c in range(cols):
            x0 = c * side // cols
            x1 = max(x0 + 1, (c + 1) * side // cols)
            blk = mask[y0:y1, x0:x1]
            out[r, c] = float(((blk - mn) / rng).mean()) if stretch else 0.0
    return out


def pick_mask(raw, mask_channel: int, mask_resize: int, sigmoid: bool = False) -> np.ndarray:
    """把 onnx 输出压成 2D float32 掩膜；边长不等于目标时双线性插值。
    sigmoid=True 给那些输出还是 logits 的模型（如 BiRefNet_lite，实测值域 -14.7..101.5）。"""
    arr = np.asarray(raw, dtype=np.float32)
    if sigmoid:
        arr = 1.0 / (1.0 + np.exp(-arr))
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        # (C,H,W) 还是 (H,W,C)：小的一维当前导通道判断
        if arr.shape[0] <= 4 and arr.shape[0] < arr.shape[-1]:
            arr = arr[mask_channel]
        else:
            arr = arr[..., mask_channel]
    if arr.ndim != 2:
        raise ValueError("输出压不成 2D，shape=%s" % (np.asarray(raw).shape,))
    if mask_resize and arr.shape != (mask_resize, mask_resize):
        mi = Image.fromarray(arr.astype(np.float32), mode="F")
        arr = np.asarray(mi.resize((mask_resize, mask_resize), Image.Resampling.BILINEAR),
                         dtype=np.float32)
    return arr


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
    # 以下选项的默认值都等于"只有 u2netp 那会儿"的行为
    ap.add_argument("--side", type=int, default=SIDE, help="预处理拉伸边长，默认 %d" % SIDE)
    ap.add_argument("--norm", default="imagenet", choices=["imagenet", "unit", "half", "raw"],
                    help="值域处理，默认 imagenet=(px/255-mean)/std")
    ap.add_argument("--mean", default="", help="norm=imagenet 的 mean，逗号分隔；默认 ImageNet")
    ap.add_argument("--std", default="", help="norm=imagenet 的 std，逗号分隔；默认 ImageNet")
    ap.add_argument("--layout", default="nchw", choices=["nchw", "nhwc"], help="输入布局，默认 nchw")
    ap.add_argument("--channel-order", default="rgb", choices=["rgb", "bgr"], help="默认 rgb")
    ap.add_argument("--output-index", type=int, default=0, help="取第几个输出，默认 0")
    ap.add_argument("--output-name", default="", help="按名字取输出，优先于 --output-index")
    ap.add_argument("--mask-channel", type=int, default=0, help="输出带通道维时取哪个通道，默认 0")
    ap.add_argument("--mask-resize", type=int, default=0,
                    help="输出边长不等于该值时插值到该边长；0＝按 --side 的边长")
    ap.add_argument("--sigmoid", action="store_true",
                    help="输出是 logits 时先过 sigmoid（BiRefNet_lite 实测 -14.7..101.5）")
    ap.add_argument("--threads", type=int, default=2, help="onnxruntime 线程数，默认 2")
    ap.add_argument("--progress", type=int, default=0, help="每 N 张打一行进度，0＝不打")
    # 以下两个是 2026-10-05 为"第 2 条：分辨率"加的，默认值＝原来的 64x48，不改默认行为
    ap.add_argument("--grid-w", type=int, default=GRID_W, help="网格宽，默认 %d" % GRID_W)
    ap.add_argument("--grid-h", type=int, default=GRID_H, help="网格高，默认 %d" % GRID_H)
    ap.add_argument("--grids", default="",
                    help="额外分辨率，如 '128x96,256x192'（主输出仍是 grids.bin/<grid-w>x<grid-h>）")
    args = ap.parse_args()

    try:
        import onnxruntime as ort
    except ImportError:
        print("需要 onnxruntime：python -m pip install onnxruntime")
        return 1

    if not os.path.isfile(args.model):
        print("模型不存在: %s" % os.path.abspath(args.model))
        return 1

    mean = parse_triple(args.mean, MEAN) if args.mean else None
    std = parse_triple(args.std, STD) if args.std else None
    mask_resize = args.mask_resize or args.side

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    bin_path = os.path.join(out_dir, "grids.bin")
    man_path = os.path.join(out_dir, "grids_manifest.tsv")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(args.model, opts, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    out_name = args.output_name or out_names[args.output_index]

    rows = read_manifest(args.manifest)
    if not rows:
        print("清单里没有可用行: %s" % args.manifest)
        return 1

    # 输出目标：主目标沿用 grids.bin/grids_manifest.tsv（--grid-w/--grid-h 定尺寸），
    # --grids 里的额外分辨率写成 grids_<W>x<H>.bin / grids_<W>x<H>_manifest.tsv
    targets = [(args.grid_w, args.grid_h, bin_path, man_path)]
    for spec in (args.grids or "").split(","):
        spec = spec.strip()
        if not spec:
            continue
        gw, gh = (int(x) for x in spec.lower().split("x"))
        targets.append((gw, gh,
                        os.path.join(out_dir, "grids_%dx%d.bin" % (gw, gh)),
                        os.path.join(out_dir, "grids_%dx%d_manifest.tsv" % (gw, gh))))

    peaks = [[] for _ in targets]
    n_ok = 0
    n_fail = 0
    handles = []
    for (_gw, _gh, bp, mp) in targets:
        fb = open(bp, "wb")
        fm = open(mp, "w", encoding="utf-8", newline="\n")
        fm.write("index\timage\tw\th\tpeak\n")
        handles.append((fb, fm))
    try:
        for parts in rows:
            path = parts[0]
            if not os.path.isfile(path):
                print("跳过（文件不存在）: %s" % path)
                n_fail += 1
                continue
            try:
                with Image.open(path) as im:
                    w, h = im.size
                    nchw = preprocess(im, side=args.side, norm=args.norm, mean=mean, std=std,
                                      layout=args.layout, order=args.channel_order)
                # 默认取第 0 个输出：和 Kotlin 侧 res.get(0) 同一口径（u2netp 的 d0 融合图）
                raw = sess.run([out_name], {in_name: nchw})[0]
                mask = pick_mask(raw, args.mask_channel, mask_resize, sigmoid=args.sigmoid)
            except Exception as exc:
                print("跳过（推理失败 %s）: %s" % (type(exc).__name__, path))
                n_fail += 1
                continue

            for ti, (gw, gh, _bp, _mp) in enumerate(targets):
                grid = mask_to_grid(mask, cols=gw, rows=gh, side=mask_resize)
                pk = float(grid.max())
                handles[ti][0].write(grid.astype(">f4").tobytes())
                peaks[ti].append(pk)
                handles[ti][1].write("%d\t%s\t%d\t%d\t%.4f\n" % (n_ok, path, w, h, pk))
            n_ok += 1
            if args.progress and n_ok % args.progress == 0:
                print("  %d/%d" % (n_ok, len(rows)), flush=True)
    finally:
        for fb, fm in handles:
            fb.close()
            fm.close()

    for ti, (gw, gh, bp, mp) in enumerate(targets):
        arr = np.asarray(peaks[ti])
        if arr.size == 0:
            continue
        below = int((arr < GRID_PEAK_GATE).sum())
        print(
            "grids=%d skipped=%d bytes=%d  %dx%d\n"
            "  峰值 min=%.3f 中位=%.3f max=%.3f，低于门槛 %.2f 的有 %d 张（%.1f%%）\n  %s\n  %s"
            % (
                n_ok,
                n_fail,
                int(arr.size) * gw * gh * 4,
                gw,
                gh,
                arr.min(),
                float(np.median(arr)),
                arr.max(),
                GRID_PEAK_GATE,
                below,
                100.0 * below / arr.size,
                bp,
                mp,
            )
        )
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
