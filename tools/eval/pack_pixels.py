#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把清单里的图片解码并缩到 64x48，打包成 Kotlin harness 能直接读的裸 ARGB 像素文件。

为什么要多这一步：Android 模块的单元测试是拿 android.jar 当编译期 JDK 的，
里面根本没有 javax.imageio / java.awt，harness 在 Kotlin 侧读不了 JPEG。
所以解码和缩放放在这里（Pillow），Kotlin 侧只拿像素跑算法，两边职责反而更清楚：
- 缩放用 Pillow 的 BILINEAR：大比例缩小时它按缩放倍数放大滤波核，效果接近面积平均，
  比端上 Bitmap.createScaledBitmap(filter=true) 的差异在工程上可以忽略（README 里有说明）。
- 像素写成大端 int32（每张 64*48*4 = 12288 字节，行优先），对应 Java 的 RandomAccessFile.readInt。

用法：
    python pack_pixels.py --manifest manifest.tsv --out-dir work
产出：
    work/pixels.bin            200 张连续拼接的 ARGB
    work/pixels_manifest.tsv   表头 index<TAB>image<TAB>w<TAB>h，index 从 0 开始，行序同输入
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image

# Windows 控制台默认 GBK，中文提示会直接抛 UnicodeEncodeError 把整批打断
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

GRID_W = 64
GRID_H = 48
PIXELS_PER_IMAGE = GRID_W * GRID_H


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join("work", "manifest.tsv"))
    ap.add_argument("--out-dir", default="work")
    ap.add_argument("--grid-w", type=int, default=GRID_W)
    ap.add_argument("--grid-h", type=int, default=GRID_H)
    args = ap.parse_args()

    manifest = os.path.abspath(args.manifest)
    out_dir = os.path.abspath(args.out_dir)
    if not os.path.isfile(manifest):
        print("manifest 不存在: %s" % manifest)
        return 1
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    # utf-8-sig：Windows 上拿记事本/PowerShell 存的清单常带 BOM，带进路径就找不到文件
    with open(manifest, "r", encoding="utf-8-sig") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip().lstrip("\ufeff")
            if not line or line.startswith("#"):
                continue
            if lineno == 1 and line.split("\t")[0].strip().lower() == "image":
                continue
            parts = line.split("\t")
            if not parts[0].strip():
                continue
            rows.append(parts[0].strip())

    bin_path = os.path.join(out_dir, "pixels.bin")
    man_path = os.path.join(out_dir, "pixels_manifest.tsv")

    n_ok = 0
    n_fail = 0
    with open(bin_path, "wb") as fb, open(man_path, "w", encoding="utf-8", newline="\n") as fm:
        fm.write("index\timage\tw\th\n")
        for path in rows:
            if not os.path.isfile(path):
                print("跳过（文件不存在）: %s" % path)
                n_fail += 1
                continue
            try:
                with Image.open(path) as im:
                    w, h = im.size
                    # 端上这个链路不读 EXIF 方向，这里也不转，保持同一坐标系
                    small = im.convert("RGB").resize(
                        (args.grid_w, args.grid_h), Image.Resampling.BILINEAR
                    )
                    rgb = np.asarray(small, dtype=np.uint8)
            except Exception as exc:  # 坏图不该打断整批
                print("跳过（解码失败 %s）: %s" % (type(exc).__name__, path))
                n_fail += 1
                continue

            if w < args.grid_w or h < args.grid_h:
                print("跳过（小于 %dx%d）: %s" % (args.grid_w, args.grid_h, path))
                n_fail += 1
                continue

            # RGB -> 0xAARRGGBB，alpha 固定不透明（JPEG 本来就没有透明通道）
            argb = (
                (np.uint32(0xFF) << 24)
                | (rgb[:, :, 0].astype(np.uint32) << 16)
                | (rgb[:, :, 1].astype(np.uint32) << 8)
                | rgb[:, :, 2].astype(np.uint32)
            ).reshape(-1)
            fb.write(argb.astype(">i4").tobytes())
            fm.write("%d\t%s\t%d\t%d\n" % (n_ok, path, w, h))
            n_ok += 1

    print(
        "packed=%d skipped=%d bytes=%d\n  %s\n  %s"
        % (n_ok, n_fail, n_ok * PIXELS_PER_IMAGE * 4, bin_path, man_path)
    )
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
