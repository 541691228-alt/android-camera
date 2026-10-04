#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 harness 跑出来的裁剪框导成图片，给美学模型或人眼比较用。

harness 只输出坐标（`work/harness_*.tsv`），要拿"算法裁的图"和"居中裁的图"去问
美学模型哪张更好看，先得把框落成文件。这个脚本做这件事：同一张原图，按算法框裁一份、
按同尺寸居中框裁一份，再拼几张接触印相（contact sheet）方便一眼扫过去。

    python tools/eval/export_crops.py --manifest work/manifest.tsv \
        --harness work/harness_model_src.tsv --out-dir work/crops

产物：
    <out>/pred/0001.jpg      算法框
    <out>/center/0001.jpg    同尺寸居中框
    <out>/pairs.tsv          两两对应的路径与坐标
    <out>/sheets/sheet_0001.jpg  缩略图拼版，绿框=算法，灰框=居中
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import center_crop  # 同一份居中框实现，别在两边各写一遍

CELL_W = 320          # 拼版里每格缩略图宽度
SHEET_COLS = 4        # 拼版列数
SHEET_ROWS = 5        # 拼版行数（一版 20 张）
GAP = 8               # 拼版格间距
LABEL_H = 18          # 拼版每格底部的文件名条高度


def read_tsv(path: Path) -> list[dict]:
    """读 TSV，自动剥 BOM；返回 list[dict]。"""
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def read_manifest(path: Path) -> dict[str, str]:
    """image 路径 → 掩膜路径（只取第一列做键）。"""
    out: dict[str, str] = {}
    for row in read_tsv(path):
        keys = list(row.keys())
        img = (row.get("image") or "").strip()
        if not img:
            img = (row.get(keys[0]) or "").strip()
        if img:
            mask = (row.get("mask") or "").strip()
            out[img] = mask
    return out


def clip_box(left: int, top: int, w: int, h: int, iw: int, ih: int):
    """把框夹进图像范围，返回 (left, top, right, bottom)，无效返回 None。"""
    x0 = max(0, min(left, iw))
    y0 = max(0, min(top, ih))
    x1 = max(0, min(left + w, iw))
    y1 = max(0, min(top + h, ih))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return x0, y0, x1, y1


def make_sheet(items: list[tuple[str, Path, Path]], dst: Path) -> None:
    """items = [(标签, 算法图路径, 居中图路径)]，每行一格、格内左右并排。"""
    cell_h = int(CELL_W * 0.75)
    cols = SHEET_COLS
    rows = (len(items) + cols - 1) // cols
    sheet_w = cols * (CELL_W + GAP) + GAP
    sheet_h = rows * (cell_h + LABEL_H + GAP) + GAP
    sheet = Image.new("RGB", (sheet_w, sheet_h), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    half = (CELL_W - 2) // 2
    for i, (label, pred_p, center_p) in enumerate(items):
        cx = GAP + (i % cols) * (CELL_W + GAP)
        cy = GAP + (i // cols) * (cell_h + LABEL_H + GAP)
        for j, (p, color) in enumerate(((pred_p, (0, 200, 0)), (center_p, (150, 150, 150)))):
            try:
                im = Image.open(p).convert("RGB")
            except Exception:
                continue
            im.thumbnail((half, cell_h))
            ox = cx + j * (half + 2) + (half - im.width) // 2
            oy = cy + (cell_h - im.height) // 2
            sheet.paste(im, (ox, oy))
            draw.rectangle([ox - 2, oy - 2, ox + im.width + 1, oy + im.height + 1], outline=color, width=2)
        draw.text((cx + 2, cy + cell_h + 3), label, fill=(220, 220, 220))
    dst.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dst, quality=88, optimize=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="把 harness 的裁剪框导成图片")
    ap.add_argument("--manifest", default="work/manifest.tsv", help="带 mask 的清单（只用来校验图片路径）")
    ap.add_argument("--harness", default="work/harness_model_src.tsv", help="harness 输出的 TSV")
    ap.add_argument("--out-dir", default="work/crops", help="输出目录")
    ap.add_argument("--quality", type=int, default=92, help="JPEG 质量")
    ap.add_argument("--limit", type=int, default=0, help="只导前 N 张，0=全部")
    ap.add_argument("--only-cropped", action="store_true",
                    help="只导算法真的动过剪刀的图（skipped=0）；不裁的图上 pred 与 center 是同一张，做偏好测试是噪声")
    ap.add_argument("--sheet", type=int, default=1, help="是否出拼版，0=不出")
    args = ap.parse_args(argv)

    manifest = read_manifest(Path(args.manifest))
    rows = read_tsv(Path(args.harness))
    if not rows:
        print("harness 文件没有数据行", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    pred_dir = out_dir / "pred"
    center_dir = out_dir / "center"
    pred_dir.mkdir(parents=True, exist_ok=True)
    center_dir.mkdir(parents=True, exist_ok=True)

    pairs: list[list[str]] = []
    sheet_items: list[tuple[str, Path, Path]] = []
    n_ok = n_err = n_size_mismatch = n_skipped_alg = n_identical = 0

    for row in rows:
        status = (row.get("status") or "").strip()
        img_path = (row.get("image") or "").strip()
        if status != "ok" or not img_path:
            n_err += 1
            continue
        if args.limit and n_ok >= args.limit:
            break
        if img_path not in manifest:
            print(f"清单里没有这张图，跳过：{img_path}", file=sys.stderr)
            n_err += 1
            continue

        try:
            cw = int(row["crop_w"])
            ch = int(row["crop_h"])
            cl = int(row["crop_left"])
            ct = int(row["crop_top"])
        except (KeyError, ValueError):
            n_err += 1
            continue
        if cw <= 0 or ch <= 0:
            n_err += 1
            continue
        if args.only_cropped and (row.get("skipped") or "0").strip().lower() in ("1", "true"):
            n_skipped_alg += 1
            continue

        src = Path(img_path)
        if not src.exists():
            print(f"图片不存在，跳过：{src}", file=sys.stderr)
            n_err += 1
            continue

        stem = f"{n_ok + 1:04d}"
        pred_p = pred_dir / f"{stem}.jpg"
        center_p = center_dir / f"{stem}.jpg"
        try:
            with Image.open(src) as im0:
                im = im0.convert("RGB")
                iw, ih = im.width, im.height
                want_w = int(row.get("w") or iw)
                want_h = int(row.get("h") or ih)
                if (want_w, want_h) != (iw, ih):
                    n_size_mismatch += 1
                pbox = clip_box(cl, ct, cw, ch, iw, ih)
                cl2, ct2, cw2, ch2 = center_crop(iw, ih, cw, ch)
                cbox = clip_box(cl2, ct2, cw2, ch2, iw, ih)
                if pbox is None or cbox is None:
                    n_err += 1
                    continue
                im.crop(pbox).save(pred_p, quality=args.quality, optimize=True)
                im.crop(cbox).save(center_p, quality=args.quality, optimize=True)
        except Exception as exc:  # 单张失败不拖垮整批
            print(f"导出失败 {src}: {type(exc).__name__}: {exc}", file=sys.stderr)
            n_err += 1
            continue

        same = 1 if pbox == cbox else 0
        n_identical += same
        pairs.append([
            stem, img_path, str(pred_p), str(center_p),
            str(pbox[0]), str(pbox[1]), str(pbox[2] - pbox[0]), str(pbox[3] - pbox[1]),
            str(cbox[0]), str(cbox[1]), str(cbox[2] - cbox[0]), str(cbox[3] - cbox[1]),
            str(same),
        ])
        sheet_items.append((stem, pred_p, center_p))
        n_ok += 1

    pairs_path = out_dir / "pairs.tsv"
    with pairs_path.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh, delimiter="\t", lineterminator="\n")
        wr.writerow([
            "index", "image", "pred", "center",
            "pred_left", "pred_top", "pred_w", "pred_h",
            "center_left", "center_top", "center_w", "center_h",
            "identical",
        ])
        wr.writerows(pairs)

    n_sheets = 0
    if args.sheet and sheet_items:
        per = SHEET_COLS * SHEET_ROWS
        for k in range(0, len(sheet_items), per):
            n_sheets += 1
            make_sheet(sheet_items[k:k + per], out_dir / "sheets" / f"sheet_{n_sheets:04d}.jpg")

    print(f"导出 {n_ok} 对（pred/center），失败 {n_err}，尺寸不一致 {n_size_mismatch}，"
          f"两张一样的 {n_identical}，因算法没裁而跳过 {n_skipped_alg}，拼版 {n_sheets} 张")
    print(f"输出目录 {out_dir}  对应表 {pairs_path}")
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
