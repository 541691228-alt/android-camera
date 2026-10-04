# -*- coding: utf-8 -*-
"""尾巴体检：找出"主体被整个切掉"的极端个案，并检查有没有运行时可见的守卫能拦住它们。

背景：全量评测里有一小撮图，算法保留率接近 0（主体被切到框外），居中裁剪反而能全留住。
我原来的想法是加一道保险 —— "框内显著度总量太低就退回不裁"。这个脚本就是拿来验证它的：
对每张图算若干**端上运行时能拿到**的统计量（u2netp 网格的框内质量占比、同尺寸位置最优性、
重要格子占比、blob 大小、峰值锐度，以及 subjectCenter 的 strength/spread、裁剪面积），
然后看坏图和正常图在这些量上分不分得开、扫阈值时"退掉多少张正常裁剪才换回一张坏图"。

结论（2026-10-05 实测：DUTS-TE 5019 + ECSSD 995 + DUT-OMRON 5123）：**拦不住**。
坏图框内质量占比中位 0.9689（正常图 0.9993），即模型自认为把显著度几乎全留下了；
要抓到全部 21 张坏图，任何一个量都得把 11000+ 张正常图一起退掉。
根因是模型与标注选了**不同的物体**（常常两者都算"画面里最显眼的东西"），不是搜框逻辑错，
所以这道保险不该加进 AutoFrame —— 该修的是上游（多主体标注 / 换显著度模型）。

用法：
    python tail_scan.py --label DUTS-TE \
        --manifest <work>/manifest.tsv \
        --grids-manifest <work>/grids_manifest.tsv \
        --grids-bin <work>/grids.bin \
        --harness <work>/harness_full-model-src.tsv

    # 出人眼对照图（原图 + 红框=算法框 + 红色半透明=GT 掩膜），确认机制用
    python tail_scan.py ... --sheet sheet.jpg --sheet-n 8
"""
import argparse
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metrics import load_mask, kept_ratio, center_crop  # noqa: E402

GW, GH = 64, 48

# 候选守卫量：全部来自端上运行时能拿到的东西
KEYS = [
    ("mass_frac", "框内显著度质量 / 全图", False),
    ("mass_opt", "框内质量 / 同尺寸最优位置", False),
    ("high_frac", "框内重要格子 / 全图重要格子", False),
    ("blob", "显著度 >=0.5*峰值 的格子占比", False),
    ("sharp", "峰值 / 均值", False),
    ("strength", "subjectCenter 的 strength", False),
    ("spread", "subjectCenter 的 spread", False),
    ("keep", "裁剪框面积占比", False),
]


def read_tsv(path):
    with open(path, encoding="utf-8-sig") as f:
        head = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == len(head):
                yield dict(zip(head, parts))


def max_window_sum(g, wk, hk):
    """同尺寸窗口的最大质量和（积分图），用来算"这个框的位置是不是最优的"。"""
    I = np.zeros((GH + 1, GW + 1), dtype=np.float64)
    I[1:, 1:] = np.cumsum(np.cumsum(g.astype(np.float64), axis=1), axis=0)
    s = (I[hk:, wk:] - I[:GH - hk + 1, wk:] - I[hk:, :GW - wk + 1]
         + I[:GH - hk + 1, :GW - wk + 1])
    return float(s.max())


def load_rows(args):
    grids = np.fromfile(args.grids_bin, dtype=">f4").reshape(-1, GH, GW)
    idx = {r["image"]: int(r["index"]) for r in read_tsv(args.grids_manifest)}
    gt = {r["image"]: r["mask"] for r in read_tsv(args.manifest)}
    rows = []
    for r in read_tsv(args.harness):
        img = r["image"]
        if img not in idx or img not in gt or r["status"] != "ok":
            continue
        g = grids[idx[img]]
        w, h = int(r["w"]), int(r["h"])
        crop = (int(r["crop_left"]), int(r["crop_top"]), int(r["crop_w"]), int(r["crop_h"]))
        # 像素框 → 网格格子，跟 Kotlin 侧同一套四舍五入
        x0 = max(0, min(GW - 1, int(round(crop[0] * GW / w))))
        y0 = max(0, min(GH - 1, int(round(crop[1] * GH / h))))
        wk = max(1, min(int(round(crop[2] * GW / w)), GW - x0))
        hk = max(1, min(int(round(crop[3] * GH / h)), GH - y0))
        sub = g[y0:y0 + hk, x0:x0 + wk]
        total, mx, mean = float(g.sum()), float(g.max()), float(g.mean())
        inmass = float(sub.sum())
        best = max_window_sum(g, wk, hk)
        thr = 0.6 * mx
        hightot = float(g[g >= thr].sum())
        inhigh = float(sub[sub >= thr].sum())

        mask = load_mask(gt[img])
        kr = kept_ratio(mask, crop)
        ckr = kept_ratio(mask, center_crop(w, h, crop[2], crop[3]))
        if kr is None or ckr is None:
            continue
        rows.append(dict(
            image=os.path.basename(img), path=img, crop=crop, kr=kr, center_kr=ckr,
            delta=kr - ckr, skipped=int(r["skipped"]),
            mass_frac=inmass / total if total > 0 else 0.0,
            mass_opt=inmass / best if best > 0 else 0.0,
            high_frac=inhigh / hightot if hightot > 0 else 0.0,
            blob=float((g >= 0.5 * mx).mean()),
            sharp=(mx / mean if mean > 0 else 0.0),
            strength=float(r["subject_strength"]) if r["subject_strength"] else 0.0,
            spread=float(r["subject_spread"]) if r["subject_spread"] else 0.0,
            keep=float(r["keep_area"]),
        ))
    return rows


def report(args, rows):
    bad = [r for r in rows if r["kr"] < args.bad_below]
    good = [r for r in rows if r["kr"] >= args.bad_below]
    tot = sum(r["delta"] for r in rows)
    badsum = sum(r["delta"] for r in bad)
    restsum = sum(r["delta"] for r in good)
    print(f"===== {args.label}  图数 {len(rows)}  保留率 <{args.bad_below} 的坏图 {len(bad)} 张"
          f"（=0 的 {sum(1 for r in rows if r['kr'] == 0)} 张）=====")
    print(f"算法 − 居中 的差值：合计 {tot:+.1f}，均值 {tot / len(rows):+.5f}")
    print(f"  其中坏图 {len(bad)} 张贡献 {badsum:+.1f}，其余 {len(good)} 张贡献 {restsum:+.1f}"
          f"  → 去掉坏图后均值 {restsum / max(1, len(good)):+.5f}")

    if bad:
        print(f"\n-- 坏图明细（按保留率升序，前 {args.top} 张）--")
        print(f"{'保留率':>7} {'居中':>7} {'质量占比':>8} {'位置最优':>8} {'重要格':>7} {'面积':>6} "
              f"{'strength':>8} {'spread':>7}  图")
        for r in sorted(bad, key=lambda r: r["kr"])[:args.top]:
            print(f"{r['kr']:7.4f} {r['center_kr']:7.4f} {r['mass_frac']:8.4f} {r['mass_opt']:8.4f} "
                  f"{r['high_frac']:7.4f} {r['keep']:6.3f} {r['strength']:8.4f} {r['spread']:7.4f}  {r['image']}")

    print(f"\n-- 坏图 vs 正常图（分位）--")
    if not bad:
        print("（这个数据集没有坏图，守卫扫描里「抓到坏图」必然是 0）")
    else:
        print(f"{'量':>10} {'坏图中位':>9} {'坏图p10':>9} {'正常中位':>9} {'正常p10':>9} {'正常p90':>9}")
        for k, _desc, _ in KEYS:
            b = np.array([r[k] for r in bad])
            gd = np.array([r[k] for r in good])
            print(f"{k:>10} {np.median(b):9.4f} {np.percentile(b, 10):9.4f} {np.median(gd):9.4f} "
                  f"{np.percentile(gd, 10):9.4f} {np.percentile(gd, 90):9.4f}")

    # 守卫扫描：只统计"真的动了剪刀"的行（skipped=1 表示面积够大不另存，退回不裁等于没变化）
    live = [r for r in rows if r["skipped"] == 0]
    live_bad = [r for r in live if r["kr"] < args.bad_below]
    print(f"\n-- 假想守卫：量 <X 或 >X 就退回不裁（只算真的裁过、面积 <97% 的 {len(live)} 张；"
          f"其中坏图 {len(live_bad)} 张）--")
    print(f"{'量':>10} {'方向':>4} {'阈值':>8} {'退回':>6} {'抓到坏图':>8} {'误伤正常':>8} {'误伤率':>7}")
    for k, _desc, _ in KEYS:
        b = np.array([r[k] for r in live_bad]) if live_bad else np.array([])
        gd = np.array([r[k] for r in live if r["kr"] >= args.bad_below])
        vals = np.concatenate([b, gd])
        for direction in ("low", "high"):
            for x in np.percentile(vals, [1, 2, 5, 10, 20]):
                rb = (b < x) if direction == "low" else (b > x)
                rg = (gd < x) if direction == "low" else (gd > x)
                if rb.sum() + rg.sum() == 0:
                    continue
                prec = rb.sum() / (rb.sum() + rg.sum())
                print(f"{k:>10} {direction:>4} {x:8.4f} {rb.sum() + rg.sum():6d} {rb.sum():8d} "
                      f"{rg.sum():8d} {prec:7.1%}")
    print("\n判读：抓到的坏图越多，误伤的正常裁剪也越多；要抓全 21 张，任何量都得退掉 11000+ 张。"
          "→ 这个方向的保险不成立。")


def make_sheet(args, rows):
    """人眼确认机制：原图 + 红框=算法框 + 红色半透明=GT 掩膜（左上角是文件名序号）。"""
    from PIL import Image, ImageDraw
    bad = sorted([r for r in rows if r["kr"] < args.bad_below], key=lambda r: r["kr"])[:args.sheet_n]
    if not bad:
        print("没有坏图，跳过拼图")
        return
    tiles = []
    for r in bad:
        img_path = r["path"]
        # 掩膜路径：manifest 里成对给出，这里按扩展名换目录约定不了，用 manifest 反查
        mask_path = MASK_OF.get(img_path)
        if not mask_path or not os.path.isfile(mask_path):
            continue
        im = Image.open(img_path).convert("RGB")
        mk = Image.open(mask_path).convert("L")
        tint = Image.new("RGB", im.size, (255, 0, 0))
        ann = Image.composite(tint, im, mk.point(lambda v: 110 if v >= 128 else 0))
        d = ImageDraw.Draw(ann)
        l, t, cw, ch = r["crop"]
        d.rectangle([l, t, l + cw - 1, t + ch - 1], outline=(0, 0, 255), width=3)
        tiles.append(ann.resize((360, 270)))
    if not tiles:
        print("没找到掩膜文件，跳过拼图")
        return
    cols = 2
    rows_n = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (360 * cols, 270 * rows_n), (20, 20, 20))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * 360, (i // cols) * 270))
    sheet.save(args.sheet, quality=90)
    print(f"\n写出对照图 {args.sheet}（蓝框=算法裁剪框，红色半透明=GT 主体，顺序＝坏图升序前 {len(tiles)} 张）")


MASK_OF = {}


def main():
    ap = argparse.ArgumentParser(description="极端个案体检：坏图统计 + 假想守卫扫描（见文件头说明）")
    ap.add_argument("--label", default="dataset", help="报告里用的数据集名")
    ap.add_argument("--manifest", required=True, help="image<TAB>mask 清单（给 GT 掩膜用）")
    ap.add_argument("--grids-manifest", required=True, help="index<TAB>image<TAB>w<TAB>h<TAB>peak 清单")
    ap.add_argument("--grids-bin", required=True, help="每张 64x48 个大端 float32 的显著度网格文件")
    ap.add_argument("--harness", required=True, help="harness 导出的 TSV")
    ap.add_argument("--bad-below", type=float, default=0.2, help="保留率低于多少算坏图（默认 0.2）")
    ap.add_argument("--top", type=int, default=25, help="坏图明细最多打几行")
    ap.add_argument("--sheet", help="把最惨的几张拼成对照图写到这个路径（需要 Pillow）")
    ap.add_argument("--sheet-n", type=int, default=8, help="对照图放几张")
    args = ap.parse_args()

    for r in read_tsv(args.manifest):
        MASK_OF[r["image"]] = r["mask"]

    rows = load_rows(args)
    if not rows:
        print("没读到任何行，检查四个输入路径是否配套")
        return 1
    report(args, rows)
    if args.sheet:
        make_sheet(args, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
