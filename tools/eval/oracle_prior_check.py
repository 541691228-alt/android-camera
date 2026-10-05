"""给算法一个"完美显著度图"（真值掩膜下采样到 64×48），把瓶颈定位到"先验层"还是"选择规则层"。

三层对比：
  A 现状       ：任务显著度图（64×48）＋ 当前权重 0.40/0.60/0
  B 完美先验   ：真值掩膜下采样到 64×48（空间分辨率不变）＋ 同一套权重
                 → B − A 全部来自"显著度图准不准"
  C 完美先验＋预言机选择（紧上限）：候选里挑"构图不比算法差"且保留率最高的那个
                 → C − B 全部来自"评分/选择规则"（评分函数换成完美的）

再叠一层分辨率消融（把完美先验先在 32×24 / 16×12 上取平均再放大回 64×48）：
  → 量化"64×48 这个网格本身"值多少。

用法：
  python oracle_prior_check.py --grids-manifest ... --grids-bin ... --manifest ... --aspects src 1 0.75
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import metrics                      # noqa: E402
import sim_bestcrop as sim          # noqa: E402
import sweep_weights as sw          # noqa: E402
from ceiling_check import kept_vec, thirds_vec   # noqa: E402


def perfect_grid(mask: np.ndarray, cols: int = 64, rows: int = 48, coarse: int = 1) -> np.ndarray:
    """真值掩膜 → 64×48 的"完美显著度图"（面积平均）。coarse>1 时先在更低分辨率取平均再放大。"""
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    cw, ch = max(1, cols // coarse), max(1, rows // coarse)
    img = img.resize((cw, ch), Image.BOX)
    if (cw, ch) != (cols, rows):
        img = img.resize((cols, rows), Image.NEAREST)
    return (np.asarray(img, dtype=np.float32) / 255.0).reshape(-1)


def eval_case(feat, I, total, cx, cy, w, h, w_thirds, w_ret, w_area,
              cols=64, rows=48):
    """返回 (算法框保留率, 算法框三分线距离, 是否采纳, 紧上限保留率)。"""
    n = len(feat["ks"])
    L = np.empty(n, np.int64); T = np.empty(n, np.int64)
    Wd = np.empty(n, np.int64); Ht = np.empty(n, np.int64)
    for i in range(n):
        L[i], T[i], Wd[i], Ht[i] = sim.to_pixels(feat, i, w, h, cols, rows)
    x0, y0, ww, hh = sw.clip_box(L, T, Wd, Ht, w, h)
    kr = kept_vec(I, total, x0, y0, ww, hh)
    th = thirds_vec(cx, cy, x0, y0, ww, hh)
    score = sim.score_candidates(feat, w_thirds, w_ret, w_area)
    i_algo, adopted, _i_base, _ = sim.pick(feat, score)
    sel = np.flatnonzero(th <= th[i_algo] + 1e-12)
    tight = float(kr[sel].max()) if sel.size else float(kr[i_algo])
    return float(kr[i_algo]), float(th[i_algo]), int(adopted), tight


def run(aspect, pairs, bin_path, keep=0.60, w=(0.40, 0.60, 0.00), limit=0, coarses=(1, 2, 4),
        cols=64, rows=48):
    rows_pairs = pairs[:limit] if limit else pairs
    res = {c: {"a": [], "b": [], "c": [], "b_thirds": [], "b_adopt": []} for c in coarses}
    a_kept, a_thirds, a_adopt = [], [], []
    skip = 0
    with open(bin_path, "rb") as bf:
        for (idx, image, ww_px, hh_px, mp) in rows_pairs:
            try:
                mask = metrics.load_mask(mp)
            except Exception:
                skip += 1
                continue
            if mask is None or not mask.any():
                skip += 1
                continue
            I = sw.mask_integral(mask)
            total = int(mask.sum())
            gcx, gcy = metrics.subject_centroid(mask)
            cx, cy = gcx * ww_px, gcy * hh_px
            grid_a = sw.read_grid(bf, idx, cols, rows)
            feat_a = sim.image_features(grid_a, ww_px, hh_px, aspect=aspect, keep=keep,
                                        cols=cols, rows=rows)
            k, t, ad, _ = eval_case(feat_a, I, total, cx, cy, ww_px, hh_px, *w,
                                    cols=cols, rows=rows)
            a_kept.append(k); a_thirds.append(t); a_adopt.append(ad)
            for c in coarses:
                g = perfect_grid(mask, cols=cols, rows=rows, coarse=c)
                feat = sim.image_features(g, ww_px, hh_px, aspect=aspect, keep=keep,
                                          cols=cols, rows=rows)
                kb, tb, adb, tight = eval_case(feat, I, total, cx, cy, ww_px, hh_px, *w,
                                               cols=cols, rows=rows)
                res[c]["a"].append(k)
                res[c]["b"].append(kb)
                res[c]["c"].append(tight)
                res[c]["b_thirds"].append(tb)
                res[c]["b_adopt"].append(adb)

    def m(v):
        return float(np.mean(v)) if len(v) else float("nan")

    print(f"\n aspect {aspect}  网格 {cols}x{rows}  n={len(a_kept)}  skip={skip}"
          f"   A 采纳率={m(a_adopt)*100:.1f}%")
    print("  | 先验分辨率 | A 现状 | B 完美先验 | C 预言机 | B-A | C-A | A三分线 | B三分线 | B采纳 |")
    print("  |---|---|---|---|---|---|---|---|---|")
    for c in coarses:
        r = res[c]
        print(f"  | {cols//c}x{rows//c} | {m(r['a']):.4f} | {m(r['b']):.4f} | {m(r['c']):.4f} | "
              f"{m(r['b'])-m(r['a']):+.5f} | {m(r['c'])-m(r['a']):+.5f} | {m(a_thirds):.4f} | "
              f"{m(r['b_thirds']):.4f} | {m(r['b_adopt'])*100:.1f}% |")
    return res, a_kept, a_thirds, a_adopt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids-manifest", required=True)
    ap.add_argument("--grids-bin", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--aspects", nargs="*", default=["src"])
    ap.add_argument("--keep", type=float, default=0.60)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--grid-w", type=int, default=64, help="网格宽，默认 64（换分辨率时配合 --grids-bin）")
    ap.add_argument("--grid-h", type=int, default=48, help="网格高，默认 48")
    args = ap.parse_args(argv)

    pairs = sw.load_pairs(args.grids_manifest, args.manifest)
    print(f"图文对 {len(pairs)} 条｜当前权重 0.40/0.60/0.00｜网格 {args.grid_w}x{args.grid_h}")
    for a in args.aspects:
        asp = None if a == "src" else float(a)
        coarses = (1, 2, 4) if a == "src" else (1,)
        run(asp, pairs, args.grids_bin, args.keep, limit=args.limit, coarses=coarses,
            cols=args.grid_w, rows=args.grid_h)
    print("\n读法：")
    print("  · B-A ＝ 换成完美显著度图（分辨率不变）能拿回多少 → 显著度模型层的价值")
    print("  · C-A ＝ 整条选择链换成预言机（要真值掩膜）能拿回多少 → 本轮所有余量的上界")
    print(f"  · 低分辨率行 ≈ {args.grid_w}x{args.grid_h} 这个网格本身值多少（完美内容、粗网格）")


if __name__ == "__main__":
    main()
