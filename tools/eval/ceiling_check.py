"""候选集里的"上限"到底是多少——把"构图合格"这条约束加回去才有意义。

三个层次：
  1. 候选集上限（无约束）：**退化解＝不裁**，src 口径下最大框就是整图，保留率恒等 1.0。所以"保留率"单独谈极限没有意义。
  2. 构图约束上限：在"主体落在三分点上（三分线距离 ≤ T）"的候选里取保留率最高的那个
     —— 这才是"评分函数换成完美的"能拿到的天花板。
  3. 紧上限：每张图用**算法自己那张框的三分线距离**当约束（构图不比算法差），保留率最高能到多少
     —— 最公平的天花板，差值就是评分/规则层的全部余量。
  4. 同尺寸滑窗真值 oracle（已发布报告 0.996097）：需要真值掩膜，只能逼近，是候选生成 / 显著度层的天花板。

用法：
  python ceiling_check.py --grids-manifest ... --grids-bin ... --manifest ... --aspects src 1 0.75
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import metrics                      # noqa: E402
import sim_bestcrop as sim          # noqa: E402
import sweep_weights as sw          # noqa: E402

TS = (0.10, 0.15, 0.20)


def kept_vec(I, total, x0, y0, ww, hh):
    x1 = x0 + ww
    y1 = y0 + hh
    inside = I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0]
    return inside.astype(np.float64) / float(total)


def thirds_vec(cx, cy, x0, y0, ww, hh):
    """候选框里主体质心的三分线距离（与 metrics.centroid_in_crop/thirds_distance 同口径）。"""
    px = np.clip((cx - x0) / np.maximum(ww, 1), 0.0, 1.0)
    py = np.clip((cy - y0) / np.maximum(hh, 1), 0.0, 1.0)
    d = np.empty((len(px), 4), np.float64)
    for k, (tx, ty) in enumerate(sw.THIRDS_PTS):
        d[:, k] = np.hypot(px - tx, py - ty)
    return d.min(axis=1)


def run(aspect, pairs, bin_path, keep, limit=0, cols=sw.GRID_W, rows_n=sw.GRID_H):
    rows = pairs[:limit] if limit else pairs
    acc = {f"orc{t}": [] for t in TS}
    acc.update({"oracle": [], "orc_tight": [], "algo": [], "center": [], "base": [],
                "algo_thirds": [], "adopt": []})
    skip = 0
    with open(bin_path, "rb") as bf:
        for (idx, image, w, h, mp) in rows:
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
            cx, cy = gcx * w, gcy * h
            grid = sw.read_grid(bf, idx, cols, rows_n)
            feat = sim.image_features(grid, w, h, aspect=aspect, keep=keep, cols=cols, rows=rows_n)
            if feat is None:
                l, t, cw, ch, _ = sim.fallback_box(w, h, aspect)
                i_algo = i_base = 0
                adopted = 0
                kr_all = kept_vec(I, total, np.array([l]), np.array([t]),
                                  np.array([cw]), np.array([ch]))
                th_all = thirds_vec(cx, cy, np.array([l]), np.array([t]),
                                    np.array([cw]), np.array([ch]))
                algo_box = (l, t, cw, ch)
            else:
                n = len(feat["ks"])
                L = np.empty(n, np.int64); T = np.empty(n, np.int64)
                Wd = np.empty(n, np.int64); Ht = np.empty(n, np.int64)
                for i in range(n):
                    L[i], T[i], Wd[i], Ht[i] = sim.to_pixels(feat, i, w, h, cols, rows_n)
                x0, y0, ww, hh = sw.clip_box(L, T, Wd, Ht, w, h)
                kr_all = kept_vec(I, total, x0, y0, ww, hh)
                th_all = thirds_vec(cx, cy, x0, y0, ww, hh)
                score = sim.score_candidates(feat, 0.40, 0.60, 0.00)
                i_algo, adopted, i_base, _ = sim.pick(feat, score)
                algo_box = tuple(int(v) for v in (L[i_algo], T[i_algo], Wd[i_algo], Ht[i_algo]))

            acc["oracle"].append(float(kr_all.max()))
            acc["algo"].append(float(kr_all[i_algo]))
            acc["base"].append(float(kr_all[i_base]))
            acc["algo_thirds"].append(float(th_all[i_algo]))
            acc["adopt"].append(int(adopted))
            # 紧上限：构图不比算法差（三分线 ≤ 算法自己那张）的候选里，保留率最高能到多少
            sel = np.flatnonzero(th_all <= th_all[i_algo] + 1e-12)
            acc["orc_tight"].append(float(kr_all[sel].max()) if sel.size else float(kr_all[i_algo]))
            for t in TS:
                sel_t = np.flatnonzero(th_all <= t)
                acc[f"orc{t}"].append(float(kr_all[sel_t].max()) if sel_t.size
                                      else float(kr_all[i_algo]))

            cb = metrics.center_crop(w, h, algo_box[2], algo_box[3])
            x0, y0, ww, hh = sw.clip_box(np.array([cb[0]]), np.array([cb[1]]),
                                         np.array([cb[2]]), np.array([cb[3]]), w, h)
            acc["center"].append(float(kept_vec(I, total, x0, y0, ww, hh)[0]))

    def m(k):
        return float(np.mean(acc[k])) if acc[k] else float("nan")

    r = {"aspect": aspect, "n": len(acc["algo"]), "skip": skip,
         "oracle": m("oracle"), "algo": m("algo"), "center": m("center"), "base": m("base"),
         "algo_thirds": m("algo_thirds"), "adopt": m("adopt"), "orc_tight": m("orc_tight")}
    for t in TS:
        r[f"orc{t}"] = m(f"orc{t}")
    r["gain_tight"] = r["orc_tight"] - r["algo"]
    r["gain10"] = r["orc0.1"] - r["algo"]
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids-manifest", required=True)
    ap.add_argument("--grids-bin", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--aspects", nargs="*", default=["src"])
    ap.add_argument("--keep", type=float, default=0.60)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--grid-w", type=int, default=sw.GRID_W)
    ap.add_argument("--grid-h", type=int, default=sw.GRID_H)
    args = ap.parse_args(argv)

    pairs = sw.load_pairs(args.grids_manifest, args.manifest)
    print(f"图文对 {len(pairs)} 条｜网格 {args.grid_w}x{args.grid_h}")
    print("\n aspect      n  无约束上限  构图<=0.10  紧上限   当前算法   居中   最大框  "
          "算法三分线  紧上限-算法  上限(<=0.10)-算法  采纳率")
    for a in args.aspects:
        asp = None if a == "src" else float(a)
        r = run(asp, pairs, args.grids_bin, args.keep, args.limit, args.grid_w, args.grid_h)
        print(f" {a:>5} {r['n']:>6}   {r['oracle']:.4f}    {r['orc0.1']:.4f}    {r['orc_tight']:.4f}"
              f"   {r['algo']:.4f}  {r['center']:.4f}  {r['base']:.4f}"
              f"     {r['algo_thirds']:.4f}     +{r['gain_tight']:.5f}     +{r['gain10']:.5f}"
              f"         {r['adopt']*100:5.1f}%")
    print("\n读法：")
    print("  · 无约束上限在 src 口径恒为 1.0 —— 最大框就是整图，'不裁'保留率 100%，是退化解。")
    print("  · 紧上限：每张图限定'构图不比算法差'，候选里保留率最高的那个 —— 最公平的天花板。")
    print("  · 紧上限 - 算法 = 评分/规则层还剩多少（理论上限，需要完美先验才够得着）。")
    print("  · 同尺寸滑窗真值 oracle = 0.996097（已发布报告）：那是显著度/候选层的天花板。")


if __name__ == "__main__":
    main()
