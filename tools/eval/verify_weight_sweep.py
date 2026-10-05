"""对 RESULTS-weight-sweep.md 的结论做独立复验（不依赖 sweep_weights.py 的聚合代码）。

复验项：
  A 用 sim_bestcrop 的 predict 输出 harness 格式 TSV，交给已发布的 run_eval.py 重新聚合
  B 配对 bootstrap 95% 置信区间 + 符号检验（不依赖正态假设）
  C 3 折交叉验证：折内选最优权重，看折外是否还成立（查选择偏倚/过拟合）
  D 换一个"居中"对照口径：居中框取该比例下的最大框，而不是跟算法框同尺寸
  E 消融：分开动"面积项"和"三分线:保留率比例"，确认哪个是因

用法：
  python verify_weight_sweep.py --out-dir D:\\spider\\.cache\\eval-duts\\verify
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import metrics  # noqa: E402
import sim_bestcrop as sb  # noqa: E402

DUTS = Path(r"D:\spider\.cache\eval-duts\work-full")
REC = (0.40, 0.60, 0.00)
BASE = (0.55, 0.25, 0.20)
GRID_COLS, GRID_ROWS = 64, 48
PIXEL_BYTES = GRID_COLS * GRID_ROWS * 4
TSV_HEADER = ("image\tw\th\tcrop_left\tcrop_top\tcrop_w\tcrop_h\tkeep_area\t"
              "skipped\tsubject_found\tstatus\n")


def read_grids_manifest(path: Path):
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        header = f.readline().rstrip("\n").split("\t")
        idx_i, img_i, w_i, h_i = (header.index("index"), header.index("image"),
                                  header.index("w"), header.index("h"))
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(idx_i, img_i, w_i, h_i):
                continue
            rows.append((int(parts[idx_i]), parts[img_i], int(parts[w_i]), int(parts[h_i])))
    return rows


def read_mask_manifest(path: Path):
    m = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        header = f.readline().rstrip("\n").split("\t")
        img_i, mask_i = header.index("image"), header.index("mask")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > max(img_i, mask_i):
                m[parts[img_i]] = parts[mask_i]
    return m


def bootstrap_ci(d, nboot, seed=7):
    rng = np.random.default_rng(seed)
    m = len(d)
    idx = rng.integers(0, m, size=(nboot, m))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_test_z(d):
    wins = int((d > 1e-9).sum())
    losses = int((d < -1e-9).sum())
    m = wins + losses
    z = (wins - m / 2) / np.sqrt(m / 4) if m else 0.0
    return wins, losses, float(z)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--duts", default=str(DUTS))
    ap.add_argument("--out-dir", default=r"D:\spider\.cache\eval-duts\verify")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 张（调试用）")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--aspects", nargs="*", default=["src", "1", "0.75"])
    args = ap.parse_args(argv)

    ds = Path(args.duts)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    grid_rows = read_grids_manifest(ds / "grids_manifest.tsv")
    if args.limit:
        grid_rows = grid_rows[: args.limit]
    masks = read_mask_manifest(ds / "manifest.tsv")
    grids = sb.load_grids(ds / "grids.bin", 0)
    print(f"图 {len(grid_rows)} 张，掩膜 {len(masks)} 条")

    # 权重候选：面积项归零族 + 基线 + 消融项
    cands = [(round(t, 2), round(1.0 - t, 2), 0.00)
             for t in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55)]
    cands += [(0.55, 0.25, 0.20), (0.55, 0.25, 0.00), (0.25, 0.55, 0.20)]
    cands = list(dict.fromkeys(cands))

    for aspect in args.aspects:
        asp = None if aspect == "src" else float(aspect)
        tag = "src" if asp is None else aspect.replace(".", "")

        recs = []        # (w, h, mask, feat, boxes)
        meta = []        # (index, image, w, h)
        for idx, img, w, h in grid_rows:
            mask_path = masks.get(img)
            if not mask_path:
                continue
            try:
                mask = metrics.load_mask(mask_path)
            except Exception:
                continue
            if not mask.any():
                continue
            grid = np.asarray(grids[idx], dtype=np.float32)
            feat = sb.image_features(grid, w, h, aspect=asp, keep=0.60)
            boxes = {}
            if feat is None:
                for c in cands:
                    boxes[c] = sb.fallback_box(w, h, aspect=asp)
            else:
                for c in cands:
                    sc = sb.score_candidates(feat, w_thirds=c[0], w_ret=c[1], w_area=c[2])
                    i, _adopted, _bi, _bs = sb.pick(feat, sc, gate="abs", gate_rel=0.05)
                    boxes[c] = sb.to_pixels(feat, i, w, h)
            recs.append((w, h, mask, feat, boxes))
            meta.append((idx, img, w, h))

        n = len(recs)
        print(f"\n===== aspect={tag}  有效图 {n} 张（掩膜非空）=====")

        per = {}
        for c in cands:
            kp, kc, tc, cc, dmax, kcmax = [], [], [], [], [], []
            cut_a = cut_c = 0
            for w, h, mask, feat, boxes in recs:
                box = boxes[c]
                kr = metrics.kept_ratio(mask, box)
                cbox = metrics.center_crop(w, h, box[2], box[3])
                krc = metrics.kept_ratio(mask, cbox)
                if kr is None or krc is None:
                    continue
                kp.append(kr)
                kc.append(krc)
                ccp = metrics.centroid_in_crop(mask, box)
                if ccp is not None:
                    tc.append(metrics.thirds_distance(ccp[0], ccp[1]))
                ccc = metrics.centroid_in_crop(mask, cbox)
                if ccc is not None:
                    cc.append(metrics.thirds_distance(ccc[0], ccc[1]))
                if kr < 0.99:
                    cut_a += 1
                if krc < 0.99:
                    cut_c += 1
                if feat is not None:
                    mbox = metrics.center_crop(w, h, feat["max_wc"], feat["max_hc"])
                    mc = metrics.centroid_in_crop(mask, mbox)
                    if mc is not None:
                        dmax.append(metrics.thirds_distance(mc[0], mc[1]))
                    kmax = metrics.kept_ratio(mask, mbox)
                    if kmax is not None:
                        kcmax.append(kmax)
            kp = np.asarray(kp)
            kc = np.asarray(kc)
            d = kp - kc
            per[c] = {
                "kept": float(kp.mean()), "center": float(kc.mean()),
                "delta": float(d.mean()), "win": float((d > 1e-9).mean()),
                "thirds": float(np.mean(tc)) if tc else 0.0,
                "thirds_c": float(np.mean(cc)) if cc else 0.0,
                "cut": cut_a / max(1, n), "cut_c": cut_c / max(1, n),
                "d": d,
                "thirds_max": float(np.mean(dmax)) if dmax else 0.0,
                "kept_maxcenter": float(np.mean(kcmax)) if kcmax else 0.0,
                "delta_maxcenter": (float(np.mean(kp[: len(kcmax)] - np.asarray(kcmax)))
                                    if kcmax else 0.0),
            }

        print("\n-- B 统计显著性（对照＝run_eval 口径的同尺寸居中框）")
        for c in (BASE, REC):
            r = per[c]
            lo, hi = bootstrap_ci(r["d"], args.boot)
            w_, l_, z = sign_test_z(r["d"])
            print(f"   {c[0]:.2f}/{c[1]:.2f}/{c[2]:.2f}  delta={r['delta']:+.5f} "
                  f"bootstrap95=[{lo:+.5f}, {hi:+.5f}]  win/lose={w_}/{l_} 符号检验 z={z:+.1f}")

        print("\n-- D 居中对照换口径（同尺寸居中 vs 该比例最大框居中）")
        for c in (BASE, REC):
            r = per[c]
            print(f"   {c[0]:.2f}/{c[1]:.2f}/{c[2]:.2f}  同尺寸 delta={r['delta']:+.5f} | "
                  f"最大框居中：保留率 {r['kept']:.4f} vs {r['kept_maxcenter']:.4f} "
                  f"delta={r['delta_maxcenter']:+.5f} | 三分线 算法={r['thirds']:.4f} "
                  f"最大框居中={r['thirds_max']:.4f}")

        for c in (BASE, REC):
            p = out / f"harness_verify_{tag}_{int(round(c[0]*100)):02d}{int(round(c[1]*100)):02d}{int(round(c[2]*100)):02d}.tsv"
            with p.open("w", encoding="utf-8", newline="") as f:
                f.write(TSV_HEADER)
                for (w, h, mask, feat, boxes), (_i, img, _w, _h) in zip(recs, meta):
                    box = boxes[c]
                    ka = box[2] * box[3] / (w * h)
                    f.write(f"{img}\t{w}\t{h}\t{box[0]}\t{box[1]}\t{box[2]}\t{box[3]}\t"
                            f"{ka:.6f}\t{1 if ka >= 0.97 else 0}\ttrue\tok\n")
            print(f"   A 写出 {p}")

        print("\n-- C 3 折交叉验证（折内选最优权重，折外验）")
        rng = random.Random(20261005)
        order = list(range(n))
        rng.shuffle(order)
        folds = [order[i::3] for i in range(3)]
        oof, picks = [], []
        for fi, test_idx in enumerate(folds):
            test_set = set(test_idx)
            train_idx = [i for i in order if i not in test_set]
            best, best_mean = None, -9.0
            for c in cands:
                m = float(np.mean([per[c]["d"][i] for i in train_idx]))
                if m > best_mean:
                    best, best_mean = c, m
            dtest = float(np.mean([per[best]["d"][i] for i in test_idx]))
            print(f"   折{fi+1}: 折内最优 {best[0]:.2f}/{best[1]:.2f}/{best[2]:.2f} "
                  f"（折内 {best_mean:+.5f}）→ 折外 {dtest:+.5f}")
            picks.append((best, dtest))
            oof += [per[best]["d"][i] for i in test_idx]
        print(f"   折外合计 {float(np.mean(oof)):+.5f}（n={len(oof)}）")

        print("\n-- E 消融")
        for c in ((0.55, 0.25, 0.20), (0.55, 0.25, 0.00), (0.25, 0.55, 0.20), (0.40, 0.60, 0.00)):
            if c in per:
                r = per[c]
                print(f"   {c[0]:.2f}/{c[1]:.2f}/{c[2]:.2f}  delta={r['delta']:+.5f} "
                      f"thirds={r['thirds']:.4f} 切主体={r['cut']*100:.1f}%（居中 {r['cut_c']*100:.1f}%）")

        json.dump({f"{k[0]:.2f}/{k[1]:.2f}/{k[2]:.2f}": {kk: vv for kk, vv in v.items() if kk != "d"}
                   for k, v in per.items()},
                  (out / f"verify_{tag}.json").open("w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
