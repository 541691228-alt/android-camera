"""扫 bestCrop 的评分权重（C 线：只走理论，不碰真机）。

思路：`sim_bestcrop.py` 已经把 Kotlin 的 bestCrop 逐位复现（六种口径 5019 行 0 不一致），
且候选框特征（thirds / retention / area）与权重解耦，所以"换一组权重"只是线性组合 + argmax，
可以在几秒内评估完整数据集。

要回答的问题：能不能找到一组权重，让 src 口径的保留率不再输给居中裁剪
（当前 0.9727 vs 0.9785，配对 t=−7.03），同时不牺牲构图（三分线距离 0.0951 不变差）。

指标口径与 `run_eval.py` 完全对齐：
  CUT_TOL = 0.99         保留率 < 0.99 就算"切到主体"
  DEAD_CENTER_TOL = 0.05 主体质心离框中心 5% 以内算"摆正中"
  居中对照框 = 与算法框同样尺寸的居中框（所以每个权重组有自己的对照）
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

import sim_bestcrop as sb

GRID_W, GRID_H = 64, 48
REC_BYTES = GRID_W * GRID_H * 4
CUT_TOL = 0.99
DEAD_CENTER_TOL = 0.05
F32 = np.float32

THIRDS_PTS = np.array(
    [[1 / 3, 1 / 3], [2 / 3, 1 / 3], [1 / 3, 2 / 3], [2 / 3, 2 / 3]], dtype=np.float64
)

BASE_W = (0.55, 0.25, 0.20, 0.75, 1.6)  # AutoFrame.kt:777-779 的现行权重


# ----------------------------------------------------------------- 读数据

def read_tsv(path):
    with open(path, encoding="utf-8") as f:
        head = f.readline().rstrip("\n").split("\t")
        rows = [ln.rstrip("\n").split("\t") for ln in f if ln.strip()]
    return head, rows


def load_pairs(grids_manifest, mask_manifest):
    """把 grids 清单和掩膜清单对齐成 [(grid_index, image, w, h, mask_path)]。"""
    _, mrows = read_tsv(mask_manifest)
    mdir = Path(mask_manifest).resolve().parent
    by_full, by_base = {}, {}
    for r in mrows:
        if len(r) < 2:
            continue
        mp = r[1] if Path(r[1]).is_absolute() else str(mdir / r[1])
        by_full[r[0]] = mp
        by_base[Path(r[0]).name] = mp
    _, grows = read_tsv(grids_manifest)
    pairs = []
    for r in grows:
        image = r[1]
        mp = by_full.get(image) or by_base.get(Path(image).name)
        if mp is None:
            continue
        pairs.append((int(r[0]), image, int(r[2]), int(r[3]), mp))
    return pairs


def read_grid(bf, idx, cols=GRID_W, rows=GRID_H):
    """读第 idx 条网格记录。默认 64×48（REC_BYTES），--grid-w/--grid-h 可换分辨率。"""
    nbytes = cols * rows * 4
    bf.seek(idx * nbytes)
    buf = bf.read(nbytes)
    if len(buf) != nbytes:
        raise EOFError(f"grids.bin 读不到第 {idx} 条记录（{cols}x{rows}）")
    return np.frombuffer(buf, dtype=">f4").reshape(rows, cols).astype(np.float32)


def mask_integral(mask):
    I = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.int64)
    I[1:, 1:] = np.cumsum(np.cumsum(mask.astype(np.int64), axis=0), axis=1)
    return I


def box_kept(I, total, x0, y0, x1, y1):
    inside = I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0]
    return inside.astype(np.float64) / float(total)


def thirds_dist(cx, cy):
    d = np.hypot(cx[:, None] - THIRDS_PTS[None, :, 0], cy[:, None] - THIRDS_PTS[None, :, 1])
    return d.min(axis=1)


def clip_box(left, top, out_w, out_h, w, h):
    """AutoFrame.kt:672-682 的像素夹取（向量化版）。"""
    left_c = np.clip(left, 0, max(0, w - 1))
    top_c = np.clip(top, 0, max(0, h - 1))
    width_c = np.minimum(np.maximum(1, out_w), np.maximum(1, w - left_c))
    height_c = np.minimum(np.maximum(1, out_h), np.maximum(1, h - top_c))
    return left_c, top_c, width_c, height_c


def rti_vec(v):
    """Kotlin roundToInt 的 half-up（向量化）。"""
    return np.floor(np.asarray(v, dtype=np.float64) + 0.5).astype(np.int64)


# ----------------------------------------------------------------- 扫参

def evaluate(pairs, bin_path, W, aspect=None, keep=None, anchor=0, gate="abs",
             gate_rel=sb.ABS_GATE_MARGIN, limit=0, sample=0, seed=7, verbose=False,
             cols=GRID_W, rows=GRID_H):
    """在 pairs 上评估权重矩阵 W（nW × 5：thirds, retention, area, cut_onset, cut_slope）。"""
    keep = sb.EVAL_MIN_KEEP if keep is None else keep
    if len(pairs) == 0:
        raise SystemExit("没有可用的图文对，检查两个清单的路径是否对得上")
    if sample and sample < len(pairs):
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(len(pairs), size=sample, replace=False))
        pairs = [pairs[i] for i in sel]
    if limit:
        pairs = pairs[:limit]

    nW = W.shape[0]
    wt, wr, wa, onset, slope = (W[:, i].astype(np.float32) for i in range(5))
    ar = np.arange(nW)

    acc = {k: np.zeros(nW, dtype=np.float64) for k in
           ("kept", "kept_c", "thirds", "thirds_c", "cut", "cut_c", "dead", "dead_c",
            "skip", "wins", "ties", "loses", "gain", "drop", "adopted", "area")}
    n_img = 0
    deltas = []
    t0 = time.time()

    with open(bin_path, "rb") as bf:
        for n, (idx, image, w, h, mp) in enumerate(pairs):
            try:
                mask = np.array(Image.open(mp).convert("L")) >= 128
            except Exception:
                continue
            if not mask.any():
                continue
            total = int(mask.sum())
            I = mask_integral(mask)
            ys, xs = np.nonzero(mask)
            cx_n = (float(xs.mean()) + 0.5) / float(w)
            cy_n = (float(ys.mean()) + 0.5) / float(h)

            grid = read_grid(bf, idx, cols, rows)
            feat = sb.image_features(grid, w, h, aspect, keep, anchor, cols, rows)

            if feat is None:  # Kotlin 兜底分支：所有权重组都拿到同一个居中框
                l, t, cw, ch, _ = sb.fallback_box(w, h, aspect)
                left = np.full(nW, l, dtype=np.int64)
                top = np.full(nW, t, dtype=np.int64)
                out_w = np.full(nW, cw, dtype=np.int64)
                out_h = np.full(nW, ch, dtype=np.int64)
                adopted = np.zeros(nW, dtype=bool)
            else:
                T = feat["thirds"]
                R = feat["retention"]
                A = feat["area"]
                SC = (T[:, None] * wt[None, :]).astype(F32)
                SC = (SC + (R[:, None] * wr[None, :])).astype(F32)
                SC = (SC + (A[:, None] * wa[None, :])).astype(F32)
                pen = np.maximum(F32(0.0), onset[None, :] - R[:, None]).astype(F32)
                SC = (SC - (pen * slope[None, :])).astype(F32)

                best_i = np.argmax(SC, axis=0)
                base_pool = np.flatnonzero(feat["ks"] == 0)
                sub = SC[base_pool, :]
                bsel = np.argmax(sub, axis=0)
                base_i = base_pool[bsel]
                base_s = sub[bsel, ar]
                best_s = SC.max(axis=0)
                margin = F32(max(0.0, float(gate_rel)))  # ABS 门槛（评测默认）
                adopted = best_s >= (base_s + margin).astype(F32)
                sel = np.where(adopted, best_i, base_i)

                left = rti_vec(feat["xks"][sel].astype(np.float64) * w / cols)
                top = rti_vec(feat["yks"][sel].astype(np.float64) * h / rows)
                out_w = np.maximum(1, rti_vec(feat["wks"][sel].astype(np.float64) * w / cols))
                out_h = np.maximum(1, rti_vec(feat["hks"][sel].astype(np.float64) * h / rows))

            left_c, top_c, width_c, height_c = clip_box(left, top, out_w, out_h, w, h)
            x0, y0 = left_c, top_c
            x1, y1 = left_c + width_c, top_c + height_c
            kept = box_kept(I, total, x0, y0, x1, y1)

            # 居中对照：与算法框同样尺寸
            cl = (w - width_c) // 2
            ct = (h - height_c) // 2
            kept_c = box_kept(I, total, cl, ct, cl + width_c, ct + height_c)

            # 构图：主体质心在框内的归一化位置
            ccx = (cx_n * w - x0) / (x1 - x0).astype(np.float64)
            ccy = (cy_n * h - y0) / (y1 - y0).astype(np.float64)
            ccx_c = (cx_n * w - cl) / width_c.astype(np.float64)
            ccy_c = (cy_n * h - ct) / height_c.astype(np.float64)
            td = thirds_dist(ccx, ccy)
            td_c = thirds_dist(ccx_c, ccy_c)
            dc = np.hypot(ccx - 0.5, ccy - 0.5)
            dc_c = np.hypot(ccx_c - 0.5, ccy_c - 0.5)

            n_img += 1
            acc["kept"] += kept
            acc["kept_c"] += kept_c
            acc["thirds"] += td
            acc["thirds_c"] += td_c
            acc["cut"] += (kept < CUT_TOL)
            acc["cut_c"] += (kept_c < CUT_TOL)
            acc["dead"] += (dc <= DEAD_CENTER_TOL)
            acc["dead_c"] += (dc_c <= DEAD_CENTER_TOL)
            acc["skip"] += (width_c * height_c >= 0.97 * w * h)
            acc["area"] += (width_c * height_c).astype(np.float64) / float(w * h)
            acc["adopted"] += adopted
            acc["wins"] += (kept > kept_c)
            acc["ties"] += np.abs(kept - kept_c) < 1e-9
            acc["loses"] += (kept < kept_c)
            acc["gain"] += np.where(kept > kept_c, kept - kept_c, 0.0)
            acc["drop"] += np.where(kept < kept_c, kept_c - kept, 0.0)
            deltas.append(kept - kept_c)

            if verbose and (n + 1) % 500 == 0:
                print(f"  ... {n+1}/{len(pairs)}  {time.time()-t0:.1f}s", flush=True)

    n = float(n_img)
    if n == 0:
        raise SystemExit("所有掩膜都读不出来")
    D = np.stack(deltas, axis=0) if deltas else np.zeros((1, nW))
    mean_d = D.mean(axis=0)
    sd = D.std(axis=0, ddof=1) if D.shape[0] > 1 else np.zeros(nW)
    se = sd / math.sqrt(D.shape[0]) if D.shape[0] > 1 else np.zeros(nW)
    with np.errstate(invalid="ignore", divide="ignore"):
        tstat = np.where(se > 0, mean_d / se, 0.0)

    res = {
        "n": int(n),
        "w_thirds": W[:, 0].tolist(), "w_ret": W[:, 1].tolist(), "w_area": W[:, 2].tolist(),
        "cut_onset": W[:, 3].tolist(), "cut_slope": W[:, 4].tolist(),
        "mean_kept": (acc["kept"] / n).tolist(),
        "mean_kept_center": (acc["kept_c"] / n).tolist(),
        "mean_delta": mean_d.tolist(),
        "t_stat": tstat.tolist(),
        "win_rate": (acc["wins"] / n).tolist(),
        "tie_rate": (acc["ties"] / n).tolist(),
        "mean_thirds": (acc["thirds"] / n).tolist(),
        "mean_thirds_center": (acc["thirds_c"] / n).tolist(),
        "cut_rate": (acc["cut"] / n).tolist(),
        "cut_rate_center": (acc["cut_c"] / n).tolist(),
        "dead_rate": (acc["dead"] / n).tolist(),
        "skip_rate": (acc["skip"] / n).tolist(),
        "mean_keep_area": (acc["area"] / n).tolist(),
        "adopt_rate": (acc["adopted"] / n).tolist(),
        "seconds": round(time.time() - t0, 2),
    }
    return res


# ----------------------------------------------------------------- 权重集

def grid_weights(kind):
    """返回 (nW × 5) 的权重矩阵。"""
    rows = []
    if kind == "simplex":
        step = 0.05
        vals = [round(i * step, 4) for i in range(int(1 / step) + 1)]
        for wt in vals:
            for wr in vals:
                wa = round(1.0 - wt - wr, 4)
                if wa < -1e-9 or wa > 1.0:
                    continue
                rows.append((wt, wr, max(0.0, wa), 0.75, 1.6))
    elif kind == "fine":
        for wt in np.arange(0.10, 0.625, 0.025):
            for wa in (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20):
                wr = round(1.0 - wt - wa, 4)
                if wr < -1e-9 or wr > 1.0:
                    continue
                rows.append((round(float(wt), 4), wr, wa, 0.75, 1.6))
    elif kind == "cut":
        for base3 in ((0.55, 0.25, 0.20), (0.40, 0.60, 0.00), (0.30, 0.70, 0.00)):
            for onset in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
                for slope in [0.0, 0.8, 1.6, 3.0, 5.0, 8.0]:
                    rows.append(base3 + (onset, slope))
    elif kind == "base":
        rows.append(BASE_W)
    else:
        raise SystemExit(f"未知权重集：{kind}")
    return np.array(rows, dtype=np.float64)


def parse_wlist(spec):
    rows = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [float(x) for x in chunk.split(",")]
        if len(parts) == 3:
            parts = parts + [0.75, 1.6]
        if len(parts) != 5:
            raise SystemExit(f"权重写法要 3 或 5 个数：{chunk}")
        rows.append(tuple(parts))
    return np.array(rows, dtype=np.float64)


def show(res, top=15, sort="delta"):
    order = np.argsort(-np.asarray(res["mean_delta"])) if sort == "delta" \
        else np.argsort(np.asarray(res["mean_thirds"]))
    print(f"\n{'#':>3} {'wT':>5} {'wR':>5} {'wA':>5} {'onset':>6} {'slope':>6} "
          f"{'kept':>7} {'center':>7} {'delta':>8} {'t':>7} {'win%':>6} "
          f"{'thirds':>7} {'thirdsC':>8} {'cut%':>6} {'skip%':>6} {'area':>6}")
    for rank, i in enumerate(order[:top], 1):
        print(f"{rank:>3} {res['w_thirds'][i]:>5.2f} {res['w_ret'][i]:>5.2f} {res['w_area'][i]:>5.2f} "
              f"{res['cut_onset'][i]:>6.2f} {res['cut_slope'][i]:>6.2f} "
              f"{res['mean_kept'][i]:>7.4f} {res['mean_kept_center'][i]:>7.4f} "
              f"{res['mean_delta'][i]:>+8.5f} {res['t_stat'][i]:>7.2f} "
              f"{res['win_rate'][i]*100:>5.1f} "
              f"{res['mean_thirds'][i]:>7.4f} {res['mean_thirds_center'][i]:>8.4f} "
              f"{res['cut_rate'][i]*100:>5.1f} {res['skip_rate'][i]*100:>5.1f} "
              f"{res['mean_keep_area'][i]:>6.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="扫 bestCrop 评分权重")
    ap.add_argument("--grids-manifest", required=True)
    ap.add_argument("--grids-bin", required=True)
    ap.add_argument("--manifest", required=True, help="image/mask 清单（表头 image mask）")
    ap.add_argument("--scan", default="base", choices=["base", "simplex", "fine", "cut", "list"])
    ap.add_argument("--weights", default="", help="--scan list 时：'t,r,a[,onset,slope];...'")
    ap.add_argument("--aspect", default="src")
    ap.add_argument("--gate-rel", type=float, default=float(sb.ABS_GATE_MARGIN))
    ap.add_argument("--anchor", type=int, default=0)
    ap.add_argument("--keep", type=float, default=None)
    ap.add_argument("--grid-w", type=int, default=GRID_W, help="网格宽，默认 64（换分辨率时配合 --grids-bin）")
    ap.add_argument("--grid-h", type=int, default=GRID_H, help="网格高，默认 48")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out", default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    aspect = None if args.aspect == "src" else float(args.aspect)
    pairs = load_pairs(args.grids_manifest, args.manifest)
    W = parse_wlist(args.weights) if args.scan == "list" else grid_weights(args.scan)
    print(f"图文对 {len(pairs)} 条｜权重 {len(W)} 组｜aspect={args.aspect}｜样本={args.sample or len(pairs)}"
          f"｜网格={args.grid_w}x{args.grid_h}")

    res = evaluate(pairs, args.grids_bin, W, aspect=aspect, keep=args.keep, anchor=args.anchor,
                   gate_rel=args.gate_rel, limit=args.limit, sample=args.sample, seed=args.seed,
                   verbose=args.verbose, cols=args.grid_w, rows=args.grid_h)
    res["aspect"] = args.aspect
    res["grid_w"] = args.grid_w
    res["grid_h"] = args.grid_h
    res["grids_manifest"] = str(args.grids_manifest)
    res["seconds_total"] = res["seconds"]
    show(res, top=args.top)
    if args.out:
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n已写 {args.out}")


if __name__ == "__main__":
    main()
