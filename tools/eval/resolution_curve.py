#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 2 条：显著度网格分辨率曲线（64x48 -> 128x96 -> 256x192）。

要回答的问题：**把显著度网格从 64x48 提上去，值多少？**

两条曲线
  A(网格)  : 真模型（u2netp / rmbg / ...）在该分辨率的网格 + 定稿权重 0.40/0.60/0.00
  B(网格)  : 真值掩膜下采样到**同一分辨率**的"完美先验" + 同一套权重
  B - A    : 该分辨率下"显著度图准不准"还剩多少余量

再加一条 coarse 消融：**评估网格固定**（默认 64x48），只把完美先验的**内容**
（真值掩膜）做粗（内容 64x48 / 32x24 / 16x12，即网格的 1/1 / 1/2 / 1/4），
看"网格本身"值多少。口径与 `oracle_prior_check.perfect_grid(coarse=k)` 一致：
先在 (W/k, H/k) 做 BOX 面积平均，再 NEAREST 放大回评估网格。

## 口径与自检

`sim_bestcrop.py` / `sweep_weights.py` 原先把网格尺寸硬编码成 64x48
（`image_features` 函数体首行 `cols, rows = GRID_W, GRID_H`、`to_pixels` 里
`xk * src_w / GRID_W` 等固定常量），任何非 64x48 的网格都会
`ValueError: cannot reshape array of size 12288 into shape (48,64)`，
`sweep_weights.py --grid-w/--grid-h` 因此是坏的。
**该缺陷已由 Lead 修好（纯增量、默认值不变）**：`image_features` / `to_pixels` /
`load_grids` 都接受 `cols` / `rows`，`sweep_weights.py:155` 与 `ceiling_check.py`
也已补传尺寸。

本文件因此有两种实现，由模块级 `USE_SHARED` 切换：
  - `USE_SHARED = True`（默认）：直接走共享 `sim.image_features` / `sim.to_pixels`；
  - `USE_SHARED = False`：走本文件内的私有副本 `_image_features_priv` / `_to_pixels_priv`
    （照抄 float32/float64 语义与 roundToInt 半进位），用于交叉验证。

`--selfcheck` 跑三项，全过才出曲线：
  1. 私有副本 vs 共享实现，64x48 真实网格上逐位比对（候选数组 + thirds/retention/area + to_pixels）；
  2. 复现已发布全量数字 A 0.975324 / 居中 0.970631（0.40/0.60/0.00）与
     A 0.972673 / 居中 0.978546（0.55/0.25/0.20）；
  3. 与 `sweep_weights.evaluate` 端到端比对（同子集、同权重，要求差 <= 1e-12）。

用法：
  python resolution_curve.py --selfcheck
  python resolution_curve.py --grids "64x48,128x96,256x192" --coarse "64x48,32x24,16x12"
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sim_bestcrop as sim          # noqa: E402
import sweep_weights as sw          # noqa: E402
from oracle_prior_check import perfect_grid   # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

GRID_W, GRID_H = 64, 48
f32 = np.float32
ONE_THIRD = f32(1.0) / f32(3.0)
TWO_THIRDS = f32(2.0) / f32(3.0)
EVAL_KEEP = 0.60
ANCHOR = 0
GATE_REL = 0.05
W_NEW = (0.40, 0.60, 0.00, 0.75, 1.6)   # RESULTS-weight-sweep.md 定稿
W_BASE = (0.55, 0.25, 0.20, 0.75, 1.6)  # 基线（自检用）

OUT_ROOT = Path(r"D:\spider\.cache\saliency-swap\res")
DUTS_WORK = Path(r"D:\spider\.cache\eval-duts\work-full")
SWAP = Path(r"D:\spider\.cache\saliency-swap")


# ================================================================= 尺寸参数化的复现器
# 与 sim_bestcrop.py 对应函数逐行等价，唯一区别是 cols/rows 由调用方给。
# float32/float64 的选取、运算次序、rti 半进位都照抄，缺一不可（否则聚合对不上）。

def rti_scalar(x: float) -> int:
    """Kotlin roundToInt：四舍五入、.5 向正无穷。"""
    return int(math.floor(x + 0.5))


def _build_candidates_priv(max_wg: int, max_hg: int, cols: int, rows: int,
                           keep: float = EVAL_KEEP, anchor: int = 0):
    """AutoFrame.kt:597-609 的 21 档尺寸 x 位置枚举（sim_bestcrop.py:82-114，尺寸参数化）。"""
    ks, xks, yks, wks, hks = [], [], [], [], []
    one = f32(1.0)
    span = f32(one - f32(keep))
    for k in range(21):
        scale = f32(one - f32(f32(f32(k) * span) / f32(20.0)))
        s_scale = f32(math.sqrt(float(max(scale, f32(0.0)))))
        wk = max(1, rti_scalar(float(f32(f32(max_wg) * s_scale))))
        wk = min(wk, cols)
        hk = max(1, rti_scalar(float(f32(f32(max_hg) * s_scale))))
        hk = min(hk, rows)

        step_x = max(1, cols // anchor) if anchor > 0 else max(1, wk // 16)
        step_y = max(1, rows // anchor) if anchor > 0 else max(1, hk // 16)
        max_x = max(0, cols - wk)
        max_y = max(0, rows - hk)

        yk = 0
        while True:
            xk = 0
            while True:
                ks.append(k); xks.append(xk); yks.append(yk); wks.append(wk); hks.append(hk)
                if xk == max_x:
                    break
                xk = min(max_x, xk + step_x)
            if yk == max_y:
                break
            yk = min(max_y, yk + step_y)
    return (np.asarray(ks, np.int64), np.asarray(xks, np.int64), np.asarray(yks, np.int64),
            np.asarray(wks, np.int64), np.asarray(hks, np.int64))


def _image_features_priv(grid: np.ndarray, src_w: int, src_h: int, aspect=None,
                         keep: float = EVAL_KEEP, anchor: int = 0,
                         cols: int = GRID_W, rows: int = GRID_H):
    """sim_bestcrop.image_features 的**私有副本**（尺寸参数化），返回值与那边同构。"""
    g = np.asarray(grid, dtype=np.float64).reshape(rows, cols)
    pos = np.where(g > 0.0, g, 0.0)
    total_weight = float(pos.sum())
    max_weight = float(pos.max()) if pos.size else 0.0
    if total_weight <= 0.0 or max_weight <= 0.0:
        return None

    ratio = f32(f32(src_w) / f32(src_h))
    safe_aspect = f32(aspect) if (aspect is not None and math.isfinite(aspect) and aspect > 0) \
        else (ratio if (math.isfinite(float(ratio)) and ratio > f32(0.0)) else f32(1.0))
    if ratio > safe_aspect:
        max_h = src_h
        max_w = rti_scalar(float(f32(f32(src_h) * safe_aspect)))
    else:
        max_w = src_w
        max_h = rti_scalar(float(f32(f32(src_w) / safe_aspect)))
    max_wc = min(max(1, max_w), max(1, src_w))
    max_hc = min(max(1, max_h), max(1, src_h))

    max_wg = min(max(1, rti_scalar(max_wc * cols / src_w)), cols)
    max_hg = min(max(1, rti_scalar(max_hc * rows / src_h)), rows)

    high_threshold = f32(0.6) * f32(max_weight)
    high_mask = g >= float(high_threshold)
    high = np.where(high_mask, g, 0.0)
    high_total = float(high.sum())

    yy = (np.arange(rows, dtype=np.float64) + 0.5)[:, None]
    xx = (np.arange(cols, dtype=np.float64) + 0.5)[None, :]

    sat_w = sim.integral(pos)
    sat_wx = sim.integral(pos * xx)
    sat_wy = sim.integral(pos * yy)
    sat_h = sim.integral(high)
    sat_hx = sim.integral(high * xx)
    sat_hy = sim.integral(high * yy)

    ks, xks, yks, wks, hks = _build_candidates_priv(max_wg, max_hg, cols, rows, keep, anchor)
    x1 = xks + wks
    y1 = yks + hks

    in_w = sim.box_sum(sat_w, yks, y1, xks, x1)
    sum_x = sim.box_sum(sat_wx, yks, y1, xks, x1)
    sum_y = sim.box_sum(sat_wy, yks, y1, xks, x1)
    in_h = sim.box_sum(sat_h, yks, y1, xks, x1)
    sum_hx = sim.box_sum(sat_hx, yks, y1, xks, x1)
    sum_hy = sim.box_sum(sat_hy, yks, y1, xks, x1)

    if high_total > 0.0:
        retention = np.clip((in_h / high_total).astype(f32), f32(0.0), f32(1.0))
    else:
        retention = np.clip((in_w / total_weight).astype(f32), f32(0.0), f32(1.0))

    has_h = in_h > 0.0
    cw = np.where(has_h, in_h, in_w)
    cx_raw = np.where(has_h, sum_hx, sum_x)
    cy_raw = np.where(has_h, sum_hy, sum_y)
    cx = np.zeros_like(cw)
    cy = np.zeros_like(cw)
    ok = cw > 0.0
    cx[ok] = np.clip(((cx_raw[ok] / cw[ok] - xks[ok]) / wks[ok]).astype(f32), f32(0.0), f32(1.0))
    cy[ok] = np.clip(((cy_raw[ok] / cw[ok] - yks[ok]) / hks[ok]).astype(f32), f32(0.0), f32(1.0))

    dx1 = (cx - ONE_THIRD).astype(f32)
    dy1 = (cy - ONE_THIRD).astype(f32)
    dx2 = (cx - TWO_THIRDS).astype(f32)
    dy2 = (cy - TWO_THIRDS).astype(f32)
    d1 = np.sqrt((dx1 * dx1 + dy1 * dy1).astype(np.float64))
    d2 = np.sqrt((dx2 * dx2 + dy2 * dy2).astype(np.float64))
    nearest = np.minimum(d1, d2)
    thirds = np.zeros_like(cx)
    thirds[ok] = np.clip((1.0 - nearest[ok] / 0.45).astype(f32), f32(0.0), f32(1.0))

    area_ratio = np.clip(((wks.astype(f32) * hks.astype(f32)) /
                          f32(max(1, max_wg) * max(1, max_hg))).astype(f32), f32(0.0), f32(1.0))

    return {
        "ks": ks, "xks": xks, "yks": yks, "wks": wks, "hks": hks,
        "thirds": thirds, "retention": retention, "area": area_ratio,
        "max_wg": max_wg, "max_hg": max_hg, "max_wc": max_wc, "max_hc": max_hc,
    }


def _to_pixels_priv(feat, i: int, src_w: int, src_h: int, cols: int = GRID_W, rows: int = GRID_H):
    """sim_bestcrop.to_pixels 的**私有副本**（尺寸参数化）。"""
    xk = int(feat["xks"][i]); yk = int(feat["yks"][i])
    wk = int(feat["wks"][i]); hk = int(feat["hks"][i])
    left = rti_scalar(xk * src_w / cols)
    top = rti_scalar(yk * src_h / rows)
    out_w = max(1, rti_scalar(wk * src_w / cols))
    out_h = max(1, rti_scalar(hk * src_h / rows))
    left_c = min(max(0, left), max(0, src_w - 1))
    top_c = min(max(0, top), max(0, src_h - 1))
    width_c = min(max(1, out_w), max(1, src_w - left_c))
    height_c = min(max(1, out_h), max(1, src_h - top_c))
    return left_c, top_c, width_c, height_c


# ---- 实现切换 --------------------------------------------------------------
# USE_SHARED=True （默认）：走共享实现 sim.image_features / sim.to_pixels
# USE_SHARED=False         ：走上面的私有副本
# 两条路径在 64x48 上必须逐位相同（--crosscheck 验证）。
USE_SHARED = True


def image_features(grid: np.ndarray, src_w: int, src_h: int, aspect=None,
                   keep: float = EVAL_KEEP, anchor: int = 0,
                   cols: int = GRID_W, rows: int = GRID_H):
    if USE_SHARED:
        return sim.image_features(grid, src_w, src_h, aspect, keep, anchor, cols, rows)
    return _image_features_priv(grid, src_w, src_h, aspect, keep, anchor, cols, rows)


def to_pixels(feat, i: int, src_w: int, src_h: int, cols: int = GRID_W, rows: int = GRID_H):
    if USE_SHARED:
        return sim.to_pixels(feat, i, src_w, src_h, cols, rows)
    return _to_pixels_priv(feat, i, src_w, src_h, cols, rows)


# ================================================================= 单图评估

def subj_centroid_norm(mask: np.ndarray):
    """主体质心的归一化坐标（与 run_eval.py 的 (mean+0.5)/w 口径一致）。"""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    return (float(xs.mean()) + 0.5) / float(w), (float(ys.mean()) + 0.5) / float(h)


def eval_one(feat, mask, I, total, cx_n, cy_n, w, h, weights,
             cols: int = GRID_W, rows: int = GRID_H):
    """一张图 + 一组权重 -> 逐图指标。口径照抄 sweep_weights.evaluate:164-227。"""
    wt, wr, wa, onset, slope = (f32(x) for x in weights[:5])
    if feat is None:
        l, t, cw, ch, _ = sim.fallback_box(w, h, None)
        x0, y0 = np.int64(l), np.int64(t)
        width_c, height_c = np.int64(cw), np.int64(ch)
        adopted = 0
    else:
        T, R, A = feat["thirds"], feat["retention"], feat["area"]
        SC = (T * wt).astype(f32)
        SC = (SC + (R * wr).astype(f32)).astype(f32)
        SC = (SC + (A * wa).astype(f32)).astype(f32)
        pen = np.maximum(f32(0.0), f32(onset) - R).astype(f32)
        SC = (SC - (pen * f32(slope)).astype(f32)).astype(f32)

        best_i = int(np.argmax(SC))
        base_pool = np.flatnonzero(feat["ks"] == 0)
        base_i = int(base_pool[int(np.argmax(SC[base_pool]))])
        base_s = f32(SC[base_i]); best_s = f32(SC[best_i])
        adopted = int(best_s >= f32(base_s + f32(max(0.0, GATE_REL))))
        sel = best_i if adopted else base_i
        x0, y0, width_c, height_c = to_pixels(feat, sel, w, h, cols, rows)

    x0, y0, width_c, height_c = sw.clip_box(
        np.asarray([x0]), np.asarray([y0]), np.asarray([width_c]), np.asarray([height_c]), w, h)
    x0 = int(x0[0]); y0 = int(y0[0]); width_c = int(width_c[0]); height_c = int(height_c[0])
    kept = float(sw.box_kept(I, total, np.asarray([x0]), np.asarray([y0]),
                             np.asarray([x0 + width_c]), np.asarray([y0 + height_c]))[0])

    cl = (w - width_c) // 2
    ct = (h - height_c) // 2
    kept_c = float(sw.box_kept(I, total, np.asarray([cl]), np.asarray([ct]),
                               np.asarray([cl + width_c]), np.asarray([ct + height_c]))[0])

    x1, y1 = x0 + width_c, y0 + height_c
    ccx = (cx_n * w - x0) / float(x1 - x0)
    ccy = (cy_n * h - y0) / float(y1 - y0)
    td = float(sw.thirds_dist(np.asarray([ccx]), np.asarray([ccy]))[0])
    ccx_c = (cx_n * w - cl) / float(width_c)
    ccy_c = (cy_n * h - ct) / float(height_c)
    td_c = float(sw.thirds_dist(np.asarray([ccx_c]), np.asarray([ccy_c]))[0])
    dc = float(np.hypot(ccx - 0.5, ccy - 0.5))

    return {
        "kept": kept, "kept_c": kept_c, "delta": kept - kept_c,
        "thirds": td, "thirds_c": td_c, "adopted": adopted,
        "cut": float(kept < 0.99), "dead": float(dc <= 0.05),
        "area": float(width_c * height_c) / float(w * h),
    }


def mean_of(rows, key):
    return float(np.mean([r[key] for r in rows])) if rows else float("nan")


def summarize(rows):
    d = [r["delta"] for r in rows]
    n = len(rows)
    sd = float(np.std(d, ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 0 else 0.0
    wins = sum(1 for r in rows if r["kept"] > r["kept_c"])
    return {
        "n": n,
        "A": mean_of(rows, "kept"),
        "center": mean_of(rows, "kept_c"),
        "delta": float(np.mean(d)) if d else float("nan"),
        "t": (float(np.mean(d)) / se) if se > 0 else 0.0,
        "win_rate": wins / n if n else float("nan"),
        "thirds": mean_of(rows, "thirds"),
        "thirds_center": mean_of(rows, "thirds_c"),
        "cut_rate": mean_of(rows, "cut"),
        "dead_rate": mean_of(rows, "dead"),
        "area": mean_of(rows, "area"),
        "adopt_rate": mean_of(rows, "adopted"),
    }


# ================================================================= 曲线主流程

def resolution_pass(pairs, bin_path, cols, rows, weights, limit=0, verbose=False):
    """一个分辨率、一组权重、一批先验来源 -> 逐图指标表。

    返回 {"algo": [...], "perfect": [...], "perfect_src": [...]}；
    perfect（B）用同一分辨率的完美先验，perfect_src 复用真值掩膜对象。
    """
    rows_sel = pairs[:limit] if limit else pairs
    algo, perfect = [], []
    nbytes = cols * rows * 4
    t0 = time.time()
    skip = 0
    with open(bin_path, "rb") as bf:
        size = Path(bin_path).stat().st_size
        avail = size // nbytes
        for n, (idx, image, w, h, mp) in enumerate(rows_sel):
            try:
                mask = np.asarray(Image.open(mp).convert("L")) >= 128
            except Exception:
                skip += 1
                continue
            if not mask.any() or idx >= avail:
                skip += 1
                continue
            total = int(mask.sum())
            I = sw.mask_integral(mask)
            cx_n, cy_n = subj_centroid_norm(mask)

            grid = sw.read_grid(bf, idx, cols, rows)
            fa = image_features(grid, w, h, None, EVAL_KEEP, ANCHOR, cols, rows)
            algo.append(eval_one(fa, mask, I, total, cx_n, cy_n, w, h, weights, cols, rows))

            g = perfect_grid(mask, cols=cols, rows=rows, coarse=1)
            fp = image_features(g, w, h, None, EVAL_KEEP, ANCHOR, cols, rows)
            perfect.append(eval_one(fp, mask, I, total, cx_n, cy_n, w, h, weights, cols, rows))

            if verbose and (n + 1) % 500 == 0:
                print(f"    ... {n+1}/{len(rows_sel)}  {time.time()-t0:.1f}s", flush=True)
    return {"algo": algo, "perfect": perfect, "skip": skip,
            "seconds": round(time.time() - t0, 2), "avail": avail}


def coarse_pass(pairs, grid_cols, grid_rows, content_cols, content_rows, weights, limit=0):
    """coarse 消融：**评估网格固定** (grid_cols, grid_rows)，只把完美先验的**内容**变粗。

    口径（与 `oracle_prior_check.perfect_grid(mask, cols, rows, coarse=k)` 完全一致）：
      1. 真值掩膜 → 灰度 L 图；
      2. `Image.resize((grid_cols//k, grid_rows//k), Image.BOX)` 做面积平均，k = 网格/内容 的整数倍；
      3. 若 `(grid_cols//k, grid_rows//k) != (grid_cols, grid_rows)`，再
         `Image.resize((grid_cols, grid_rows), Image.NEAREST)` 放大回评估网格。

    所以这一组量的是"**64×48 这个网格本身值多少**"：内容退化、格子数不动。
    **不是**"把评估网格降到 32×24"（那是另一个口径，本文件不再产出）。
    """
    k = max(1, int(round(grid_cols / float(content_cols))))
    if grid_cols % content_cols or grid_rows % content_rows:
        raise SystemExit(f"coarse 内容分辨率 {content_cols}x{content_rows} 必须整除评估网格 "
                         f"{grid_cols}x{grid_rows}")
    k2 = max(1, int(round(grid_rows / float(content_rows))))
    if k != k2:
        raise SystemExit(f"coarse 的横纵倍率不一致：{k} vs {k2}")
    rows_sel = pairs[:limit] if limit else pairs
    out = []
    for (idx, image, w, h, mp) in rows_sel:
        try:
            mask = np.asarray(Image.open(mp).convert("L")) >= 128
        except Exception:
            continue
        if not mask.any():
            continue
        total = int(mask.sum())
        I = sw.mask_integral(mask)
        cx_n, cy_n = subj_centroid_norm(mask)
        g = perfect_grid(mask, cols=grid_cols, rows=grid_rows, coarse=k)
        f = image_features(g, w, h, None, EVAL_KEEP, ANCHOR, grid_cols, grid_rows)
        out.append(eval_one(f, mask, I, total, cx_n, cy_n, w, h, weights, grid_cols, grid_rows))
    return out, k


# ================================================================= 自检

def crosscheck(pairs, limit=2020):
    """私有副本 vs 共享实现，在 64x48 真实网格上逐位比对。

    两条路径一致本身就是证据：说明我对 sim_bestcrop 语义的复现没有偏差，
    也说明换分辨率时共享实现接上 cols/rows 后行为正确。
    """
    rows_sel = pairs[:limit] if limit else pairs
    bin_path = DUTS_WORK / "grids.bin"
    bad = 0
    checked = 0
    with open(bin_path, "rb") as bf:
        for (idx, image, w, h, mp) in rows_sel:
            grid = sw.read_grid(bf, idx, 64, 48)
            mine = _image_features_priv(grid, w, h, None, EVAL_KEEP, ANCHOR, 64, 48)
            ref = sim.image_features(grid, w, h, None, EVAL_KEEP, ANCHOR, 64, 48)
            if (mine is None) != (ref is None):
                print(f"  逐位自检 FAIL idx={idx}: None 判定不一致"); bad += 1; continue
            if mine is None:
                continue
            checked += 1
            for key in ("ks", "xks", "yks", "wks", "hks"):
                if not np.array_equal(mine[key], ref[key]):
                    print(f"  逐位自检 FAIL idx={idx} 字段 {key}"); bad += 1; break
            else:
                for key in ("thirds", "retention", "area"):
                    if not np.array_equal(mine[key], ref[key]):
                        d = float(np.abs(mine[key].astype(np.float64) - ref[key].astype(np.float64)).max())
                        print(f"  逐位自检 FAIL idx={idx} 字段 {key} 最大差 {d:g}"); bad += 1; break
            # to_pixels 也逐候选比
            if mine is not None:
                for i in range(len(mine["ks"])):
                    a = _to_pixels_priv(mine, i, w, h, 64, 48)
                    b = sim.to_pixels(ref, i, w, h, 64, 48)
                    if a != b:
                        print(f"  逐位自检 FAIL idx={idx} to_pixels[{i}] {a} != {b}"); bad += 1; break
    print(f"  逐位自检（私有副本 vs 共享实现）：比对 {checked} 条，不一致 {bad} 条")
    return bad == 0


def published_selfcheck(pairs, limit=0):
    """用本文件的实现跑已发布数字（RESULTS-weight-sweep.md 3.3 节）。

    这四个数是**全量 5019 张**的，所以这里默认忽略 --limit；
    子集上它们本来就不该相等（DUTS-TE 清单不是随机序）。
    """
    want = {"A(0.40/0.60/0.00)": 0.975324, "center(0.40/0.60/0.00)": 0.970631,
            "A(0.55/0.25/0.20)": 0.972673, "center(0.55/0.25/0.20)": 0.978546}
    r1 = resolution_pass(pairs, str(DUTS_WORK / "grids.bin"), 64, 48, W_NEW)
    r2 = resolution_pass(pairs, str(DUTS_WORK / "grids.bin"), 64, 48, W_BASE)
    s1, s2 = summarize(r1["algo"]), summarize(r2["algo"])
    got = {
        "A(0.40/0.60/0.00)": s1["A"], "center(0.40/0.60/0.00)": s1["center"],
        "A(0.55/0.25/0.20)": s2["A"], "center(0.55/0.25/0.20)": s2["center"],
    }
    ok = True
    if limit:
        print(f"  （已发布值是全量 {len(pairs)} 张的口径；下面是全量复算，--limit 不适用于这一项）")
    for k, v in want.items():
        g = got[k]
        same = f"{g:.6f}" == f"{v:.6f}"
        ok &= same
        print(f"  {k:28s} 期望 {v:.6f}  实得 {g:.6f}  {'OK' if same else 'MISMATCH'}")
    return ok


def harness_selfcheck(pairs, tol=1e-12):
    """与 sweep_weights.evaluate 端到端比对（同子集、同权重）。"""
    ok = True
    for name, W3 in (("A 0.40/0.60/0.00", (0.40, 0.60, 0.00)),
                     ("base 0.55/0.25/0.20", (0.55, 0.25, 0.20))):
        r = sw.evaluate(pairs, str(DUTS_WORK / "grids.bin"),
                        np.asarray([W3 + (0.75, 1.6)], dtype=np.float32), cols=64, rows=48)
        my = summarize(resolution_pass(pairs, str(DUTS_WORK / "grids.bin"), 64, 48,
                                       W3 + (0.75, 1.6))["algo"])
        d1 = abs(my["A"] - float(r["mean_kept"][0]))
        d2 = abs(my["center"] - float(r["mean_kept_center"][0]))
        same = d1 <= tol and d2 <= tol
        ok &= same
        print(f"  harness {name:22s} kept {my['A']:.6f} vs {float(r['mean_kept'][0]):.6f}"
              f"  居中 {my['center']:.6f} vs {float(r['mean_kept_center'][0]):.6f}"
              f"  差 {max(d1,d2):.2e}  {'OK' if same else 'MISMATCH'}")
    return ok


# ================================================================= 主入口

def resolve_bin(spec, args):
    """分辨率 spec -> 网格 bin 路径。

    优先 `--bin-map`（形如 '128x96=<path>,256x192=<path>'），否则回落到
    '64x48 -> --grids-bin'、'其它 -> <bin-dir>/grids_<W>x<H>.bin'。
    """
    m = getattr(args, "_bin_map", {})
    if spec in m:
        return m[spec]
    cw, ch = (int(x) for x in spec.lower().split("x"))
    if (cw, ch) == (64, 48):
        return args.grids_bin
    return str(Path(args.bin_dir) / f"grids_{cw}x{ch}.bin")


def parse_bin_map(text, known):
    """'128x96=path,256x192=path' -> {'128x96': path}；未给的分辨率回落到默认约定。"""
    out = {}
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"--bin-map 里 '{part}' 不是 key=path 形式")
        key, path = part.split("=", 1)
        key = key.strip().lower()
        if known and key not in known:
            print(f"  警告：--bin-map 的 {key} 不在 --grids 里，忽略")
            continue
        out[key] = path.strip()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="显著度网格分辨率曲线")
    ap.add_argument("--grids-manifest", default=str(DUTS_WORK / "grids_manifest.tsv"))
    ap.add_argument("--grids-bin", default=str(DUTS_WORK / "grids.bin"))
    ap.add_argument("--manifest", default=str(DUTS_WORK / "manifest.tsv"))
    ap.add_argument("--grids", default="64x48,128x96,256x192",
                    help="要评估的分辨率，形如 '64x48,128x96'")
    ap.add_argument("--bin-dir", default=str(OUT_ROOT / "u2netp"),
                    help="更高分辨率的 grids_<W>x<H>.bin 所在目录")
    ap.add_argument("--bin-map", default="",
                    help="显式指定某分辨率的 bin，形如 '128x96=D:\\\\...\\\\grids.bin,256x192=...'")
    ap.add_argument("--coarse", default="64x48,32x24,16x12",
                    help="coarse 消融的**内容**分辨率；评估网格由 --coarse-grid 固定")
    ap.add_argument("--coarse-grid", default="64x48",
                    help="coarse 消融时固定的评估网格，默认 64x48")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--weights", default="0.40,0.60,0.00",
                    help="A/B 用的权重；只支持 3 项（onset/slope 用默认）")
    ap.add_argument("--selfcheck", action="store_true", help="只跑自检")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    pairs = sw.load_pairs(args.grids_manifest, args.manifest)
    print(f"图文对 {len(pairs)} 条｜权重 {args.weights}｜"
          f"{'limit ' + str(args.limit) if args.limit else '全量'}")

    if args.selfcheck:
        print("\n[自检 1] 私有副本 vs 共享 sim.image_features（64x48 逐位比对）")
        ok1 = crosscheck(pairs, args.limit or 2020)
        print("\n[自检 2] 复现已发布聚合数字（RESULTS-weight-sweep.md 3.3 节）")
        ok2 = published_selfcheck(pairs, args.limit)
        print("\n[自检 3] 与 sweep_weights.evaluate 端到端比对（同子集、同权重）")
        ok3 = harness_selfcheck(pairs, tol=1e-12)
        print(f"\n自检结果：逐位 {'通过' if ok1 else '未通过'}，"
              f"聚合 {'通过' if ok2 else '未通过'}，harness {'通过' if ok3 else '未通过'}")
        return 0 if (ok1 and ok2 and ok3) else 1

    w3 = [float(x) for x in args.weights.split(",")]
    if len(w3) == 3:
        weights = (w3[0], w3[1], w3[2], 0.75, 1.6)
    else:
        raise SystemExit("--weights 只支持 3 项（thirds,ret,area）")

    out = {"manifest": args.manifest, "n_pairs": len(pairs), "weights": list(weights),
           "resolutions": {}, "coarse": {}}

    grid_specs = [s.strip().lower() for s in args.grids.split(",") if s.strip()]
    args._bin_map = parse_bin_map(args.bin_map, set(grid_specs))

    for spec in grid_specs:
        cw, ch = (int(x) for x in spec.split("x"))
        bin_path = resolve_bin(spec, args)
        if not Path(bin_path).is_file():
            print(f"\n跳过 {spec}：缺 {bin_path}")
            out["resolutions"][spec] = {"error": "missing bin", "bin": bin_path}
            continue
        print(f"\n=== {spec} ===", flush=True)
        t0 = time.time()
        r = resolution_pass(pairs, bin_path, cw, ch, weights, limit=args.limit, verbose=True)
        sa, sb = summarize(r["algo"]), summarize(r["perfect"])
        dt = time.time() - t0
        print(f"  A={sa['A']:.6f}  居中={sa['center']:.6f}  差值={sa['delta']:+.5f} (t={sa['t']:+.2f})")
        print(f"  B={sb['A']:.6f}  居中={sb['center']:.6f}  差值={sb['delta']:+.5f} (t={sb['t']:+.2f})")
        print(f"  B-A={sb['A']-sa['A']:+.6f} | A三分线={sa['thirds']:.4f} B三分线={sb['thirds']:.4f}"
              f" | A采纳={sa['adopt_rate']*100:.1f}% B采纳={sb['adopt_rate']*100:.1f}%"
              f" | 评估 {dt:.1f}s (skip={r['skip']})")
        out["resolutions"][spec] = {
            "bin": bin_path, "grid_w": cw, "grid_h": ch, "avail": r["avail"],
            "skip": r["skip"], "eval_seconds": round(dt, 2),
            "A": sa, "B": sb, "B_minus_A": sb["A"] - sa["A"],
        }

    if args.coarse:
        gc, gr = (int(x) for x in args.coarse_grid.lower().split("x"))
        print(f"\n=== coarse 消融（评估网格固定 {gc}x{gr}，只把完美先验内容变粗）===", flush=True)
        print("    口径：内容 BOX 面积平均到 (W/k, H/k) 后 NEAREST 放大回评估网格；"
              "与 oracle_prior_check.perfect_grid(coarse=k) 一致", flush=True)
        for spec in [s.strip().lower() for s in args.coarse.split(",") if s.strip()]:
            cw, ch = (int(x) for x in spec.split("x"))
            t0 = time.time()
            rows, k = coarse_pass(pairs, gc, gr, cw, ch, weights, limit=args.limit)
            s = summarize(rows)
            dt = time.time() - t0
            print(f"  {cw:>3d}x{ch:<3d} (=网格/{k}): 保留率={s['A']:.6f} 居中={s['center']:.6f}"
                  f" 差值={s['delta']:+.5f} (t={s['t']:+.2f}) 三分线={s['thirds']:.4f}"
                  f" 采纳={s['adopt_rate']*100:.1f}%  评估 {dt:.1f}s")
            out["coarse"][spec] = {"content_w": cw, "content_h": ch, "grid_w": gc, "grid_h": gr,
                                   "coarse_k": k, "eval_seconds": round(dt, 2), "summary": s}

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n已写 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
