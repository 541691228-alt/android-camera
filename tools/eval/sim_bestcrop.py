#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线复现 AutoFrame.bestCrop / scoreCrop，用来在全量数据上扫评分权重。

为什么要复现而不是改 Kotlin 再跑：一趟 gradle harness 要 2 分钟，而且 Kotlin 里那三个
系数是编译期常量，扫一次得改一次源码。这里把同一套逻辑搬到 Python，候选框的**特征**
（thirds / retention / areaRatio / cutPenalty）跟权重无关，可以先算好一次，
之后任何一组权重只是做一次线性组合 + argmax，几秒就能扫几百组。

对齐口径（跟 Kotlin 逐位一致才有资格用来下结论）：
  * 所有 Float 运算用 np.float32 模拟（Kotlin Float == IEEE binary32）
  * 累加量（inWeight / sumX / highTotal ...）是 Double，用 float64
  * roundToInt 是「half up」（floor(x+0.5)），不是 Python 的银行家舍入
  * 三个常数：ONE_THIRD = 1f/3f、TWO_THIRDS = 2f/3f（float32 近似值）

用法：
  # 逐行校验复现是否与真 Kotlin 一致（必须先全对，再谈扫参）
  python sim_bestcrop.py --check \
      --grids-manifest <work-full/grids_manifest.tsv> \
      --grids-bin <work-full/grids.bin> \
      --harness <work-full/harness_full-model-src.tsv>
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

GRID_W = 64
GRID_H = 48
GRID_BYTES = GRID_W * GRID_H * 4

f32 = np.float32
ONE_THIRD = f32(1.0) / f32(3.0)
TWO_THIRDS = f32(2.0) / f32(3.0)

# 生产口径（AutoFrame.kt:779 与 :777）
W_THIRDS = f32(0.55)
W_RETENTION = f32(0.25)
W_AREA = f32(0.20)
CUT_ONSET = f32(0.75)
CUT_SLOPE = f32(1.6)
ABS_GATE_MARGIN = f32(0.05)
MIN_GATE_MARGIN = f32(0.01)
EVAL_MIN_KEEP = f32(0.60)  # EvalHarnessTest.kt:228 传给 bestCrop 的就是 0.60f


def rti(x) -> np.ndarray | int:
    """Kotlin roundToInt：四舍五入、.5 向正无穷。"""
    return np.floor(np.asarray(x, dtype=np.float64) + 0.5)


def rti_scalar(x: float) -> int:
    return int(math.floor(x + 0.5))


def load_grids(path: Path, count: int, cols: int = GRID_W, rows: int = GRID_H) -> np.memmap:
    """每张图 cols×rows 个大端 float32，行主序（i = row*cols + col）。默认 64×48。"""
    nbytes = int(cols) * int(rows) * 4
    size = path.stat().st_size
    avail = size // nbytes
    if avail < count:
        raise SystemExit(f"{path} 只有 {avail} 条记录，清单要 {count} 条")
    return np.memmap(path, dtype=">f4", mode="r", shape=(avail, int(rows), int(cols)))


def integral(a: np.ndarray) -> np.ndarray:
    """带零边界的积分图，方便按 [y0:y1, x0:x1] 取框内和。"""
    s = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.float64)
    s[1:, 1:] = a.astype(np.float64).cumsum(axis=0).cumsum(axis=1)
    return s


def box_sum(sat: np.ndarray, y0, y1, x0, x1):
    """积分图取矩形和，参数都是数组（逐候选框）。"""
    return sat[y1, x1] - sat[y0, x1] - sat[y1, x0] + sat[y0, x0]


def build_candidates(max_wg: int, max_hg: int, cols: int, rows: int,
                     keep: float = EVAL_MIN_KEEP, anchor: int = 0):
    """复现 AutoFrame.kt:597-609 的 21 档尺寸 × 位置枚举（含 Kotlin 的步进写法）。"""
    ks, xks, yks, wks, hks = [], [], [], [], []
    one = f32(1.0)
    span = f32(one - f32(keep))
    for k in range(21):
        # 1f - k * (1f - keep) / 20f
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


def image_features(grid: np.ndarray, src_w: int, src_h: int, aspect: float | None = None,
                   keep: float = EVAL_MIN_KEEP, anchor: int = 0,
                   cols: int = GRID_W, rows: int = GRID_H):
    """算出一张图所有候选框的「与权重无关」特征，外加基准框索引所需的分组信息。

    cols/rows 默认 64×48（原行为逐位不变）；换分辨率时调用方必须传实际网格尺寸，
    否则 grid 的 reshape 会按错的形状解析。

    返回 None 表示命中 Kotlin 的兜底分支（全 0 权重 / 参数非法），调用方应直接用居中最大框。
    """
    cols, rows = int(cols), int(rows)
    # 1) 最大可用框（像素 → 格子），AutoFrame.kt:526-557
    #    aspect=None 表示 src 口径：Kotlin 取 (w.toFloat() / h.toFloat())，是 float32 除法
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

    g = np.asarray(grid, dtype=np.float64).reshape(rows, cols)
    pos = np.where(g > 0.0, g, 0.0)
    total_weight = float(pos.sum())
    max_weight = float(pos.max()) if pos.size else 0.0
    if total_weight <= 0.0 or max_weight <= 0.0:
        return None
    max_wg = min(max(1, rti_scalar(max_wc * cols / src_w)), cols)
    max_hg = min(max(1, rti_scalar(max_hc * rows / src_h)), rows)

    high_threshold = f32(0.6) * f32(max_weight)
    high_mask = g >= float(high_threshold)
    high = np.where(high_mask, g, 0.0)
    high_total = float(high.sum())

    yy = (np.arange(rows, dtype=np.float64) + 0.5)[:, None]
    xx = (np.arange(cols, dtype=np.float64) + 0.5)[None, :]

    sat_w = integral(pos)
    sat_wx = integral(pos * xx)
    sat_wy = integral(pos * yy)
    sat_h = integral(high)
    sat_hx = integral(high * xx)
    sat_hy = integral(high * yy)

    ks, xks, yks, wks, hks = build_candidates(max_wg, max_hg, cols, rows, keep, anchor)
    x1 = xks + wks
    y1 = yks + hks

    in_w = box_sum(sat_w, yks, y1, xks, x1)
    sum_x = box_sum(sat_wx, yks, y1, xks, x1)
    sum_y = box_sum(sat_wy, yks, y1, xks, x1)
    in_h = box_sum(sat_h, yks, y1, xks, x1)
    sum_hx = box_sum(sat_hx, yks, y1, xks, x1)
    sum_hy = box_sum(sat_hy, yks, y1, xks, x1)

    # retention：AutoFrame.kt:739-743
    if high_total > 0.0:
        retention = np.clip((in_h / high_total).astype(f32), f32(0.0), f32(1.0))
    else:
        retention = np.clip((in_w / total_weight).astype(f32), f32(0.0), f32(1.0))

    # thirds：AutoFrame.kt:747-762
    has_h = in_h > 0.0
    cw = np.where(has_h, in_h, in_w)  # sumHW > 0 时用重要格子
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

    # areaRatio：AutoFrame.kt:764-766
    area_ratio = np.clip(((wks.astype(f32) * hks.astype(f32)) /
                          f32(max(1, max_wg) * max(1, max_hg))).astype(f32), f32(0.0), f32(1.0))

    return {
        "ks": ks, "xks": xks, "yks": yks, "wks": wks, "hks": hks,
        "thirds": thirds, "retention": retention, "area": area_ratio,
        "max_wg": max_wg, "max_hg": max_hg, "max_wc": max_wc, "max_hc": max_hc,
    }


def score_candidates(feat, w_thirds=W_THIRDS, w_ret=W_RETENTION, w_area=W_AREA,
                     cut_onset=CUT_ONSET, cut_slope=CUT_SLOPE) -> np.ndarray:
    """scoreCrop 的线性部分，AutoFrame.kt:777-779。"""
    t = (f32(w_thirds) * feat["thirds"]).astype(f32)
    r = (f32(w_ret) * feat["retention"]).astype(f32)
    a = (f32(w_area) * feat["area"]).astype(f32)
    base = ((t + r).astype(f32) + a).astype(f32)
    ret = feat["retention"]
    pen = np.where(ret < f32(cut_onset),
                   ((f32(cut_onset) - ret).astype(f32) * f32(cut_slope)).astype(f32),
                   f32(0.0))
    return (base - pen).astype(f32)


def pick(feat, score, gate="abs", gate_rel=ABS_GATE_MARGIN):
    """复现 AutoFrame.kt:580-682 的选框 + 采纳门槛（返回 best 索引、是否采纳、以及基准分）。"""
    ks = feat["ks"]
    best_i = int(np.argmax(score))  # np.argmax 取首个最大值，与 Kotlin 的严格 > 一致
    base_idx = np.flatnonzero(ks == 0)
    base_i = int(base_idx[int(np.argmax(score[base_idx]))])
    base_score = f32(score[base_i])
    best_score = f32(score[best_i])

    if gate == "abs":
        margin = f32(max(0.0, float(gate_rel)))
    elif gate == "range":
        margin = f32(max(float(MIN_GATE_MARGIN),
                         min(max(float(gate_rel), 0.0), 1.0) * max(0.0, float(score.max() - score.min()))))
    else:  # headroom
        margin = f32(max(float(MIN_GATE_MARGIN),
                         min(max(float(gate_rel), 0.0), 1.0) * max(0.0, 1.0 - float(base_score))))
    adopted = bool(best_score >= f32(base_score + margin))
    return (best_i if adopted else base_i), adopted, base_i, base_score


def to_pixels(feat, i: int, src_w: int, src_h: int,
              cols: int = GRID_W, rows: int = GRID_H):
    """格子框 → 源图像素并夹进画面，AutoFrame.kt:672-682。cols/rows 默认 64×48（原行为不变）。"""
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


def fallback_box(src_w: int, src_h: int, aspect: float | None = None):
    """AutoFrame.kt:526-546 的居中兜底框。"""
    ratio = f32(f32(src_w) / f32(src_h))
    safe_aspect = f32(aspect) if (aspect is not None and math.isfinite(aspect) and aspect > 0) \
        else (ratio if (math.isfinite(float(ratio)) and ratio > f32(0.0)) else f32(1.0))
    if ratio > safe_aspect:
        max_h, max_w = src_h, rti_scalar(float(f32(f32(src_h) * safe_aspect)))
    else:
        max_w, max_h = src_w, rti_scalar(float(f32(f32(src_w) / safe_aspect)))
    max_wc = min(max(1, max_w), max(1, src_w))
    max_hc = min(max(1, max_h), max(1, src_h))
    left = min(max(0, (src_w - max_wc) // 2), max(0, src_w - max_wc))
    top = min(max(0, (src_h - max_hc) // 2), max(0, src_h - max_hc))
    return left, top, max_wc, max_hc, f32(0.0)


def predict(grid: np.ndarray, src_w: int, src_h: int, aspect: float | None = None,
            keep: float = EVAL_MIN_KEEP, anchor: int = 0,
            gate: str = "abs", gate_rel=ABS_GATE_MARGIN, weights=None):
    """一图一算：返回 (left, top, w, h, score, adopted)。weights = (w_thirds, w_ret, w_area[, onset, slope])"""
    feat = image_features(grid, src_w, src_h, aspect, keep, anchor)
    if feat is None:
        l, t, w, h, s = fallback_box(src_w, src_h, aspect)
        return l, t, w, h, s, False
    kw = {}
    if weights:
        kw["w_thirds"], kw["w_ret"], kw["w_area"] = weights[0], weights[1], weights[2]
        if len(weights) > 3:
            kw["cut_onset"], kw["cut_slope"] = weights[3], weights[4]
    score = score_candidates(feat, **kw)
    i, adopted, _base_i, _base = pick(feat, score, gate, gate_rel)
    l, t, w, h = to_pixels(feat, i, src_w, src_h)
    return l, t, w, h, f32(score[i]), adopted


# ---------------------------------------------------------------- 校验

def read_manifest(path: Path):
    rows = []
    with path.open(encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if parts[0].strip().lower() == "index":
                continue
            if len(parts) < 4:
                continue
            rows.append((int(parts[0]), parts[1].strip(), int(parts[2]), int(parts[3])))
    return rows


def read_harness(path: Path):
    out = {}
    with path.open(encoding="utf-8-sig") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        idx = {name: i for i, name in enumerate(header)}
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            out[parts[idx["image"]]] = parts
    return out, idx


def cmd_check(args):
    rows = read_manifest(Path(args.grids_manifest))
    grids = load_grids(Path(args.grids_bin), max(r[0] for r in rows) + 1)
    harness, idx = read_harness(Path(args.harness))
    if args.limit:
        rows = rows[: args.limit]

    keep = f32(args.keep)
    mismatch = []
    compared = 0
    for index, image, w, h in rows:
        ref = harness.get(image)
        if ref is None:
            continue
        compared += 1
        aspect = None if args.aspect == "src" else float(args.aspect)
        l, t, cw, ch, sc, adopted = predict(
            grids[index], w, h, aspect=aspect, keep=keep, anchor=args.anchor,
            gate=args.gate, gate_rel=f32(args.gate_rel))
        ref_box = (int(ref[idx["crop_left"]]), int(ref[idx["crop_top"]]),
                   int(ref[idx["crop_w"]]), int(ref[idx["crop_h"]]))
        if (l, t, cw, ch) != ref_box:
            mismatch.append((image, ref_box, (l, t, cw, ch)))

    print(f"对比 {compared} 行，不一致 {len(mismatch)} 行")
    for image, ref_box, sim_box in mismatch[:10]:
        print(f"  {Path(image).name}\n    kotlin={ref_box}\n    sim   ={sim_box}")
    return 0 if not mismatch else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="复现 AutoFrame.bestCrop 并校验/扫参")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="逐行校验复现结果与真 Kotlin harness 是否一致")
    c.add_argument("--grids-manifest", required=True)
    c.add_argument("--grids-bin", required=True)
    c.add_argument("--harness", required=True)
    c.add_argument("--aspect", default="src")
    c.add_argument("--gate", default="abs", choices=["abs", "range", "headroom"])
    c.add_argument("--gate-rel", type=float, default=float(ABS_GATE_MARGIN))
    c.add_argument("--anchor", type=int, default=0)
    c.add_argument("--keep", type=float, default=float(EVAL_MIN_KEEP))
    c.add_argument("--limit", type=int, default=0)
    c.set_defaults(func=cmd_check)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
