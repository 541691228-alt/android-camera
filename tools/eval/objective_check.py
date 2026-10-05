#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 4 条：把有退化解的目标函数换掉，并在新度量下重判"翻正"结论。

背景
----
评测主口径 kept_ratio（框内真值主体像素 / 全图主体像素）在 **src（原图比例）** 下有一个
永远可达、且不需要任何图像信息就能拿到的退化解：**裁剪框 = 整图**，此时 kept_ratio ≡ 1.0000。

    sup_{B ⊆ 画面} kept_ratio(B) = 1，取到它的 B 是整图（候选集里 k=0 档，宽高等于原图）。
    任何真裁剪 B ⊊ 画面都有 kept_ratio(B) ≤ 1。

所以"不裁"是这个度量下的全局最优。它无法区分"算法真的框对了主体"和"算法什么都不做"。
同尺寸居中对照之所以常常赢，根因也在这里：算法一旦裁小，它的 kept 就要跟"自己不裁时的 1.0"
抢分；而居中对照的尺寸跟着算法走，在主体本来就靠中间的图上天然占便宜。

本脚本做四件事：
  1. 形式化这个退化解，并给最小反例表（同一张图：整图 / 同尺寸居中 / 算法框）。
  2. 实现 4 个没有无条件退化解的候选度量（公式见下）。
  3. 在 DUTS-TE 的 src / 1:1 / 3:4 上，用这 4 个度量重排
     {基线权重 0.55/0.25/0.20, 新权重 0.40/0.60/0.00, 不裁(同比例最大居中框), 同尺寸居中}。
  4. 报"翻正"是否随度量改变、哪个度量最能区分"真裁剪"与"不裁"。

度量定义（S = 真值主体像素集，B = 裁剪框像素集，|·| 为像素数）
--------------------------------------------------------------
  r   = |S∩B| / |S|                    （= kept_ratio，范围 [0,1]，**有退化解**，只作对照）
  d   = 主体质心到最近三分点的欧氏距离（框内归一化坐标，与 sweep_weights.py 同口径，不夹取）
  c   = clip(1 − d/0.2357, 0, 1)       （构图质量；0.2357 = 画面正中到最近三分点，c∈[0,1]）
  M1 J  = r · c                        （联合分·乘积：不裁时若主体居中，c≈0 → J≈0）
  M2 H  = 2rc/(r+c)                    （联合分·调和平均：任一侧为 0 则 0，惩罚更狠）
  M3 F  = 2|S∩B| / (|S| + |B|)         （Dice/F1：precision 项按框面积惩罚，不裁的框最大最吃亏）
  M4 Rc = r · 1[d ≤ T], T = 0.10       （构图约束下的保留率：构图不达标直接记 0）
  辅助 Δc = r − r_同尺寸居中            （相对居中的成对增益；居中恒为 0）

只读依赖：metrics.py / sim_bestcrop.py / sweep_weights.py。本轮不改任何共享文件，
也不动 app/ 下的 Kotlin 生产代码。
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

import sim_bestcrop as sb            # noqa: E402
import sweep_weights as sw           # noqa: E402

# Windows 控制台默认 GBK，文档里的 '↳' 之类字符会让 print 抛 UnicodeEncodeError。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

F32 = np.float32
GRID_W, GRID_H = 64, 48
DMAX = math.hypot(1.0 / 6.0, 1.0 / 6.0)   # 0.23570226039551587，画面正中到最近三分点
T_COMPOSE = 0.10                          # M4 的构图可行阈值，与 ceiling_check.py 的 TS 一致

BASE_W = (0.55, 0.25, 0.20, 0.75, 1.6)    # 现行生产权重（AutoFrame.kt:777-779）
NEW_W = (0.40, 0.60, 0.00, 0.75, 1.6)     # RESULTS-weight-sweep.md 推荐权重
POLICIES = (("base", BASE_W), ("new", NEW_W))

# 单框度量键（box/inside 另外带）
MKEYS = ("kept", "J", "H", "F", "Rc", "thirds", "comp", "area")


# ------------------------------------------------------------------ 单框度量
def box_metrics(I, total, box, w, h, cx_n, cy_n):
    """一个像素框 (left, top, cw, ch) 的全部度量；框无效返回 None。"""
    l, t, cw, ch = box
    x0, y0 = max(0, int(l)), max(0, int(t))
    x1, y1 = min(w, int(l) + int(cw)), min(h, int(t) + int(ch))
    if x1 <= x0 or y1 <= y0:
        return None
    inside = int(I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0])
    r = inside / float(total)
    area_px = (x1 - x0) * (y1 - y0)
    F = 2.0 * inside / float(total + area_px)
    ccx = (cx_n * w - x0) / float(x1 - x0)
    ccy = (cy_n * h - y0) / float(y1 - y0)
    d = min(math.hypot(ccx - px, ccy - py) for px, py in sw.THIRDS_PTS)
    c = max(0.0, min(1.0, 1.0 - d / DMAX))
    return {"box": [x0, y0, x1 - x0, y1 - y0], "inside": inside,
            "kept": r, "thirds": d, "comp": c, "J": r * c,
            "H": (2.0 * r * c / (r + c)) if (r + c) > 0 else 0.0,
            "F": F, "Rc": (r if d <= T_COMPOSE else 0.0),
            "area": area_px / float(w * h)}


def cand_arrays(feat, w, h, cx_n, cy_n, total, I, cols=GRID_W, rows=GRID_H):
    """整批候选框的度量（向量化，取框口径与 sweep_weights.evaluate 一致）。"""
    left = sw.rti_vec(feat["xks"].astype(np.float64) * w / cols)
    top = sw.rti_vec(feat["yks"].astype(np.float64) * h / rows)
    out_w = np.maximum(1, sw.rti_vec(feat["wks"].astype(np.float64) * w / cols))
    out_h = np.maximum(1, sw.rti_vec(feat["hks"].astype(np.float64) * h / rows))
    x0, y0, ww, hh = sw.clip_box(left, top, out_w, out_h, w, h)
    x1, y1 = x0 + ww, y0 + hh
    inside = (I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0]).astype(np.float64)
    r = inside / float(total)
    area_px = (ww * hh).astype(np.float64)
    F = 2.0 * inside / (float(total) + area_px)
    ccx = (cx_n * w - x0) / np.maximum(ww, 1).astype(np.float64)
    ccy = (cy_n * h - y0) / np.maximum(hh, 1).astype(np.float64)
    d = sw.thirds_dist(ccx, ccy)
    c = np.clip(1.0 - d / DMAX, 0.0, 1.0)
    H = np.where((r + c) > 0, 2.0 * r * c / np.maximum(r + c, 1e-12), 0.0)
    return {"kept": r, "F": F, "J": r * c, "H": H, "Rc": np.where(d <= T_COMPOSE, r, 0.0),
            "thirds": d, "comp": c, "inside": inside,
            "area": area_px / float(w * h),
            "box": np.stack([x0, y0, ww, hh], axis=1)}


def select_index(feat, weights, gate_rel):
    """复现 sweep_weights.evaluate 的打分 + ABS 采纳门槛，返回被选中的候选下标。"""
    wt, wr, wa, onset, slope = weights
    T = feat["thirds"]; R = feat["retention"]; A = feat["area"]
    SC = (T * F32(wt)).astype(F32)
    SC = (SC + (R * F32(wr))).astype(F32)
    SC = (SC + (A * F32(wa))).astype(F32)
    pen = np.maximum(F32(0.0), F32(onset) - R).astype(F32)
    SC = (SC - (pen * F32(slope))).astype(F32)
    best_i = int(np.argmax(SC))
    base_pool = np.flatnonzero(feat["ks"] == 0)
    base_i = int(base_pool[int(np.argmax(SC[base_pool]))])
    adopted = bool(F32(SC[best_i]) >= F32(SC[base_i] + F32(max(0.0, float(gate_rel)))))
    return (best_i if adopted else base_i), adopted


def crop_from_feat(feat, i, w, h, cols=GRID_W, rows=GRID_H):
    """候选下标 → 像素框。"""
    l = sw.rti_vec(np.float64(feat["xks"][i]) * w / cols)
    t = sw.rti_vec(np.float64(feat["yks"][i]) * h / rows)
    cw = max(1, sw.rti_vec(np.float64(feat["wks"][i]) * w / cols))
    ch = max(1, sw.rti_vec(np.float64(feat["hks"][i]) * h / rows))
    x0, y0, ww, hh = sw.clip_box(np.array([l]), np.array([t]), np.array([cw]),
                                 np.array([ch]), w, h)
    return [int(x0[0]), int(y0[0]), int(ww[0]), int(hh[0])]


def parse_aspect(s):
    """'src' → None；'1:1'/'3:4' → 1.0/0.75；'0.75' 原样。与 sweep_weights --aspect 口径一致。"""
    if s == "src":
        return None
    if ":" in s:
        a, b = s.split(":")
        return float(a) / float(b)
    return float(s)


# ------------------------------------------------------------------ 主评估
def eval_aspect(pairs, bin_path, aspect_name, gate_rel, limit=0, verbose=True,
                cols=GRID_W, rows=GRID_H, max_examples=600):
    aspect = parse_aspect(aspect_name)
    rows_sel = pairs[:limit] if limit else pairs

    acc = {}
    names = [n for n, _ in POLICIES]
    for n in names:
        for k in MKEYS:
            acc[f"{n}_{k}"] = []
            acc[f"c_{n}_{k}"] = []          # 同尺寸居中
        for mk in MKEYS:
            acc[f"{n}_d{mk}"] = []
            acc[f"win_{mk}_{n}"] = []       # 单图：算法框严格优于同尺寸居中
        for mk in ("kept", "J", "H", "F"):
            acc[f"beat_{mk}_{n}"] = []      # 单图：算法框严格优于不裁(最大框)
    for k in MKEYS:
        acc[f"max_{k}"] = []                # 不裁：本比例最大居中框（src 下＝整图）
    for k in MKEYS:
        acc[f"best_{k}"] = []               # 候选集里该度量的最优值（oracle）
        acc[f"amax_area_{k}"] = []          # 该度量最优候选的面积占比
        acc[f"amax_kept_{k}"] = []          # 该度量最优候选的 kept
        acc[f"amax_thirds_{k}"] = []        # 该度量最优候选的 thirds
    oracle_hit = {k: [] for k in MKEYS}
    oracle_gap = {k: [] for k in ("J", "H", "F")}
    examples = []
    n_img = 0
    t0 = time.time()

    with open(bin_path, "rb") as bf:
        for n, (idx, image, w, h, mp) in enumerate(rows_sel):
            try:
                mask = np.array(Image.open(mp).convert("L")) >= 128
            except Exception:
                continue
            if not mask.any():
                continue
            total = int(mask.sum())
            I = sw.mask_integral(mask)
            ys, xs = np.nonzero(mask)
            cx_n = (float(xs.mean()) + 0.5) / float(w)
            cy_n = (float(ys.mean()) + 0.5) / float(h)
            grid = sw.read_grid(bf, idx, cols, rows)
            feat = sb.image_features(grid, w, h, aspect, sb.EVAL_MIN_KEEP, 0)

            mm = box_metrics(I, total, sb.fallback_box(w, h, aspect)[:4], w, h, cx_n, cy_n)
            for k in MKEYS:
                acc[f"max_{k}"].append(mm[k])

            per = {}
            if feat is None:      # Kotlin 兜底：所有候选取同一个居中最大框
                for nm in names:
                    per[nm] = dict(mm)
                best = {k: mm[k] for k in MKEYS}
                amax = {k: (mm["area"], mm["kept"], mm["thirds"]) for k in MKEYS}
            else:
                ca = cand_arrays(feat, w, h, cx_n, cy_n, total, I, cols, rows)
                for k in MKEYS:
                    a = int(np.argmax(ca[k]))
                    oracle_hit[k].append(1.0 if feat["ks"][a] == 0 else 0.0)
                for k in ("J", "H", "F"):
                    oracle_gap[k].append(float(ca[k].max()) - float(mm[k]))
                best = {k: float(ca[k].max()) for k in MKEYS}
                amax = {k: (float(ca["area"][int(np.argmax(ca[k]))]),
                            float(ca["kept"][int(np.argmax(ca[k]))]),
                            float(ca["thirds"][int(np.argmax(ca[k]))])) for k in MKEYS}
                for nm, wts in POLICIES:
                    si, _ad = select_index(feat, wts, gate_rel)
                    p = {k: float(ca[k][si]) for k in MKEYS}
                    p["box"] = [int(v) for v in ca["box"][si]]
                    p["inside"] = float(ca["inside"][si])
                    p["adopted"] = _ad
                    per[nm] = p
            for k in MKEYS:
                acc[f"best_{k}"].append(best[k])
                acc[f"amax_area_{k}"].append(amax[k][0])
                acc[f"amax_kept_{k}"].append(amax[k][1])
                acc[f"amax_thirds_{k}"].append(amax[k][2])

            for nm in names:
                p = per[nm]
                for k in MKEYS:
                    acc[f"{nm}_{k}"].append(p[k])
                bx = p["box"]
                cl = (w - bx[2]) // 2
                ct = (h - bx[3]) // 2
                cm = box_metrics(I, total, (cl, ct, bx[2], bx[3]), w, h, cx_n, cy_n)
                for k in MKEYS:
                    acc[f"c_{nm}_{k}"].append(cm[k])
                    acc[f"{nm}_d{k}"].append(p[k] - cm[k])
                    acc[f"win_{k}_{nm}"].append(1.0 if p[k] > cm[k] + 1e-12 else 0.0)
                for mk in ("kept", "J", "H", "F"):
                    acc[f"beat_{mk}_{nm}"].append(1.0 if p[mk] > mm[mk] + 1e-12 else 0.0)
                p["center"] = cm

            if len(examples) < max_examples:
                examples.append({"image": image, "w": w, "h": h, "max": mm,
                                 "base": per["base"], "new": per["new"]})
            n_img += 1
            if verbose and (n + 1) % 1000 == 0:
                print(f"  ... {n+1}/{len(rows_sel)}  {time.time()-t0:.1f}s", flush=True)

    out = {"aspect": aspect_name, "n": n_img, "seconds": round(time.time() - t0, 2),
           "dmax": DMAX, "T_compose": T_COMPOSE, "gate_rel": gate_rel}
    for k, v in acc.items():
        out[k] = float(np.mean(v)) if len(v) else float("nan")
    out["oracle_hit"] = {k: (float(np.mean(v)) if v else float("nan")) for k, v in oracle_hit.items()}
    out["oracle_gap"] = {k: (float(np.mean(v)) if v else float("nan")) for k, v in oracle_gap.items()}
    for nm in names:
        for k in MKEYS:
            d = np.asarray(acc[f"{nm}_d{k}"], dtype=np.float64)
            sd = float(d.std(ddof=1)) if d.size > 1 else 0.0
            out[f"t_{k}_{nm}"] = float(d.mean() / (sd / math.sqrt(d.size))) if sd > 0 else 0.0
    out["_examples"] = examples
    return out


# ------------------------------------------------------------------ 反例
def pick_counterexample(res, which="base", mode="center_wins"):
    """mode=center_wins：同尺寸居中在 kept 上领先算法最多的那张图。"""
    ex = res.get("_examples", [])
    if not ex:
        return None
    if mode == "center_wins":
        return min(ex, key=lambda e: e[which]["kept"] - e[which]["center"]["kept"])
    return max(ex, key=lambda e: e[which]["kept"] - e[which]["center"]["kept"])


def print_example(e, which="base"):
    if not e:
        print("（没有可用反例）")
        return
    print(f"\n-- 最小反例（{Path(e['image']).name}  {e['w']}x{e['h']}）--")
    print(f"{'框':<12} {'(l,t,w,h)':<24} {'面积占比':>8} {'kept':>7} {'thirds':>7} "
          f"{'comp':>6} {'J':>7} {'F(Dice)':>8} {'Rc':>7}")
    rows = [("整图/不裁", e["max"]),
            ("同尺寸居中", e[which]["center"]),
            (f"{which}算法框", e[which])]
    for label, m in rows:
        b = m["box"]
        print(f"{label:<12} {str(tuple(b)):<24} {m['area']:>8.3f} {m['kept']:>7.4f} "
              f"{m['thirds']:>7.4f} {m['comp']:>6.4f} {m['J']:>7.4f} {m['F']:>8.4f} {m['Rc']:>7.4f}")


# ------------------------------------------------------------------ 打印
def show(res):
    a = res["aspect"]
    print(f"\n===== aspect={a}  n={res['n']}  ({res['seconds']}s) =====")
    print(f"{'策略':<18} {'kept':>7} {'comp':>6} {'J':>7} {'H':>7} {'F(Dice)':>8} "
          f"{'Rc':>7} {'thirds':>7} {'面积':>6}")
    for nm in ("base", "new"):
        tag = "基线0.55/0.25/0.20" if nm == "base" else "新0.40/0.60/0.00"
        print(f"{tag+' 算法':<18} {res[f'{nm}_kept']:>7.4f} {res[f'{nm}_comp']:>6.4f} "
              f"{res[f'{nm}_J']:>7.4f} {res[f'{nm}_H']:>7.4f} {res[f'{nm}_F']:>8.4f} "
              f"{res[f'{nm}_Rc']:>7.4f} {res[f'{nm}_thirds']:>7.4f} {res[f'{nm}_area']:>6.3f}")
        print(f"{'  ↳ 同尺寸居中':<18} {res[f'c_{nm}_kept']:>7.4f} {res[f'c_{nm}_comp']:>6.4f} "
              f"{res[f'c_{nm}_J']:>7.4f} {res[f'c_{nm}_H']:>7.4f} {res[f'c_{nm}_F']:>8.4f} "
              f"{res[f'c_{nm}_Rc']:>7.4f} {res[f'c_{nm}_thirds']:>7.4f} {res[f'c_{nm}_area']:>6.3f}")
    print(f"{'不裁(最大框)':<18} {res['max_kept']:>7.4f} {res['max_comp']:>6.4f} "
          f"{res['max_J']:>7.4f} {res['max_H']:>7.4f} {res['max_F']:>8.4f} "
          f"{res['max_Rc']:>7.4f} {res['max_thirds']:>7.4f} {res['max_area']:>6.3f}")

    print("\n-- 算法 − 同尺寸居中（配对差；win=算法严格领先的图片占比）--")
    print(f"{'指标':<8} {'Δbase':>9} {'t':>7} {'win':>7} | {'Δnew':>9} {'t':>7} {'win':>7}")
    for k in MKEYS:
        print(f"{k:<8} {res[f'base_d{k}']:>+9.5f} {res[f't_{k}_base']:>7.2f} "
              f"{res[f'win_{k}_base']*100:>6.1f}% | {res[f'new_d{k}']:>+9.5f} "
              f"{res[f't_{k}_new']:>7.2f} {res[f'win_{k}_new']*100:>6.1f}%")
    print("（kept 行即已发布口径：base Δ−0.00587 / new Δ+0.00469）")

    print("\n-- 退化性：候选集里各度量的最优值 vs 不裁框 --")
    print(f"{'指标':<8} {'不裁框':>8} {'居中(新)':>8} {'算法(新)':>8} {'候选最优':>8} {'最优在ks=0':>10}")
    for k in MKEYS:
        print(f"{k:<8} {res[f'max_{k}']:>8.4f} {res[f'c_new_{k}']:>8.4f} "
              f"{res[f'new_{k}']:>8.4f} {res[f'best_{k}']:>8.4f} "
              f"{res['oracle_hit'][k]*100:>9.1f}%")

    print("\n-- 各度量自己的最优候选长什么样（候选集 argmax）--")
    print(f"{'指标':<8} {'最优值':>8} {'该候选面积':>10} {'该候选kept':>10} {'该候选thirds':>12}")
    for k in MKEYS:
        print(f"{k:<8} {res[f'best_{k}']:>8.4f} {res[f'amax_area_{k}']:>10.3f} "
              f"{res[f'amax_kept_{k}']:>10.4f} {res[f'amax_thirds_{k}']:>12.4f}")

    print("\n-- 同图对比：算法框 vs 不裁(最大框)，度量更高的比例 --")
    print(f"{'策略':<8} {'kept':>8} {'J':>8} {'H':>8} {'F(Dice)':>8}")
    for nm in ("base", "new"):
        print(f"{nm:<8} {res[f'beat_kept_{nm}']*100:>7.1f}% {res[f'beat_J_{nm}']*100:>7.1f}% "
              f"{res[f'beat_H_{nm}']*100:>7.1f}% {res[f'beat_F_{nm}']*100:>7.1f}%")


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description="无退化目标函数：定义 + 重判")
    ap.add_argument("--grids-manifest", required=True)
    ap.add_argument("--grids-bin", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--aspects", nargs="*", default=["src"])
    ap.add_argument("--gate-rel", type=float, default=float(sb.ABS_GATE_MARGIN))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default=r"D:\spider\.cache\eval-duts\objective")
    ap.add_argument("--tag", default="duts")
    ap.add_argument("--no-json", action="store_true")
    args = ap.parse_args(argv)

    pairs = sw.load_pairs(args.grids_manifest, args.manifest)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"图文对 {len(pairs)} 条 | aspects={args.aspects} | gate_rel={args.gate_rel}"
          f" | limit={args.limit}")

    for a in args.aspects:
        res = eval_aspect(pairs, args.grids_bin, a, args.gate_rel, limit=args.limit)
        show(res)
        e = pick_counterexample(res, "base", "center_wins")
        print_example(e, "base")
        ex = res.pop("_examples", [])
        if not args.no_json:
            stem = f"objective_{args.tag}_{a.replace(':', 'to')}"
            (outdir / f"{stem}.json").write_text(
                json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            keep = [{**{k: v for k, v in x.items() if k in ("image", "w", "h")},
                     "max": x["max"], "base": x["base"], "new": x["new"]} for x in ex[:800]]
            (outdir / f"{stem}_examples.json").write_text(
                json.dumps(keep, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"已写 {outdir / (stem + '.json')}")


if __name__ == "__main__":
    main()
