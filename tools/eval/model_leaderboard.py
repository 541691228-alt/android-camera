#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 1 条：显著度模型排行榜（新权重 0.40/0.60/0.00 下换模型值多少）。

在已定稿的评分权重 `0.40/0.60/0.00`（RESULTS-weight-sweep.md §4）下，对
u2netp（产线）/ u2net / RMBG-1.4 / ISNet / BiRefNet_lite 排一个榜：
每个数据集 × 口径（src / 1:1 / 3:4）× 模型给出算法保留率 A、同尺寸居中保留率、差值 Δ 与配对 t、
三分线距离、切主体率、采纳率、裁剪面积、整图框率，外加两个模型无关的量：
  · B = 给"完美先验"（真值掩膜下采样到 64×48）跑同一套权重的保留率 → B−A 就是"显著度模型层的差距"
  · 中位 IoU = 网格二值化 >0.5 对真值掩膜缩到 64×48 的 IoU

实现上不做任何新口径：逐图特征走 `sim_bestcrop.image_features`，选框/采纳门槛走 sweep_weights 里
那套（与 `sweep_weights.evaluate` 逐位一致），完美先验生成走 `oracle_prior_check.perfect_grid`，
掩膜加载走 `metrics.load_mask`。一图只加载一次掩膜，多模型 / 多权重 / 多口径复用。

子命令：
  subset  从数据集真值清单里抽 n 张（固定 seed）写成 image<TAB>mask 子集清单
  probe   某个 onnx 的"口径校验"：原生分辨率中位 IoU + ms/图（支持 --sigmoid）
  run     跑排行榜，每个数据集写一份 JSON
  tables  把 JSON 渲染成 markdown 表

示例（全程 --threads 4，绝不改 app/ 下任何生产代码）：
  python model_leaderboard.py subset --dataset duts --n 600 --out D:\\spider\\.cache\\saliency-swap\\birefnet\\subset_duts600.tsv
  python model_leaderboard.py probe --model D:\\spider\\.cache\\saliency-swap\\models\\birefnet_lite.onnx `
      --manifest D:\\spider\\.cache\\eval-duts\\work-full\\manifest.tsv --side 1024 --norm imagenet --sigmoid --n 20 --threads 4
  python model_leaderboard.py run --tag full --models u2netp u2net rmbg isnet
  python model_leaderboard.py run --tag biref --models u2netp u2net rmbg isnet birefnet
  python model_leaderboard.py tables
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import metrics                      # noqa: E402
import sim_bestcrop as sim          # noqa: E402
import sweep_weights as sw          # noqa: E402
from oracle_prior_check import perfect_grid   # noqa: E402

GRID_W, GRID_H = 64, 48
REC_BYTES = GRID_W * GRID_H * 4
CUT_TOL = 0.99
DEAD_CENTER_TOL = 0.05
F32 = np.float32
EVAL_KEEP = 0.60
ANCHOR = 0
GATE_REL = float(sim.ABS_GATE_MARGIN)

LOG_DIR = Path(r"D:\spider\.cache\saliency-swap\logs")
BIREF_DIR = Path(r"D:\spider\.cache\saliency-swap\birefnet")
SWAP = Path(r"D:\spider\.cache\saliency-swap")
DUTS_WORK = Path(r"D:\spider\.cache\eval-duts\work-full")
ECSSD_WORK = Path(r"D:\spider\.cache\eval-duts2\ECSSD\work")
OMRON_WORK = Path(r"D:\spider\.cache\eval-duts2\DUT-OMRON\work")

DATASETS = {
    "duts": {"label": "DUTS-TE", "manifest": DUTS_WORK / "manifest.tsv"},
    "ecssd": {"label": "ECSSD", "manifest": Path(r"D:\spider\.cache\eval-duts2\ECSSD\manifest.tsv")},
    "omron": {"label": "DUT-OMRON", "manifest": Path(r"D:\spider\.cache\eval-duts2\DUT-OMRON\manifest.tsv")},
}

MODEL_LABELS = {
    "u2netp": "u2netp（产线 4.6MB / 320²）",
    "u2net": "u2net（176MB / 320²）",
    "rmbg": "RMBG-1.4（176MB / 1024² mean0.5）",
    "isnet": "ISNet（176MB / 1024²）",
    "birefnet": "BiRefNet_lite（224MB / 1024² sigmoid）",
}

MODELS_DIRS = {
    "u2netp": {"duts": DUTS_WORK, "ecssd": ECSSD_WORK, "omron": OMRON_WORK},
    "u2net": {ds: SWAP / "u2net" / sub for ds, sub in
              (("duts", "duts-te"), ("ecssd", "ecssd"), ("omron", "omron"))},
    "rmbg": {ds: SWAP / "rmbg" / sub for ds, sub in
             (("duts", "duts-te"), ("ecssd", "ecssd"), ("omron", "omron"))},
    # ISNet 的 600 子集网格由 Lead 生成；按 Lead 要求 out-dir 统一为 isnet\duts
    # （不是 duts-te），与 u2net/rmbg 的全量目录名不同，这里显式区分。
    "isnet": {ds: SWAP / "isnet" / sub for ds, sub in
              (("duts", "duts"), ("ecssd", "ecssd"), ("omron", "omron"))},
    "birefnet": {ds: BIREF_DIR / sub for ds, sub in
                 (("duts", "duts-te"), ("ecssd", "ecssd"), ("omron", "omron"))},
}

# 主权重（RESULTS-weight-sweep.md 定稿）与基线权重
W_NEW = (0.40, 0.60, 0.00, 0.75, 1.6)
W_BASE = (0.55, 0.25, 0.20, 0.75, 1.6)
DEFAULT_WEIGHTS = (W_BASE, W_NEW)
PRIMARY = 1  # 主表用 W_NEW


# ----------------------------------------------------------------- 读写

def read_tsv(path):
    with open(path, encoding="utf-8-sig") as f:
        head = f.readline().rstrip("\n").split("\t")
        rows = [ln.rstrip("\n").split("\t") for ln in f if ln.strip()]
    return head, rows


def aspect_of(spec):
    return None if spec == "src" else float(spec)


def aspect_name(a):
    if a is None:
        return "src"
    return "1" if abs(float(a) - 1.0) < 1e-12 else str(a)


def load_gt(ds):
    """真值清单 → [(image, mask)]（保持清单顺序）。"""
    _, rows = read_tsv(DATASETS[ds]["manifest"])
    out = []
    for r in rows:
        if len(r) >= 2 and r[0] and r[1]:
            out.append((r[0], r[1]))
    return out


def load_grid_index(d):
    """模型网格目录 → {image_basename: (idx, w, h)}；目录/记录不齐时返回 None。"""
    d = Path(d)
    man, bp = d / "grids_manifest.tsv", d / "grids.bin"
    if not man.is_file() or not bp.is_file():
        return None, d, "目录或清单不存在"
    size = bp.stat().st_size
    if size == 0:
        return None, d, "grids.bin 为空（生成任务还没落盘）"
    _, rows = read_tsv(man)
    avail = size // REC_BYTES
    if avail < len(rows):
        return None, d, f"grids.bin 只有 {avail} 条，清单 {len(rows)} 条（还在写？）"
    idx = {}
    for r in rows:
        if len(r) < 4:
            continue
        idx[Path(r[1]).name] = (int(r[0]), int(r[2]), int(r[3]))
    return idx, d, ""


def open_memmap(d):
    return sim.load_grids(Path(d) / "grids.bin", 0)


# ----------------------------------------------------------------- 核心评估

def eval_image(feat, I, total, cx_n, cy_n, w, h, aspect, W, cols=GRID_W, rows=GRID_H):
    """一张图 × 一组权重（nW×5）→ 每组的逐图指标（与 sweep_weights.evaluate 同一条代码路径）。"""
    nW = W.shape[0]
    wt, wr, wa, onset, slope = (W[:, i].astype(F32) for i in range(5))

    if feat is None:
        l, t, cw, ch, _ = sim.fallback_box(w, h, aspect)
        left = np.full(nW, l, dtype=np.int64)
        top = np.full(nW, t, dtype=np.int64)
        out_w = np.full(nW, cw, dtype=np.int64)
        out_h = np.full(nW, ch, dtype=np.int64)
        adopted = np.zeros(nW, dtype=bool)
    else:
        T, R, A = feat["thirds"], feat["retention"], feat["area"]
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
        base_s = sub[bsel, np.arange(nW)]
        best_s = SC.max(axis=0)
        margin = F32(max(0.0, GATE_REL))
        adopted = best_s >= (base_s + margin).astype(F32)
        sel = np.where(adopted, best_i, base_i)

        left = sw.rti_vec(feat["xks"][sel].astype(np.float64) * w / cols)
        top = sw.rti_vec(feat["yks"][sel].astype(np.float64) * h / rows)
        out_w = np.maximum(1, sw.rti_vec(feat["wks"][sel].astype(np.float64) * w / cols))
        out_h = np.maximum(1, sw.rti_vec(feat["hks"][sel].astype(np.float64) * h / rows))

    x0, y0, width_c, height_c = sw.clip_box(left, top, out_w, out_h, w, h)
    x1, y1 = x0 + width_c, y0 + height_c
    kept = sw.box_kept(I, total, x0, y0, x1, y1)

    cl = (w - width_c) // 2
    ct = (h - height_c) // 2
    kept_c = sw.box_kept(I, total, cl, ct, cl + width_c, ct + height_c)

    ccx = (cx_n * w - x0) / (x1 - x0).astype(np.float64)
    ccy = (cy_n * h - y0) / (y1 - y0).astype(np.float64)
    ccx_c = (cx_n * w - cl) / width_c.astype(np.float64)
    ccy_c = (cy_n * h - ct) / height_c.astype(np.float64)
    td = sw.thirds_dist(ccx, ccy)
    td_c = sw.thirds_dist(ccx_c, ccy_c)
    dc = np.hypot(ccx - 0.5, ccy - 0.5)

    nsm = 1.0 - ((width_c * height_c) >= 0.97 * w * h).astype(np.float64)
    return {
        "kept": kept, "kept_c": kept_c, "delta": kept - kept_c,
        "kept_ns": kept * nsm, "ns": nsm,
        "thirds": td, "thirds_c": td_c,
        "cut": (kept < CUT_TOL).astype(np.float64),
        "cut_c": (kept_c < CUT_TOL).astype(np.float64),
        "dead": (dc <= DEAD_CENTER_TOL).astype(np.float64),
        "skip": ((width_c * height_c) >= 0.97 * w * h).astype(np.float64),
        "area": (width_c * height_c).astype(np.float64) / float(w * h),
        "adopted": adopted.astype(np.float64),
        "win": (kept > kept_c).astype(np.float64),
        "tie": (np.abs(kept - kept_c) < 1e-9).astype(np.float64),
    }


class Acc:
    """按权重组累加逐图指标。"""

    KEYS = ("kept", "kept_c", "thirds", "thirds_c", "cut", "cut_c", "dead", "skip",
            "area", "adopted", "win", "tie", "kept_ns", "ns")

    def __init__(self, nW):
        self.nW = nW
        self.sum = {k: np.zeros(nW, dtype=np.float64) for k in self.KEYS}
        self.deltas = []
        self.skipmask = []
        self.n = 0

    def add(self, r):
        for k in self.KEYS:
            self.sum[k] += r[k]
        self.deltas.append(r["delta"])
        self.skipmask.append(1.0 - r["ns"])
        self.n += 1

    def result(self):
        n = float(self.n) if self.n else float("nan")
        D = np.stack(self.deltas, axis=0) if self.deltas else np.zeros((1, self.nW))
        S = np.stack(self.skipmask, axis=0) if self.skipmask else np.zeros((1, self.nW))
        mean_d = D.mean(axis=0)
        sd = D.std(axis=0, ddof=1) if D.shape[0] > 1 else np.zeros(self.nW)
        se = sd / np.sqrt(D.shape[0]) if D.shape[0] > 1 else np.zeros(self.nW)
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.where(se > 0, mean_d / se, 0.0)
        # 去掉整图兜底（skip）后的对照：隔离"选择规则兜底"与"显著度质量"两种效应
        mean_d_ns = np.zeros(self.nW)
        t_ns = np.zeros(self.nW)
        for j in range(self.nW):
            col = D[S[:, j] == 0, j]
            if col.size > 1:
                mean_d_ns[j] = col.mean()
                se2 = col.std(ddof=1) / np.sqrt(col.size)
                t_ns[j] = mean_d_ns[j] / se2 if se2 > 0 else 0.0
        out = {"n": int(self.n), "mean_delta": mean_d.tolist(), "t_stat": t.tolist(),
               "mean_delta_noskip": mean_d_ns.tolist(), "t_stat_noskip": t_ns.tolist()}
        for k in self.KEYS:
            out["mean_" + k] = (self.sum[k] / n).tolist()
        cnt = np.where(self.sum["ns"] > 0, self.sum["ns"], np.nan)
        out["mean_kept_noskip"] = (self.sum["kept_ns"] / cnt).tolist()
        out["win_rate"] = out.pop("mean_win")
        out["tie_rate"] = out.pop("mean_tie")
        out["lose_rate"] = [1.0 - w - t2 for w, t2 in zip(out["win_rate"], out["tie_rate"])]
        return out


def run_dataset(ds, model_names, aspects, W, weights_desc, subset_keys=None, verbose=True):
    gt = load_gt(ds)
    if subset_keys is not None:
        gt = [(im, mk) for (im, mk) in gt if Path(im).name in subset_keys]
    info, skipped = {}, {}
    for m in model_names:
        idx, d, why = load_grid_index(MODELS_DIRS[m][ds])
        if idx is None:
            skipped[m] = f"{d}：{why}"
            continue
        info[m] = {"dir": str(d), "idx": idx, "mm": open_memmap(d)}
    if not info:
        return None, skipped

    # 交集：只有所有参评模型都有的图才进对比（保证每个模型 n 相同）
    common = [(im, mk) for (im, mk) in gt
              if all(Path(im).name in v["idx"] for v in info.values())]
    n_models_total = len(gt)
    wh = {}
    for im, _mk in common:
        base = Path(im).name
        dims = {(v["idx"][base][1], v["idx"][base][2]) for v in info.values()}
        if len(dims) > 1 and verbose:
            print(f"  ! 各模型 w/h 不一致: {base} {dims}")
        wh[base] = next(iter(dims))

    acc = {m: {aspect_name(a): Acc(W.shape[0]) for a in aspects} for m in info}
    ious = {m: [] for m in info}
    oracle = {aspect_name(a): Acc(1) for a in aspects}
    W_oracle = np.array([W_NEW], dtype=np.float64)

    t0 = time.time()
    used = 0
    for n, (im, mkpath) in enumerate(common):
        base = Path(im).name
        try:
            mask = metrics.load_mask(mkpath)
        except Exception:
            continue
        if mask is None or not mask.any():
            continue
        h, w = int(mask.shape[0]), int(mask.shape[1])
        total = int(mask.sum())
        I = sw.mask_integral(mask)
        cent = metrics.subject_centroid(mask)
        if cent is None:
            continue
        cx_n, cy_n = float(cent[0]), float(cent[1])
        gt_small = np.asarray(
            Image.open(mkpath).convert("L").resize((GRID_W, GRID_H), Image.Resampling.BILINEAR),
            dtype=np.float32) / 255.0 >= 0.5

        for a in aspects:
            feat_b = sim.image_features(perfect_grid(mask), w, h, a, EVAL_KEEP, ANCHOR)
            oracle[aspect_name(a)].add(
                eval_image(feat_b, I, total, cx_n, cy_n, w, h, a, W_oracle))

        for m, v in info.items():
            idx, mw, mh = v["idx"][base]
            if (mw, mh) != (w, h):
                ww, hh = mw, mh
            else:
                ww, hh = w, h
            grid = v["mm"][idx]
            pred = np.asarray(grid, dtype=np.float32) > 0.5
            inter = int(np.logical_and(pred, gt_small).sum())
            union = int(np.logical_or(pred, gt_small).sum())
            ious[m].append(inter / union if union else 0.0)
            for a in aspects:
                feat = sim.image_features(grid, ww, hh, a, EVAL_KEEP, ANCHOR)
                acc[m][aspect_name(a)].add(
                    eval_image(feat, I, total, cx_n, cy_n, ww, hh, a, W))
        used += 1
        if verbose and (n + 1) % 500 == 0:
            print(f"  ... {n+1}/{len(common)}  {time.time()-t0:.0f}s", flush=True)

    out = {
        "dataset": ds, "label": DATASETS[ds]["label"],
        "manifest": str(DATASETS[ds]["manifest"]),
        "n_gt": len(gt), "n_common": len(common), "n_used": used,
        "weights": [list(w) for w in W], "weights_desc": weights_desc, "primary": PRIMARY,
        "aspects": [aspect_name(a) for a in aspects],
        "models": {}, "skipped": skipped, "seconds": round(time.time() - t0, 1),
    }
    for m in info:
        iou = np.asarray(ious[m], dtype=np.float64)
        out["models"][m] = {
            "label": MODEL_LABELS.get(m, m),
            "dir": info[m]["dir"],
            "n_grids": len(info[m]["idx"]),
            "median_iou": float(np.median(iou)) if iou.size else float("nan"),
            "mean_iou": float(iou.mean()) if iou.size else float("nan"),
            "by_aspect": {aspect_name(a): {
                "model": acc[m][aspect_name(a)].result(),
                "oracle": oracle[aspect_name(a)].result(),
            } for a in aspects},
        }
    return out, skipped


# ----------------------------------------------------------------- 子命令

def cmd_subset(args):
    gt = load_gt(args.dataset)
    rng = np.random.default_rng(args.seed)
    n = min(args.n, len(gt))
    sel = np.sort(rng.choice(len(gt), size=n, replace=False))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("image\tmask\n")
        for i in sel:
            f.write(f"{gt[i][0]}\t{gt[i][1]}\n")
    print(f"{DATASETS[args.dataset]['label']}：从 {len(gt)} 张抽 {n} 张（seed={args.seed}）→ {out}")
    return 0


def cmd_probe(args):
    """口径校验：原生分辨率中位 IoU（比较 u2netp 参考 0.7743）+ ms/图。"""
    import onnxruntime as ort
    sys.path.insert(0, str(SWAP))
    from model_grid import preprocess, pick_mask, read_manifest

    mean = None if not args.mean else np.array([float(x) for x in args.mean.split(",")], np.float32)
    std = None if not args.std else np.array([float(x) for x in args.std.split(",")], np.float32)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    opts.inter_op_num_threads = 1
    t0 = time.time()
    sess = ort.InferenceSession(args.model, opts, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    rows = [r for r in read_manifest(args.manifest) if len(r) >= 2]
    rng = np.random.default_rng(args.seed)
    if args.n and len(rows) > args.n:
        sel = sorted(rng.choice(len(rows), args.n, replace=False).tolist())
        rows = [rows[i] for i in sel]
    ious, ious_n, t_inf = [], [], time.time()
    fg_p, fg_g = [], []
    for parts in rows:
        with Image.open(parts[0]) as im:
            x = preprocess(im, side=args.side, norm=args.norm, mean=mean, std=std)
        raw = sess.run([out_name], {in_name: x})[0]
        m = pick_mask(raw, 0, 0, sigmoid=args.sigmoid)
        with Image.open(parts[1]).convert("L") as gim:
            gw, gh = gim.size
            g_small = gim if gim.size == (m.shape[1], m.shape[0]) else \
                gim.resize((m.shape[1], m.shape[0]), Image.Resampling.BILINEAR)
            gt = np.asarray(g_small, dtype=np.float32) / 255.0 >= 0.5
            gt_native = np.asarray(gim, dtype=np.float32) / 255.0 >= 0.5
        pred = m > 0.5
        u = int(np.logical_or(pred, gt).sum())
        ious.append(float(np.logical_and(pred, gt).sum()) / u if u else 0.0)
        # 原图分辨率：把概率图升采样回 GT 尺寸再二值化（基准论文口径）
        pim = Image.fromarray((np.clip(m, 0.0, 1.0) * 255).astype(np.uint8)).resize(
            (gw, gh), Image.Resampling.BILINEAR)
        pred_n = np.asarray(pim, dtype=np.float32) / 255.0 >= 0.5
        un = int(np.logical_or(pred_n, gt_native).sum())
        ious_n.append(float(np.logical_and(pred_n, gt_native).sum()) / un if un else 0.0)
        fg_p.append(float(pred.mean()))
        fg_g.append(float(gt.mean()))
    dt = time.time() - t_inf
    a = np.asarray(ious)
    an = np.asarray(ious_n)
    print(f"== 口径校验 {args.label or Path(args.model).name} ==")
    print(f"  side={args.side} norm={args.norm} sigmoid={args.sigmoid} n={len(a)} threads={args.threads}")
    print(f"  中位 IoU（模型分辨率）= {np.median(a):.4f}  均值={a.mean():.4f}  "
          f"p10={np.percentile(a,10):.4f}  <0.5 的 {int((a<0.5).sum())} 张")
    print(f"  中位 IoU（升采样回原图）= {np.median(an):.4f}  均值={an.mean():.4f}  "
          f"p10={np.percentile(an,10):.4f}  <0.5 的 {int((an<0.5).sum())} 张")
    print(f"  前景占比：预测均值={np.mean(fg_p):.4f}  GT={np.mean(fg_g):.4f}")
    print(f"  {dt/max(1,len(a))*1000:.0f} ms/图（含模型加载 {time.time()-t0:.1f}s）")
    print("  参考：u2netp 320²/imagenet 原生中位 IoU 0.7743（本轮用两种分辨率复核，见 RESULTS §6）")
    return 0


def cmd_run(args):
    weights = sw.parse_wlist(args.weights) if args.weights else np.array(DEFAULT_WEIGHTS)
    aspects = [aspect_of(a) for a in args.aspects]
    subset_keys = None
    if args.subset:
        _, rows = read_tsv(args.subset)
        subset_keys = set()
        for r in rows:
            for x in r[:2]:
                if x:
                    subset_keys.add(Path(x).name)
        print(f"限定子集：{args.subset}（{len(subset_keys)} 个文件名）")
    models = args.models
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        print(f"\n===== {DATASETS[ds]['label']} =====")
        t0 = time.time()
        res, skipped = run_dataset(ds, models, aspects, weights,
                                   args.weights or "base;new", subset_keys=subset_keys)
        for m, why in skipped.items():
            print(f"  跳过 {m}：{why}")
        if res is None:
            print("  没有可参评的模型，跳过")
            continue
        res["tag"] = args.tag
        out = Path(args.out_dir) / f"leaderboard-{args.tag}-{ds}.json" if args.out_dir else \
            LOG_DIR / f"leaderboard-{args.tag}-{ds}.json"
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  n={res['n_used']}/{res['n_common']}（真值 {res['n_gt']}）"
              f"  用时 {time.time()-t0:.0f}s → {out}")
        pr = res["primary"]
        for m, v in res["models"].items():
            row = []
            for a in res["aspects"]:
                d = v["by_aspect"][a]["model"]
                row.append(f"{a}: Δ{d['mean_delta'][pr]:+.5f} t{d['t_stat'][pr]:+.2f} "
                           f"A{d['mean_kept'][pr]:.4f}")
            print(f"  {m:<9} IoU={v['median_iou']:.4f}  " + " | ".join(row))
    return 0


def _fmt(v, spec=".4f", plus=False):
    if v != v:
        return "n/a"
    return format(v, ("+" if plus else "") + spec)


def cmd_tables(args):
    files = sorted(LOG_DIR.glob("leaderboard-*.json")) if not args.json else \
        [Path(x) for x in args.json]
    data = {}
    for f in files:
        j = json.loads(Path(f).read_text(encoding="utf-8"))
        data.setdefault(j["tag"], {})[j["dataset"]] = (j, Path(f))
    for tag in sorted(data):
        print(f"\n## tag = {tag}")
        for ds in ("duts", "ecssd", "omron"):
            if ds not in data[tag]:
                continue
            j, f = data[tag][ds]
            print(f"\n### {j['label']}（n={j['n_used']}，清单 {j['n_gt']}）← {f.name}")
            pr = j["primary"]
            print("\n| 模型 | 中位IoU | 口径 | A 保留率 | 居中 | Δ | t | 基线Δ | 三分线 | 切主体 | 采纳率 | 面积 | 整图框 | B | B−A |")
            print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for m, v in j["models"].items():
                for a in j["aspects"]:
                    blk = v["by_aspect"][a]
                    mm, ob = blk["model"], blk["oracle"]
                    print(f"| {m} | {_fmt(v['median_iou'])} | {a} | {_fmt(mm['mean_kept'][pr])} | "
                          f"{_fmt(mm['mean_kept_c'][pr])} | {_fmt(mm['mean_delta'][pr], '+.5f')} | "
                          f"{_fmt(mm['t_stat'][pr], '+.2f')} | {_fmt(mm['mean_delta'][0], '+.5f')} | "
                          f"{_fmt(mm['mean_thirds'][pr])} | {_fmt(mm['mean_cut'][pr]*100, '.1f')}% | "
                          f"{_fmt(mm['mean_adopted'][pr]*100, '.1f')}% | {_fmt(mm['mean_area'][pr], '.3f')} | "
                          f"{_fmt(mm['mean_skip'][pr]*100, '.1f')}% | {_fmt(ob['mean_kept'][0])} | "
                          f"{_fmt(ob['mean_kept'][0]-mm['mean_kept'][pr], '+.5f')} |")
            print("\n| 模型 | 基线 A | 基线 Δ | 基线 t | 基线三分线 | 基线切主体 | 基线面积 |")
            print("|---|---|---|---|---|---|---|")
            for m, v in j["models"].items():
                for a in j["aspects"]:
                    mm = v["by_aspect"][a]["model"]
                    print(f"| {m} | {a} | {_fmt(mm['mean_kept'][0])} | {_fmt(mm['mean_delta'][0], '+.5f')} | "
                          f"{_fmt(mm['t_stat'][0], '+.2f')} | {_fmt(mm['mean_thirds'][0])} | "
                          f"{_fmt(mm['mean_cut'][0]*100, '.1f')}% | {_fmt(mm['mean_area'][0], '.3f')} |")
    return 0


def cmd_dump(args):
    """全精度打印排行榜 JSON（含完美先验 B 与 B−A），供报告逐数复现。"""
    files = [Path(x) for x in args.json] if args.json else \
        sorted(LOG_DIR.glob(f"leaderboard-{args.tag}-*.json"))
    for f in files:
        j = json.loads(f.read_text(encoding="utf-8"))
        pr = j["primary"]
        print(f"\n===== {j['label']}  n={j['n_used']}/{j['n_common']}（真值 {j['n_gt']}）"
              f"  weights={j['weights_desc']}  ← {f.name}")
        for a in j["aspects"]:
            ob = next(iter(j["models"].values()))["by_aspect"][a]["oracle"]
            print(f"  [口径 {a}] 完美先验 B：kept={ob['mean_kept'][0]:.4f} thirds={ob['mean_thirds'][0]:.4f} "
                  f"adopt={ob['mean_adopted'][0]*100:.1f}% area={ob['mean_area'][0]:.3f} "
                  f"skip={ob['mean_skip'][0]*100:.1f}% cut={ob['mean_cut'][0]*100:.1f}%")
            for m, v in j["models"].items():
                mm = v["by_aspect"][a]["model"]
                print(f"    {m:<9} A={mm['mean_kept'][pr]:.4f} C={mm['mean_kept_c'][pr]:.4f} "
                      f"d={mm['mean_delta'][pr]:+.5f} t={mm['t_stat'][pr]:+.2f} "
                      f"B-A={ob['mean_kept'][0]-mm['mean_kept'][pr]:+.5f} "
                      f"thirds={mm['mean_thirds'][pr]:.4f} cut={mm['mean_cut'][pr]*100:.1f}% "
                      f"adopt={mm['mean_adopted'][pr]*100:.1f}% area={mm['mean_area'][pr]:.3f} "
                      f"skip={mm['mean_skip'][pr]*100:.1f}% win={mm['win_rate'][pr]*100:.1f}% "
                      f"| 去兜底 A={mm['mean_kept_noskip'][pr]:.4f} d={mm['mean_delta_noskip'][pr]:+.5f} "
                      f"t={mm['t_stat_noskip'][pr]:+.2f}（占比 {mm['mean_ns'][pr]*100:.1f}%）")
        print("  -- 基线权重对照 --")
        for m, v in j["models"].items():
            row = []
            for a in j["aspects"]:
                mm = v["by_aspect"][a]["model"]
                row.append(f"{a}: A={mm['mean_kept'][0]:.4f} d={mm['mean_delta'][0]:+.5f} "
                           f"t={mm['t_stat'][0]:+.2f}")
            print(f"    {m:<9} IoU={v['median_iou']:.4f}(mean {v['mean_iou']:.4f}) " + " | ".join(row))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="显著度模型排行榜（新权重 0.40/0.60/0.00）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("subset")
    p.add_argument("--dataset", required=True, choices=list(DATASETS))
    p.add_argument("--n", type=int, default=600)
    p.add_argument("--seed", type=int, default=20261005)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_subset)

    p = sub.add_parser("probe")
    p.add_argument("--model", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--side", type=int, default=320)
    p.add_argument("--norm", default="imagenet")
    p.add_argument("--mean", default="")
    p.add_argument("--std", default="")
    p.add_argument("--sigmoid", action="store_true")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=20261005)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--label", default="")
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("run")
    p.add_argument("--tag", default="full")
    p.add_argument("--datasets", nargs="*", default=["duts", "ecssd", "omron"], choices=list(DATASETS))
    p.add_argument("--models", nargs="*", default=["u2netp", "u2net", "rmbg", "isnet", "birefnet"],
                   choices=list(MODELS_DIRS))
    p.add_argument("--aspects", nargs="*", default=["src", "1", "0.75"])
    p.add_argument("--weights", default="", help="'t,r,a[,onset,slope];...'；默认 基线;新权重")
    p.add_argument("--subset", default="", help="只评该清单里的图（image<TAB>mask 或纯图片清单）")
    p.add_argument("--out-dir", default="")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("tables")
    p.add_argument("--json", nargs="*", default=[])
    p.set_defaults(func=cmd_tables)

    p = sub.add_parser("dump")
    p.add_argument("--json", nargs="*", default=[])
    p.add_argument("--tag", default="full")
    p.set_defaults(func=cmd_dump)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
