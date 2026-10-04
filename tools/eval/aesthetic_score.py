#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给裁剪图打美学分（无参考），只用 onnxruntime，不需要 torch。

两个模型都是 AVA 数据集训出来的"这张照片好不好看"的回归模型，输出 10 档评分的分布，
验收时取期望（Σ p_i · i，i = 1..10，就是 NIMA 论文里的 mean score）：

* **TOPIQ-IAA**（ResNet50，IQA-PyTorch）：主指标，AVA 上 PLCC 0.737。
* **NIMA MobileNet**（titu1994/neural-image-assessment）：小模型第二意见，12 MB。

预处理严格按各自的官方推理代码来，别自己猜：

* TOPIQ：RGB、拉伸到 384×384、`/255`；网络内部自己做 ImageNet 均值方差归一化。
* NIMA：RGB、拉伸到 224×224、`/127.5 - 1`（keras `preprocess_input`，输入是 0~255）。

配对公平性：同一对里算法框和居中框的**尺寸完全一样**，所以拉伸带来的形变偏差在配对内抵消，
不需要保守长宽比去补边。

模型文件不入库（280 MB + 13 MB），下载：

    curl -L -o topiq_iaa_res50.onnx        https://huggingface.co/cromsc/topiq-iaa-res50/resolve/main/topiq_iaa_res50.onnx
    curl -L -o nima_mobilenet_aesthetic.onnx https://huggingface.co/cromsc/nima-mobilenet-aesthetic/resolve/main/nima_mobilenet_aesthetic.onnx

用法：

    python tools/eval/aesthetic_score.py --pairs work/crops_trim/pairs.tsv \
        --model-dir D:/spider/.cache/aesthetic --full

产出 `<out-dir>/scores_topiq.tsv`、`scores_nima.tsv`（`index/pred_score/center_score`，直接喂 `pref_eval.py`）
与合并的 `scores_all.tsv`。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BINS = np.arange(1, 11, dtype=np.float32)     # AVA 1~10 档

MODELS = {
    # 名字: (文件名, 输入尺寸(H, W), 布局, 预处理)
    "topiq": ("topiq_iaa_res50.onnx", (384, 384), "NCHW", "div255"),
    "nima": ("nima_mobilenet_aesthetic.onnx", (224, 224), "NHWC", "mobilenet"),
}
HF_URLS = {
    "topiq": "https://huggingface.co/cromsc/topiq-iaa-res50/resolve/main/topiq_iaa_res50.onnx",
    "nima": "https://huggingface.co/cromsc/nima-mobilenet-aesthetic/resolve/main/nima_mobilenet_aesthetic.onnx",
}


def read_tsv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def preprocess(img: Image.Image, size: tuple[int, int], layout: str, mode: str) -> np.ndarray:
    im = img.resize(size, Image.BILINEAR)     # size 是 (W, H)
    a = np.asarray(im, dtype=np.float32)
    if mode == "div255":
        a = a / 255.0
    elif mode == "mobilenet":
        a = a / 127.5 - 1.0
    else:
        raise ValueError(f"未知预处理 {mode}")
    if layout == "NCHW":
        a = np.ascontiguousarray(a.transpose(2, 0, 1)[None])
    else:
        a = np.ascontiguousarray(a[None])
    return a


class Scorer:
    """按需加载模型；score() 返回 {模型名: (期望分, 分布标准差)}。"""

    def __init__(self, names: list[str], model_dir: Path):
        self.sess: dict[str, tuple[ort.InferenceSession, str, tuple[int, int], str, str]] = {}
        for name in names:
            fname, size, layout, mode = MODELS[name]
            path = model_dir / fname
            if not path.exists():
                raise SystemExit(
                    f"缺模型文件 {path}\n下载：curl -L -o {path} {HF_URLS[name]}"
                )
            so = ort.SessionOptions()
            so.log_severity_level = 3
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
            self.sess[name] = (sess, sess.get_inputs()[0].name, size, layout, mode)

    def score(self, img: Image.Image) -> dict[str, tuple[float, float]]:
        out: dict[str, tuple[float, float]] = {}
        for name, (sess, iname, size, layout, mode) in self.sess.items():
            x = preprocess(img, size, layout, mode)
            dist = np.asarray(sess.run(None, {iname: x})[0][0], dtype=np.float64)
            dist = dist / max(dist.sum(), 1e-12)
            mean = float((dist * BINS).sum())
            var = float((dist * (BINS - mean) ** 2).sum())
            out[name] = (mean, var ** 0.5)
        return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="给裁剪图打美学分")
    ap.add_argument("--pairs", default="work/crops_trim/pairs.tsv")
    ap.add_argument("--model-dir", default=os.environ.get("AESTHETIC_MODEL_DIR", "aesthetic_models"))
    ap.add_argument("--models", default="topiq,nima")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--full", action="store_true", help="顺带给原图打分（看裁一刀本身是加分还是减分）")
    ap.add_argument("--timing", type=int, default=0, help="拿第 1 张重复 N 次测速")
    args = ap.parse_args(argv)

    names = [n.strip() for n in args.models.split(",") if n.strip()]
    for n in names:
        if n not in MODELS:
            print(f"不认识的模型 {n}，可选 {list(MODELS)}", file=sys.stderr)
            return 2

    pairs_path = Path(args.pairs)
    rows = read_tsv(pairs_path)
    if not rows:
        print(f"没读到 pairs：{pairs_path}", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir) if args.out_dir else pairs_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    sc = Scorer(names, Path(args.model_dir))
    print(f"模型 {', '.join(names)}　样本 {len(rows)} 对"
          + ("（含原图）" if args.full else ""), flush=True)

    if args.timing and rows:
        img = Image.open(rows[0]["pred"]).convert("RGB")
        t0 = time.perf_counter()
        for _ in range(args.timing):
            sc.score(img)
        dt = (time.perf_counter() - t0) / args.timing
        print(f"单图 {dt * 1000:.0f} ms（{len(names)} 个模型）", flush=True)

    per_model: dict[str, list[list[str]]] = {n: [] for n in names}
    all_rows: list[dict] = []
    n_done = 0
    t_start = time.perf_counter()
    for row in rows:
        if args.limit and n_done >= args.limit:
            break
        idx = (row.get("index") or "").strip()
        pred_p = (row.get("pred") or "").strip()
        center_p = (row.get("center") or "").strip()
        full_p = (row.get("image") or "").strip()
        try:
            s_pred = sc.score(Image.open(pred_p).convert("RGB"))
            s_center = sc.score(Image.open(center_p).convert("RGB"))
            s_full = sc.score(Image.open(full_p).convert("RGB")) if args.full else {}
        except Exception as exc:
            print(f"打分失败 {pred_p}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        rec: dict[str, str] = {"index": idx, "image": full_p, "pred": pred_p, "center": center_p}
        for n in names:
            pv, ps = s_pred[n]
            cv, cs = s_center[n]
            per_model[n].append([idx, f"{pv:.6f}", f"{cv:.6f}"])
            rec[f"pred_{n}"] = f"{pv:.6f}"
            rec[f"center_{n}"] = f"{cv:.6f}"
            rec[f"pred_{n}_std"] = f"{ps:.6f}"
            rec[f"center_{n}_std"] = f"{cs:.6f}"
            if n in s_full:
                fv, fs = s_full[n]
                rec[f"full_{n}"] = f"{fv:.6f}"
                rec[f"full_{n}_std"] = f"{fs:.6f}"
        all_rows.append(rec)
        n_done += 1
        if n_done % 25 == 0:
            print(f"  {n_done}/{len(rows)} …", flush=True)

    for n in names:
        dst = out_dir / f"scores_{n}.tsv"
        with dst.open("w", encoding="utf-8", newline="") as fh:
            wr = csv.writer(fh, delimiter="\t", lineterminator="\n")
            wr.writerow(["index", "pred_score", "center_score"])
            wr.writerows(per_model[n])
        print(f"写出 {dst}（{len(per_model[n])} 行）")

    if all_rows:
        cols = list(all_rows[0].keys())
        cols = ["index", "image", "pred", "center"] + [c for c in cols if c not in ("index", "image", "pred", "center")]
        dst_all = out_dir / "scores_all.tsv"
        with dst_all.open("w", encoding="utf-8", newline="") as fh:
            wr = csv.writer(fh, delimiter="\t", lineterminator="\n")
            wr.writerow(cols)
            wr.writerows([[r.get(c, "") for c in cols] for r in all_rows])
        print(f"写出 {dst_all}")

        for n in names:
            p = np.array([float(r[f"pred_{n}"]) for r in all_rows])
            c = np.array([float(r[f"center_{n}"]) for r in all_rows])
            line = (f"{n:<6} 算法 {p.mean():.3f}（σ {p.std():.3f}）　居中 {c.mean():.3f}（σ {c.std():.3f}）"
                    f"　平均分差 {np.mean(p - c):+.4f}")
            if args.full:
                fl = np.array([float(r[f"full_{n}"]) for r in all_rows])
                line += f"　原图 {fl.mean():.3f}（差分 {np.mean(p - fl):+.4f}）"
            print(line)
    print(f"用时 {time.perf_counter() - t_start:.1f} s，共 {n_done} 对")
    return 0 if n_done else 1


if __name__ == "__main__":
    raise SystemExit(main())
