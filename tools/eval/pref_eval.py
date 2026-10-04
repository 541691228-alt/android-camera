#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把美学模型对"算法裁的图"和"居中裁的图"的打分，算成一份偏好对照报告。

DUTS 的掩膜只能证明"主体没被切掉、主体更靠三分点"，证明不了"这张照片更好看"。
要补上这一环，就得给同一张原图的两种裁法各打一个美学分，然后看算法赢不赢。
打分由外部模型做（见 aesthetic_score.py），这个脚本只负责统计。

输入：
    --pairs   export_crops.py 产出的 pairs.tsv（index/pred/center/坐标/identical）
    --scores  打分表，两种格式都认：
                index<TAB>pred_score<TAB>center_score
                path<TAB>score        （用 pred/center 路径去匹）

    python tools/eval/pref_eval.py --scores work/crops_trim/scores.tsv \
        --label model-src --out-md work/report_pref_model_src.md

报告给出：胜/平/败、平均分差、符号检验 p 值，并按"两种裁法差多少"分档 —— 差别本来就很小的
那些图，赢不赢没什么信息量。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

TIE_EPS = 1e-6        # 分差小于这个值算平
STRENGTH_BINS = ((0.85, "几乎一样"), (0.60, "略有不同"), (0.0, "差得较多"))


def read_tsv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def as_float(value) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def load_scores(path: Path) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    """返回 (index → (pred, center), 路径 → 分数)。"""
    rows = read_tsv(path)
    if not rows:
        return {}, {}
    cols = {c.lower() for c in rows[0].keys()}
    by_index: dict[str, tuple[float, float]] = {}
    by_path: dict[str, float] = {}
    if {"pred_score", "center_score"} <= cols:
        for row in rows:
            idx = (row.get("index") or "").strip()
            p = as_float(row.get("pred_score"))
            c = as_float(row.get("center_score"))
            if idx and p is not None and c is not None:
                by_index[idx] = (p, c)
        return by_index, by_path
    # 单列格式：第一列当路径，找一个叫 score 的列（没有就取最后一列）
    key_col = list(rows[0].keys())[0]
    score_col = next((c for c in rows[0].keys() if c.lower() in ("score", "aesthetic", "quality", "nima")),
                     list(rows[0].keys())[-1])
    for row in rows:
        p = (row.get(key_col) or "").strip()
        s = as_float(row.get(score_col))
        if p and s is not None:
            by_path[p] = s
    return by_index, by_path


def box_iou(a: list[float], b: list[float]) -> float:
    """两个框 [left, top, w, h] 的 IoU。"""
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def sign_test_p(wins: int, loses: int) -> float:
    """两向符号检验的精确双尾 p 值（丢掉平局）。"""
    n = wins + loses
    if n == 0:
        return 1.0
    k = min(wins, loses)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def strength_bin(similarity: float) -> str:
    for limit, name in STRENGTH_BINS:
        if similarity >= limit:
            return name
    return STRENGTH_BINS[-1][1]


def summarize(pairs: list[dict]) -> dict:
    deltas = [p["delta"] for p in pairs]
    n = len(pairs)
    wins = sum(1 for d in deltas if d > TIE_EPS)
    loses = sum(1 for d in deltas if d < -TIE_EPS)
    ties = n - wins - loses
    pred = [p["pred"] for p in pairs]
    center = [p["center"] for p in pairs]
    mean_delta = sum(deltas) / n if n else 0.0
    scale = (sum(abs(v) for v in pred) + sum(abs(v) for v in center)) / (2 * n) if n else 0.0
    out = {
        "n": n,
        "win": wins, "tie": ties, "lose": loses,
        "win_rate": wins / n if n else 0.0,
        "lose_rate": loses / n if n else 0.0,
        "mean_pred": sum(pred) / n if n else 0.0,
        "mean_center": sum(center) / n if n else 0.0,
        "mean_delta": mean_delta,
        "median_delta": sorted(deltas)[n // 2] if n else 0.0,
        "rel_delta": mean_delta / scale if scale else 0.0,
        "sign_test_p": sign_test_p(wins, loses),
    }
    buckets: dict[str, dict] = {}
    for p in pairs:
        name = strength_bin(p["iou"])
        b = buckets.setdefault(name, {"n": 0, "win": 0, "tie": 0, "lose": 0, "sum_delta": 0.0})
        b["n"] += 1
        b["sum_delta"] += p["delta"]
        if p["delta"] > TIE_EPS:
            b["win"] += 1
        elif p["delta"] < -TIE_EPS:
            b["lose"] += 1
        else:
            b["tie"] += 1
    for b in buckets.values():
        b["win_rate"] = b["win"] / b["n"] if b["n"] else 0.0
        b["mean_delta"] = b["sum_delta"] / b["n"] if b["n"] else 0.0
    out["by_strength"] = buckets
    return out


def write_markdown(report: dict, label: str, dst: Path, scores_path: str) -> None:
    s = report["summary"]
    if s["sign_test_p"] < 0.05:
        verdict = (f"算法在 {s['n']} 对里赢 {s['win']} 对（{s['win_rate']:.1%}），"
                   f"平均分差 {s['mean_delta']:+.4f}，符号检验 p={s['sign_test_p']:.4f} —— 差异显著。")
    else:
        verdict = (f"算法在 {s['n']} 对里赢 {s['win']} 对（{s['win_rate']:.1%}），"
                   f"平均分差 {s['mean_delta']:+.4f}，符号检验 p={s['sign_test_p']:.4f} —— "
                   f"差异不显著，别当结论用。")
    lines = [
        f"# 美学偏好对照：{label}",
        "",
        f"打分表：`{scores_path}`　样本：{s['n']} 对（算法 vs 同尺寸居中裁剪，同一张原图）。",
        "只有算法真的动过剪刀的图才进来 —— 不裁的图上两张是同一张，比了也没信息量。",
        "",
        "## 总体",
        "",
        "| 指标 | 值 |",
        "| --- | --- |",
        f"| 算法胜 / 平 / 负 | {s['win']} / {s['tie']} / {s['lose']}（胜率 {s['win_rate']:.1%}） |",
        f"| 平均分：算法 / 居中 | {s['mean_pred']:.4f} / {s['mean_center']:.4f} |",
        f"| 平均分差（算法−居中） | {s['mean_delta']:+.4f}（相对 {s['rel_delta']:+.2%}） |",
        f"| 分差中位数 | {s['median_delta']:+.4f} |",
        f"| 符号检验 p（双尾） | {s['sign_test_p']:.4f} |",
        "",
        "## 按两种裁法差多少分档",
        "",
        "| 框重合度 | 对数 | 胜/平/负 | 胜率 | 平均分差 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, name in STRENGTH_BINS:
        b = s["by_strength"].get(name)
        if not b:
            continue
        lines.append(f"| {name} | {b['n']} | {b['win']}/{b['tie']}/{b['lose']} | "
                     f"{b['win_rate']:.1%} | {b['mean_delta']:+.4f} |")
    lines += [
        "",
        "## 结论",
        "",
        verdict,
        "",
        "> 注意：这是「同一张图两种裁法」的偏好，分数来自某个美学模型，换模型结论可能变。",
        "> 样本是 DUTS-TE 的 200 张里算法真裁过的那部分，不是通用相册。",
        "",
    ]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="算美学模型对两种裁法的偏好")
    ap.add_argument("--pairs", default="work/crops_trim/pairs.tsv")
    ap.add_argument("--scores", required=True)
    ap.add_argument("--label", default="model-src")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-md", default="")
    args = ap.parse_args(argv)

    pairs_path = Path(args.pairs)
    scores_path = Path(args.scores)
    rows = read_tsv(pairs_path)
    if not rows:
        print(f"没有读到 pairs：{pairs_path}", file=sys.stderr)
        return 2
    by_index, by_path = load_scores(scores_path)
    if not by_index and not by_path:
        print(f"没有读到任何分数：{scores_path}", file=sys.stderr)
        return 2

    used: list[dict] = []
    n_missing = 0
    for row in rows:
        idx = (row.get("index") or "").strip()
        pred_p = (row.get("pred") or "").strip()
        center_p = (row.get("center") or "").strip()
        score = by_index.get(idx)
        if score is None and by_path:
            p_sc = by_path.get(pred_p)
            c_sc = by_path.get(center_p)
            if p_sc is not None and c_sc is not None:
                score = (p_sc, c_sc)
        if score is None:
            n_missing += 1
            continue
        try:
            box_p = [float(row["pred_left"]), float(row["pred_top"]), float(row["pred_w"]), float(row["pred_h"])]
            box_c = [float(row["center_left"]), float(row["center_top"]), float(row["center_w"]), float(row["center_h"])]
        except (KeyError, ValueError):
            n_missing += 1
            continue
        if (row.get("identical") or "0").strip() in ("1", "true"):
            continue
        used.append({
            "index": idx,
            "pred": score[0], "center": score[1],
            "delta": score[0] - score[1],
            "iou": box_iou(box_p, box_c),
        })

    if not used:
        print("配对后一对都没剩下（检查 scores 里的 index/路径能不能对上）", file=sys.stderr)
        return 1

    summary = summarize(used)
    report = {
        "label": args.label,
        "pairs_file": str(pairs_path),
        "scores_file": str(scores_path),
        "n_missing_scores": n_missing,
        "summary": summary,
        "rows": [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()} for r in used],
    }
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    if args.out_md:
        write_markdown(report, args.label, Path(args.out_md), str(scores_path))

    s = summary
    print(f"{args.label}: n={s['n']} 胜/平/负={s['win']}/{s['tie']}/{s['lose']} "
          f"平均分差={s['mean_delta']:+.4f} p={s['sign_test_p']:.4f}"
          + (f"（{n_missing} 对没分）" if n_missing else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
