# -*- coding: utf-8 -*-
"""把多份 run_eval.py 的报告并排成一张表 —— 换数据集 / 换口径交叉验证用。

每行一份报告，列是同一套指标，方便一眼看出"结论在别的数据集上还成不成立"。
用法：
    python compare_reports.py --reports a.json b.json c.json --out-md compare.md
    python compare_reports.py --reports *.json --labels DUTS/模型 ECSSD/模型 DUT-OMRON/模型
不给 --labels 就用报告里的 config.label（可能不唯一，会加序号）。
"""
import argparse
import glob
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 列定义：(表头, 取值函数)。取值函数拿 report dict，返回字符串。
COLUMNS = [
    ("图数", lambda r: f"{r.get('n_images', 0)}"),
    ("预测保留率", lambda r: f"{_g(r, 'pred', 'mean_kept_ratio'):.4f}"),
    ("居中保留率", lambda r: f"{_g(r, 'center', 'mean_kept_ratio'):.4f}"),
    ("同尺寸上界", lambda r: f"{_g(r, 'optimal', 'mean_kept_ratio'):.4f}"),
    ("胜/平/负", lambda r: "{:.1f}/{:.1f}/{:.1f}".format(
        100 * _n(r, "win_rate"), 100 * _n(r, "tie_rate"), 100 * _n(r, "lose_rate"))),
    ("提升捕获", lambda r: f"{100 * _n(r, 'capture_ratio'):+.1f}%"),
    ("不裁比例", lambda r: f"{100 * _n(r, 'skip_rate'):.1f}%"),
    ("平均裁剪面积", lambda r: f"{_n(r, 'mean_keep_area'):.3f}"),
    ("三分距 预测", lambda r: f"{_g(r, 'composition', 'pred', 'mean_thirds_distance'):.4f}"),
    ("三分距 居中", lambda r: f"{_g(r, 'composition', 'center', 'mean_thirds_distance'):.4f}"),
    ("摆正中 预测", lambda r: f"{100 * _g(r, 'composition', 'pred', 'dead_center_rate'):.1f}%"),
    ("摆正中 居中", lambda r: f"{100 * _g(r, 'composition', 'center', 'dead_center_rate'):.1f}%"),
    ("切主体 预测", lambda r: f"{100 * _g(r, 'composition', 'pred', 'cut_rate'):.1f}%"),
    ("切主体 居中", lambda r: f"{100 * _g(r, 'composition', 'center', 'cut_rate'):.1f}%"),
    ("主体损失>1% 预测", lambda r: f"{100 * _g(r, 'composition', 'pred_loss', 'lost_gt_1pct_rate'):.1f}%"),
    ("主体损失>10% 预测", lambda r: f"{100 * _g(r, 'composition', 'pred_loss', 'lost_gt_10pct_rate'):.1f}%"),
    ("主体损失>1% 居中", lambda r: f"{100 * _g(r, 'composition', 'center_loss', 'lost_gt_1pct_rate'):.1f}%"),
    ("错误行", lambda r: f"{r.get('n_errors', 0)}"),
]


def _dig(d, keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _n(report, *keys):
    v = _dig(report, keys)
    return float(v) if isinstance(v, (int, float)) else float("nan")


def _g(report, *keys):
    return _n(report, *keys)


def _fmt(v, suffix=""):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{v}{suffix}"


def load(paths, labels):
    reports, names = [], []
    for i, p in enumerate(paths):
        with open(p, encoding="utf-8-sig") as fh:
            r = json.load(fh)
        reports.append(r)
        if labels and i < len(labels):
            names.append(labels[i])
        else:
            cfg = r.get("config", {})
            lab = cfg.get("label") or os.path.basename(p)
            asp = cfg.get("aspect")
            names.append(f"{lab}" + (f" [{asp}]" if asp else ""))
    return reports, names


def main():
    ap = argparse.ArgumentParser(description="把多份评测报告并排成一张表")
    ap.add_argument("--reports", nargs="+", required=True, help="report_*.json 路径（支持通配符）")
    ap.add_argument("--labels", nargs="*", default=None, help="每份报告的行名，顺序与 --reports 一致")
    ap.add_argument("--out-md", default="", help="写 markdown 表格")
    ap.add_argument("--out-json", default="", help="写合并后的 JSON（只留关键字段）")
    ap.add_argument("--title", default="", help="markdown 大标题")
    args = ap.parse_args()

    paths = []
    for pat in args.reports:
        hit = sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat]
        if not hit:
            print(f"警告：没有匹配到 {pat}", file=sys.stderr)
        paths.extend(hit)
    if not paths:
        print("没有可读的报告", file=sys.stderr)
        return 1

    reports, names = load(paths, args.labels)
    head = ["口径"] + [c[0] for c in COLUMNS]
    rows = []
    for name, r in zip(names, reports):
        rows.append([name] + [c[1](r) for c in COLUMNS])

    w = [max(len(head[i]), max(len(row[i]) for row in rows)) for i in range(len(head))]
    # 终端表格：列多，用竖线隔开仍然好读
    print(" | ".join(head[i].ljust(w[i]) for i in range(len(head))))
    print("-+-".join("-" * w[i] for i in range(len(head))))
    for row in rows:
        print(" | ".join(row[i].ljust(w[i]) for i in range(len(head))))

    md = []
    if args.title:
        md += [f"## {args.title}", ""]
    md += ["| " + " | ".join(head) + " |", "| " + " | ".join(["---"] * len(head)) + " |"]
    for row in rows:
        md.append("| " + " | ".join(row) + " |")
    text = "\n".join(md) + "\n"

    if args.out_md:
        with open(args.out_md, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        print(f"\n已写 {args.out_md}")
    if args.out_json:
        slim = []
        for name, r in zip(names, reports):
            slim.append({
                "name": name,
                "config": r.get("config", {}),
                "n_images": r.get("n_images"),
                "metrics": {c[0]: c[1](r) for c in COLUMNS},
            })
        with open(args.out_json, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(slim, fh, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"已写 {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
