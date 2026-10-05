"""把 sweep_weights.py 产出的 JSON 汇总成 markdown 表（供 RESULTS 文档直接粘）。

用法：
  python sweep_report.py --dir D:\\spider\\.cache\\eval-duts\\sweep-study --out report.md
  python sweep_report.py --files a.json b.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def regime_label(path: Path) -> str:
    """sweep_<ds>_<aspect>.json → <ds>/<aspect>"""
    stem = path.stem
    if stem.startswith("sweep_"):
        stem = stem[len("sweep_"):]
    return stem.replace("_", "/", 1)


def load_rows(path: Path):
    d = json.loads(path.read_text(encoding="utf-8"))
    label = regime_label(path)
    rows = []
    for i in range(len(d["w_thirds"])):
        rows.append({
            "regime": label,
            "file": path.name,
            "w": (d["w_thirds"][i], d["w_ret"][i], d["w_area"][i]),
            "onset": d["cut_onset"][i],
            "slope": d["cut_slope"][i],
            "n": d["n"],
            "kept": d["mean_kept"][i],
            "center": d["mean_kept_center"][i],
            "delta": d["mean_delta"][i],
            "t": d["t_stat"][i],
            "win": d["win_rate"][i],
            "thirds": d["mean_thirds"][i],
            "thirds_c": d["mean_thirds_center"][i],
            "cut": d["cut_rate"][i],
            "skip": d["skip_rate"][i],
            "area": d["mean_keep_area"][i],
            "dead": d["dead_rate"][i],
        })
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="")
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--out", default="")
    ap.add_argument("--baseline", default="0.55,0.25,0.20", help="基线权重，用于对比列")
    args = ap.parse_args(argv)

    files = [Path(p) for p in args.files]
    if args.dir:
        files += sorted(Path(args.dir).glob("sweep_*.json"))
    if not files:
        raise SystemExit("没有输入 JSON")
    base = tuple(float(x) for x in args.baseline.split(","))

    allrows = []
    for f in files:
        allrows += load_rows(f)

    lines = []
    lines.append("| 口径 | 权重 (thirds/ret/area) | 算法保留率 | 居中保留率 | 差值 | t | 赢率 | 三分线 | 三分线(居中) | 切主体 | 整图框 | 裁剪面积 | 死中 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in allrows:
        w = r["w"]
        tag = " **基线**" if (round(w[0], 3), round(w[1], 3), round(w[2], 3)) == \
            (round(base[0], 3), round(base[1], 3), round(base[2], 3)) else ""
        lines.append(
            f"| {r['regime']} | {w[0]:.2f}/{w[1]:.2f}/{w[2]:.2f}{tag} | {r['kept']:.4f} | "
            f"{r['center']:.4f} | {r['delta']:+.5f} | {r['t']:+.2f} | {r['win']*100:.1f}% | "
            f"{r['thirds']:.4f} | {r['thirds_c']:.4f} | {r['cut']*100:.1f}% | "
            f"{r['skip']*100:.1f}% | {r['area']:.3f} | {r['dead']*100:.1f}% |")

    # 每个口径：基线差值 vs 最好差值
    lines.append("")
    lines.append("| 口径 | 图像数 | 基线差值 | 基线 t | 最好差值 | 对应权重 | 最好 t |")
    lines.append("|---|---|---|---|---|---|---|")
    regimes = sorted({r["regime"] for r in allrows})
    for reg in regimes:
        rs = [r for r in allrows if r["regime"] == reg]
        b = [r for r in rs if (round(r["w"][0], 3), round(r["w"][1], 3), round(r["w"][2], 3)) ==
             (round(base[0], 3), round(base[1], 3), round(base[2], 3))]
        b = b[0] if b else rs[0]
        best = max(rs, key=lambda r: r["delta"])
        lines.append(f"| {reg} | {b['n']} | {b['delta']:+.5f} | {b['t']:+.2f} | {best['delta']:+.5f} | "
                     f"{best['w'][0]:.2f}/{best['w'][1]:.2f}/{best['w'][2]:.2f} | {best['t']:+.2f} |")

    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"\n已写 {args.out}")


if __name__ == "__main__":
    main()
