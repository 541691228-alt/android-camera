"""核对 RESULTS-weight-sweep.md 里的数字与 sweep JSON 是否一致（只读）。"""
import json
from pathlib import Path

D = Path(r"D:\spider\.cache\eval-duts\sweep-study")
WANT = [(0.55, 0.25, 0.20), (0.40, 0.60, 0.00)]

for f in sorted(D.glob("sweep_*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    label = f.stem.replace("sweep_", "").replace("_", "/")
    print(f"\n== {label}  n={d['n']}")
    for i in range(len(d["w_thirds"])):
        key = (round(d["w_thirds"][i], 2), round(d["w_ret"][i], 2), round(d["w_area"][i], 2))
        if key not in WANT:
            continue
        print(f"  {key[0]:.2f}/{key[1]:.2f}/{key[2]:.2f}  kept={d['mean_kept'][i]:.4f} "
              f"center={d['mean_kept_center'][i]:.4f} delta={d['mean_delta'][i]:+.5f} "
              f"t={d['t_stat'][i]:+.2f} win={d['win_rate'][i]*100:.1f}% "
              f"thirds={d['mean_thirds'][i]:.4f} cut={d['cut_rate'][i]*100:.1f}% "
              f"skip={d['skip_rate'][i]*100:.1f}% area={d['mean_keep_area'][i]:.3f} "
              f"dead={d['dead_rate'][i]*100:.1f}% adopt={d['adopt_rate'][i]*100:.1f}%")
    best = max(range(len(d["w_thirds"])), key=lambda i: d["mean_delta"][i])
    print(f"  best_delta={d['mean_delta'][best]:+.5f} @ "
          f"{d['w_thirds'][best]:.2f}/{d['w_ret'][best]:.2f}/{d['w_area'][best]:.2f} "
          f"(t={d['t_stat'][best]:+.2f})")
