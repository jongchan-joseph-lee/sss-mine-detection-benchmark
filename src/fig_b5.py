"""Figure for the augmentation experiment: test-year FROC and Pd by object size at 4 FA/frame."""
import json
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.argv = ["x"]
exec(open("b5_eval.py").read().replace('if __name__ == "__main__":\n    main()', ""))
R = json.load(open("b5_eval.json"))
FA = np.logspace(np.log10(0.1), np.log10(10), 60)
RUNS = [("ref_full", "YOLOv8n 25 ep (reference)", "#9a9a9a", "-"),
        ("n_base40last_full", "40 ep, original tiles", "#4a4a4a", "--"),
        ("n_aug40last_full", "40 ep, + augmented tiles", "#d08a1e", "-"),
        ("n_aug40last_sliced", "40 ep, + augmented, sliced inference", "#8a4f00", ":")]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
for key, lab, col, ls in RUNS:
    rows, _ = per_image(json.load(open(f"b5_dets/{key}.json")))
    c, _ = froc([r for mm, _ in rows for r in mm], sum(g for _, g in rows), len(rows), tuple(FA))
    ax[0].plot(FA, [p for _, p in c], ls, color=col, lw=2, label=lab)
ax[0].set_xscale("log"); ax[0].set_ylim(0, 1); ax[0].set_xlim(0.1, 10)
ax[0].set_xlabel("False alarms per frame"); ax[0].set_ylabel("Recall (Pd)")
ax[0].set_title("Test years 2010 + 2018: 578 frames, 114 objects", fontsize=11)
ax[0].axvline(1, color="#bbbbbb", ls=":", lw=1); ax[0].legend(frameon=False, fontsize=9, loc="lower right")
bands = ["0-30", "30-50", "50-inf"]
w = 0.2
for i, (key, lab, col, _) in enumerate(RUNS):
    v = [R[key]["pd_by_size_at_4fa"][b]["pd"] for b in bands]
    ax[1].bar(np.arange(3) + (i - 1.5) * w, v, w, color=col, label=lab)
n = [R["ref_full"]["pd_by_size_at_4fa"][b]["n"] for b in bands]
ax[1].set_xticks(range(3)); ax[1].set_xticklabels([f"<30 px\n(n={n[0]})", f"30–50 px\n(n={n[1]})", f"≥50 px\n(n={n[2]})"])
ax[1].set_ylim(0, 1); ax[1].set_ylabel("Pd @ 4 FA/frame"); ax[1].set_title("Recall by object size at 4 FA/frame (1024-px scale)", fontsize=11)
ax[1].grid(alpha=.3, axis="y")
for a, t in zip(ax, "ab"):
    a.text(-0.1, 1.04, f"({t})", transform=a.transAxes, fontweight="bold", fontsize=13)
fig.tight_layout()
fig.savefig("../docs/figures/augmentation_froc.png", dpi=150)
