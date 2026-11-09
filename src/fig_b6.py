"""GPU comparison figure: architecture x training data on test years (sliced inference), with training time.

  python fig_b6.py   ->  docs/figures/architecture_comparison.png   (needs gpu_runs/ and b5_dets/*60_*.json:
                          tar xzf gpu_runs/dets_gpu.tar.gz)
"""
import json

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "font.size": 12,
                     "axes.spines.top": False, "axes.spines.right": False})
R = json.load(open("gpu_runs/b5_eval_gpu.json"))
E = json.load(open("gpu_runs/environment.json"))["runs"]
GREY, AMBER = "#9a9a9a", "#d08a1e"
ARCH = [("n", "YOLOv8n\n3.1 M"), ("s", "YOLOv8s\n11.1 M"), ("rtdetr", "RT-DETR-l\n32.8 M")]
g = R["ref_sliced"]["fa_grid"]

fig, ax = plt.subplots(1, 3, figsize=(14, 4.3), gridspec_kw={"width_ratios": [1, 1, 0.8]})
x = np.arange(3); w = 0.36
for j, (fa, title) in enumerate([(4, "Pd @ 4 false alarms / frame"), (8, "Pd @ 8 false alarms / frame")]):
    i = g.index(fa)
    for k, (dk, col, lab) in enumerate([("base", GREY, "original tiles"), ("aug", AMBER, "+ size-matched augmentation")]):
        v = [R[f"{a}_{dk}60_sliced"]["pd"][i] for a, _ in ARCH]
        ci = np.array([R[f"{a}_{dk}60_sliced"]["ci95"][i] for a, _ in ARCH])
        ax[j].bar(x + (k - 0.5) * w, v, w, color=col, label=lab, yerr=[np.array(v) - ci[:, 0], ci[:, 1] - np.array(v)],
                  capsize=3, error_kw=dict(ecolor="#555555", lw=1))
    ax[j].axhline(R["ref_sliced"]["pd"][i], color="#444444", ls=":", lw=1.2)
    ax[j].text(-0.55, R["ref_sliced"]["pd"][i] + 0.015, "CPU reference (YOLOv8n, 25 ep)", ha="left", fontsize=10, color="#444444")
    ax[j].set_xticks(x); ax[j].set_xticklabels([l for _, l in ARCH]); ax[j].set_ylim(0, 1)
    ax[j].set_title(title, fontsize=12); ax[j].grid(axis="y", alpha=.3)
ax[0].set_ylabel("Recall (Pd), test years 2010 + 2018")
ax[0].legend(frameon=False, fontsize=10, loc="lower left")
for k, (dk, col) in enumerate([("base", GREY), ("aug", AMBER)]):
    ax[2].bar(x + (k - 0.5) * w, [E[f"{a}_{dk}60"]["train_time_s"] / 60 for a, _ in ARCH], w, color=col)
ax[2].set_xticks(x); ax[2].set_xticklabels([l.split("\n")[0] for _, l in ARCH]); ax[2].set_ylabel("minutes")
ax[2].set_title("Training time, 60 epochs (RTX 3070 Ti)", fontsize=12); ax[2].grid(axis="y", alpha=.3)
for a_, t in zip(ax, "abc"):
    a_.text(-0.12, 1.04, f"({t})", transform=a_.transAxes, fontweight="bold", fontsize=13)
fig.tight_layout()
fig.savefig("../docs/figures/architecture_comparison.png", dpi=160, facecolor="white")
