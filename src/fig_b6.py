"""Detector trade-off: test-year recall vs per-frame inference latency (sliced inference), CPU and GPU.

  python fig_b6.py   ->  docs/figures/detector_tradeoff.png
Inputs: gpu_runs/b5_eval_gpu.json (recall, 95% CI) and gpu_runs/latency.json (bench_latency.py, RTX 3070 Ti PC).
"""
import json

import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "font.size": 12,
                     "axes.spines.top": False, "axes.spines.right": False})
R = json.load(open("gpu_runs/b5_eval_gpu.json"))
L = json.load(open("gpu_runs/latency.json"))["runs"]
GREY, AMBER, INK = "#8a8a8a", "#d08a1e", "#333333"
ARCH = [("n", "YOLOv8n · 3.1 M", "o"), ("s", "YOLOv8s · 11.1 M", "s"), ("rtdetr", "RT-DETR-l · 32.8 M", "^")]
FA = 8
i = R["ref_sliced"]["fa_grid"].index(FA)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
for ax, dev, title in [(axes[0], "cpu", "CPU (Intel, 8 threads)"), (axes[1], "cuda", "GPU (RTX 3070 Ti)")]:
    for a, lab, mk in ARCH:
        for dk, col, fill in [("base", GREY, "white"), ("aug", AMBER, AMBER)]:
            k = f"{a}_{dk}60"
            x = L[k][f"{dev}_sliced_ms"]["median"]
            y = R[f"{k}_sliced"]["pd"][i]
            lo, hi = R[f"{k}_sliced"]["ci95"][i]
            ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt=mk, ms=10, mfc=fill, mec=col, mew=2, ecolor=col, elinewidth=1.2, capsize=3)
        xa = L[f"{a}_aug60"][f"{dev}_sliced_ms"]["median"]
        off = {"n": (8, 22) if dev == "cpu" else (-8, 22), "s": (8, 22), "rtdetr": (-8, 22)}[a]
        ax.annotate(lab, (xa, R[f"{a}_aug60_sliced"]["pd"][i]), xytext=off, textcoords="offset points", fontsize=11,
                    color=INK, ha="right" if off[0] < 0 else "left")
    ax.set_xscale("log")
    ticks, lim = ({"cpu": ([100, 200, 500, 1000, 2000], (80, 3000)), "cuda": ([10, 20, 50, 100], (8, 150))})[dev]
    ax.set_xticks(ticks); ax.set_xticklabels([f"{t:,}" for t in ticks]); ax.minorticks_off(); ax.set_xlim(*lim)
    ax.set_title(title, fontsize=12.5)
    ax.set_xlabel("Latency per 1024-px frame, sliced inference (ms, median)")
    ax.grid(alpha=.3, which="both")
axes[0].set_ylabel(f"Recall (Pd) @ {FA} false alarms / frame")
axes[0].set_ylim(0.55, 1.0)
h = [plt.Line2D([], [], marker="o", ls="", ms=9, mfc="white", mec=GREY, mew=2, label="default augmentation only"),
     plt.Line2D([], [], marker="o", ls="", ms=9, mfc=AMBER, mec=AMBER, mew=2, label="+ size-matched augmentation")]
axes[0].legend(handles=h, frameon=False, loc="lower left", fontsize=10.5)
for a_, t in zip(axes, "ab"):
    a_.text(-0.1, 1.04, f"({t})", transform=a_.transAxes, fontweight="bold", fontsize=13)
fig.tight_layout()
fig.savefig("../docs/figures/detector_tradeoff.png", dpi=160, facecolor="white")
