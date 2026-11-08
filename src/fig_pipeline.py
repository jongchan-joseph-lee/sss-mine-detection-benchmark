"""Overview figures (few boxes, large text): (1) pipeline, (2) detector and re-check classifier.

  python fig_pipeline.py   ->  docs/figures/pipeline.png, docs/figures/architecture.png
Parameter counts are read from the trained weights (weights/yolov8n_aug40.pt); numbers in detail live in the README tables.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"]})
INK, GREY, LIGHT = "#333333", "#8a8a8a", "#f2f2f2"
AMBER, AMBER_BG = "#c47f12", "#fbeedb"
STYLE = {"base": (LIGHT, INK, "-"), "new": (AMBER_BG, AMBER, "-"), "opt": ("white", GREY, "--"), "eval": ("#e3e3e3", INK, "-")}
T, S = 15, 12.5                                          # title / sub font sizes


def box(ax, x0, y0, x1, y1, title, sub="", kind="base"):
    fc, ec, ls = STYLE[kind]
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.1",
                                fc=fc, ec=ec, lw=1.6, ls=ls))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if sub:
        ax.text(cx, cy + 0.17, title, ha="center", va="center", fontsize=T, fontweight="bold", color=INK)
        ax.text(cx, cy - 0.2, sub, ha="center", va="center", fontsize=S, color=INK)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=T, fontweight="bold", color=INK)


def arrow(ax, pts, color=INK, ls="-", label=None, lpos=None):
    for (xa, ya), (xb, yb) in zip(pts[:-2], pts[1:-1]):
        ax.plot([xa, xb], [ya, yb], color=color, lw=1.6, ls=ls, solid_capstyle="butt")
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6, ls=ls, shrinkA=0, shrinkB=0, mutation_scale=16))
    if label:
        ax.text(*lpos, label, fontsize=S, color=color, ha="center", va="bottom", style="italic")


# =============================================================== Fig. 1 pipeline
fig, ax = plt.subplots(figsize=(13, 4.1))
ax.set_xlim(0, 13); ax.set_ylim(0.3, 3.75); ax.axis("off")
TR, TE = (2.35, 3.45), (0.45, 1.55)                       # training lane, test lane (y0, y1)

box(ax, 0.1, 1.4, 2.1, 2.5, "SSS frames", "1,170 JPEG")
ax.add_patch(FancyBboxPatch((2.6, 0.45), 2.0, 3.0, boxstyle="round,pad=0,rounding_size=0.1", fc=LIGHT, ec=INK, lw=1.6))
ax.text(3.6, 3.1, "Year split", ha="center", fontsize=T, fontweight="bold", color=INK)
ax.text(3.6, 2.45, "train\n2015 · 17 · 21", ha="center", va="center", fontsize=S, color=INK, linespacing=1.3)
ax.text(3.6, 1.0, "test\n2010 · 18", ha="center", va="center", fontsize=S, color=INK, linespacing=1.3)
arrow(ax, [(2.1, 1.95), (2.6, 1.95)])

box(ax, 5.2, TR[0], 7.6, TR[1], "Augmentation", "26–50 px · copy-paste", kind="new")
box(ax, 8.1, TR[0], 10.2, TR[1], "YOLOv8n", "40 epochs")
arrow(ax, [(4.6, 2.9), (5.2, 2.9)])
arrow(ax, [(7.6, 2.9), (8.1, 2.9)])

box(ax, 5.2, TE[0], 7.6, TE[1], "Sliced inference", "320-px tiles", kind="new")
box(ax, 8.1, TE[0], 10.2, TE[1], "Re-check", "optional", kind="opt")
box(ax, 10.7, TE[0], 12.9, TE[1], "Evaluation", "Pd @ FA / frame", kind="eval")
arrow(ax, [(4.6, 1.0), (5.2, 1.0)])
arrow(ax, [(7.6, 1.0), (8.1, 1.0)], color=GREY, ls="--")
arrow(ax, [(10.2, 1.0), (10.7, 1.0)])
arrow(ax, [(9.15, TR[0]), (9.15, 1.95), (6.4, 1.95), (6.4, TE[1])], label="weights", lpos=(7.8, 1.97))

fig.savefig("../docs/figures/pipeline.png", dpi=170, bbox_inches="tight", facecolor="white")
plt.close(fig)


# =============================================================== Fig. 2 architecture
from ultralytics import YOLO  # noqa: E402

npar = [sum(p.numel() for p in l.parameters()) for l in YOLO("weights/yolov8n_aug40.pt").model.model]
P_BB, P_NECK, P_HEAD, P_ALL = sum(npar[:10]), sum(npar[10:22]), sum(npar[22:]), sum(npar)
M = lambda n: f"{n / 1e6:.2f} M"  # noqa: E731

fig = plt.figure(figsize=(13, 6.4))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.05], hspace=0.25)

# (a) detector
ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 13); ax.set_ylim(0, 2.9); ax.axis("off")
ax.text(0.0, 2.65, "(a)", fontsize=16, fontweight="bold")
ax.text(0.45, 2.65, f"YOLOv8n detector · {M(P_ALL)} parameters", fontsize=T, color=INK)
Y = (0.75, 1.95)
box(ax, 0.1, Y[0], 2.1, Y[1], "Input", "320 × 320 tile")
box(ax, 2.35, Y[0], 4.75, Y[1], "Backbone", f"CSP · {M(P_BB)}")
box(ax, 5.25, Y[0], 7.65, Y[1], "Neck", f"FPN + PAN · {M(P_NECK)}")
for i, (s, y) in enumerate([(8, 1.6), (16, 1.12), (32, 0.64)]):
    kind = "new" if s == 8 else "eval"
    ax.add_patch(FancyBboxPatch((8.15, y - 0.2), 2.1, 0.4, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=STYLE[kind][0], ec=STYLE[kind][1], lw=1.4))
    ax.text(9.2, y, f"head · stride {s}", ha="center", va="center", fontsize=S, color=INK)
    arrow(ax, [(7.65, 1.35), (7.9, 1.35), (7.9, y), (8.15, y)]) if i == 0 else \
        ax.plot([7.9, 7.9, 8.15], [1.35, y, y], color=INK, lw=1.6)
box(ax, 10.75, Y[0], 12.9, Y[1], "Output", "box + class")
ax.plot([10.25, 10.5, 10.5], [1.6, 1.6, 0.64], color=INK, lw=1.6)
ax.plot([10.25, 10.5], [1.12, 1.12], color=INK, lw=1.6)
ax.plot([10.25, 10.5], [0.64, 0.64], color=INK, lw=1.6)
arrow(ax, [(10.5, 1.35), (10.75, 1.35)])
arrow(ax, [(2.1, 1.35), (2.35, 1.35)])
arrow(ax, [(4.75, 1.35), (5.25, 1.35)])
ax.text(9.2, 0.18, "30–50 px object ≈ 4–6 cells at stride 8", ha="center", fontsize=S, color=AMBER, style="italic")

# (b) re-check classifier
bx = fig.add_subplot(gs[1]); bx.set_xlim(0, 13); bx.set_ylim(0, 3.0); bx.axis("off")
bx.text(0.0, 2.75, "(b)", fontsize=16, fontweight="bold")
bx.text(0.45, 2.75, "Re-check classifier · highlight–shadow profile", fontsize=T, color=INK)
Y = (0.35, 2.3)
box(bx, 0.1, Y[0], 2.1, Y[1], "Candidate", "x, y, w, h")
bx.add_patch(FancyBboxPatch((2.6, Y[0]), 3.9, Y[1] - Y[0], boxstyle="round,pad=0,rounding_size=0.1", fc=LIGHT, ec=INK, lw=1.6))
bx.text(4.55, 2.0, "Across-track profile", ha="center", fontsize=T, fontweight="bold", color=INK)
ins = bx.inset_axes([2.85, 0.85, 3.4, 0.95], transform=bx.transData)
ins.set_facecolor("none")
u = np.linspace(-1.5, 1.5, 61)
prof = 7 * np.exp(-((u + 0.05) / 0.18) ** 2) - 7 / (1 + np.exp(-(u - 0.35) / 0.06)) / (1 + np.exp((u - 1.3) / 0.06))
ins.plot(u, prof, color=INK, lw=1.8)
ins.axhline(0, color=GREY, lw=0.8, ls=":")
ins.set_xticks([-1.5, 0, 1.5]); ins.set_xticklabels(["nadir side", "object", "shadow side"], fontsize=10.5)
ins.set_yticks([]); ins.tick_params(length=2)
for s_ in ("top", "right", "left"):
    ins.spines[s_].set_visible(False)
ins.text(0.0, 7.8, "highlight", fontsize=11.5, color=AMBER, ha="center")
ins.text(0.85, -5.0, "shadow", fontsize=11.5, color=AMBER, ha="center", va="bottom")
ins.set_ylim(-9.5, 10.5)
box(bx, 7.0, Y[0], 8.9, Y[1], "Features", "64 values")
box(bx, 9.4, Y[0], 11.0, Y[1], "LR", "3 classes")
box(bx, 11.5, Y[0], 12.9, Y[1], "Keep", "or drop", kind="eval")
for xa, xb in [(2.1, 2.6), (6.5, 7.0), (8.9, 9.4), (11.0, 11.5)]:
    arrow(bx, [(xa, 1.32), (xb, 1.32)])

fig.savefig("../docs/figures/architecture.png", dpi=170, bbox_inches="tight", facecolor="white")
plt.close(fig)
