"""Paper-style figures: (1) end-to-end pipeline, (2) detector and re-check classifier architecture.

  python fig_pipeline.py   ->  docs/figures/pipeline.png, docs/figures/architecture.png
Layer sizes in the architecture figure are read from the trained weights (weights/yolov8n_aug40.pt).
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"]})
INK, GREY, LIGHT = "#3a3a3a", "#8a8a8a", "#f3f3f3"
AMBER, AMBER_BG = "#c47f12", "#fbeedb"
STYLE = {"base": (LIGHT, INK, "-"), "new": (AMBER_BG, AMBER, "-"), "opt": ("white", GREY, "--"),
         "eval": ("#e6e6e6", INK, "-")}


def box(ax, x0, y0, x1, y1, title, body="", kind="base", tsize=10.5, bsize=8.8, align="center"):
    fc, ec, ls = STYLE[kind]
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.3, ls=ls))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if body:
        n = body.count("\n") + 1
        ax.text(cx, cy + 0.11 * n + 0.04, title, ha="center", va="center", fontsize=tsize, fontweight="bold", color=INK)
        tx = cx if align == "center" else x0 + 0.12
        ax.text(tx, cy + 0.11 * n - 0.2, body, ha=align, va="top", fontsize=bsize, color=INK, linespacing=1.35)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=tsize, fontweight="bold", color=INK)


def arrow(ax, pts, color=INK, ls="-", lw=1.3, label=None, lpos=None, lsize=8.5):
    for (xa, ya), (xb, yb) in zip(pts[:-2], pts[1:-1]):
        ax.plot([xa, xb], [ya, yb], color=color, lw=lw, ls=ls, solid_capstyle="butt")
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0, mutation_scale=13))
    if label:
        ax.text(*lpos, label, fontsize=lsize, color=color, ha="center", va="bottom", style="italic")


# =============================================================== Fig. 1 pipeline
fig, ax = plt.subplots(figsize=(18, 6.9))
ax.set_xlim(0, 18); ax.set_ylim(0, 6.9); ax.axis("off")

for x, t in [(1.5, "Data"), (4.4, "Split"), (10.8, "Training   ·   Test   ·   Validation"), (16.9, "Evaluation")]:
    ax.text(x, 6.62, t, ha="center", fontsize=11.5, fontweight="bold", color=GREY)
ax.plot([0.2, 17.8], [6.48, 6.48], color="#cccccc", lw=0.8)

box(ax, 0.2, 3.75, 2.8, 6.2, "Public SSS dataset",
    "Santos et al. 2024\nGavia AUV · Marine Sonic\n900 / 1800 kHz\n1,170 JPEG frames\nlabels: MILCO, NOMBO")
box(ax, 0.2, 0.35, 2.8, 3.2, "Duplicate screening",
    "down-scaled re-exports\nof the same scene\n417 pairs found\n→ 753 unique scenes")
arrow(ax, [(1.5, 3.75), (1.5, 3.2)])
arrow(ax, [(2.8, 1.8), (3.3, 1.8)])

# year split with three compartments aligned to the lanes
ax.add_patch(FancyBboxPatch((3.3, 0.35), 2.2, 5.85, boxstyle="round,pad=0,rounding_size=0.08", fc=LIGHT, ec=INK, lw=1.3))
ax.text(4.4, 5.92, "Official year split", ha="center", fontsize=10.5, fontweight="bold", color=INK)
for y in (3.85, 1.45):
    ax.plot([3.3, 5.5], [y, y], color=GREY, lw=0.8, ls=":")
ax.text(4.4, 5.05, "train\n2015 · 2017 · 2021\n85 % of scenes", ha="center", va="center", fontsize=8.8, color=INK, linespacing=1.4)
ax.text(4.4, 2.65, "test\n2010 · 2018\n578 frames\n114 objects", ha="center", va="center", fontsize=8.8, color=INK, linespacing=1.4)
ax.text(4.4, 0.9, "val\n15 % of train scenes\n26 frames · 47 obj.", ha="center", va="center", fontsize=8.8, color=INK, linespacing=1.3)

# ---- training lane (y 4.3-5.9)
box(ax, 6.0, 4.3, 8.6, 5.9, "Base tiles", "320 × 320 px\nobject-centred + background\n1,046 tiles")
box(ax, 9.0, 4.3, 12.7, 5.9, "Size-matched augmentation",
    "scale-down: objects → 26–50 px, year-balanced\nshadow-aware copy-paste (same side of nadir)\nspeckle · gamma · blur · JPEG re-encode · 1,081 tiles",
    kind="new")
box(ax, 13.1, 4.3, 15.6, 5.9, "YOLOv8n training", "COCO init · 40 epochs\nflips, brightness\nlast-epoch weights")
arrow(ax, [(5.5, 5.1), (6.0, 5.1)])
arrow(ax, [(8.6, 5.1), (9.0, 5.1)])
arrow(ax, [(12.7, 5.1), (13.1, 5.1)])
ax.text(16.7, 5.1, "architecture\n→ Fig. 2", ha="center", va="center", fontsize=8.8, color=GREY, style="italic")

# ---- test lane (y 1.85-3.6)
box(ax, 6.0, 1.85, 8.6, 3.6, "Test frame", "1024 × 1024 px\n(416-px exports up-sampled)\ngrey → 3 channels")
box(ax, 9.0, 1.85, 11.3, 3.6, "Sliced inference", "320-px windows\nstride 256\npoint NMS", kind="new")
box(ax, 11.7, 1.85, 13.1, 3.6, "Candidates", "(x, y,\nscore,\nclass)")
box(ax, 13.5, 1.85, 15.6, 3.6, "Cascade re-check", "highlight–shadow\nprofile + LR\n(optional)", kind="opt")
arrow(ax, [(5.5, 2.7), (6.0, 2.7)])
arrow(ax, [(8.6, 2.7), (9.0, 2.7)])
arrow(ax, [(11.3, 2.7), (11.7, 2.7)])
arrow(ax, [(13.1, 2.7), (13.5, 2.7)], color=GREY, ls="--")
arrow(ax, [(15.6, 2.7), (16.0, 2.7)])
arrow(ax, [(12.4, 3.6), (12.4, 3.75), (15.8, 3.75), (15.8, 2.95), (16.0, 2.95)], color=INK, label="detector alone", lpos=(14.55, 3.77), lsize=8)
arrow(ax, [(14.35, 4.3), (14.35, 4.05), (10.15, 4.05), (10.15, 3.6)], label="weights", lpos=(12.2, 4.08))

# ---- validation lane (y 0.35-1.45)
box(ax, 11.7, 0.38, 15.6, 1.5, "Operating points on val",
    "detector: max Pd at ≤ 4 FA/frame\nre-check: keep ≥ 95 % of true detections", bsize=8.6)
arrow(ax, [(5.5, 0.9), (11.7, 0.9)], color=GREY, ls="--", label="val scenes (not used for training)", lpos=(8.6, 0.93))
arrow(ax, [(14.55, 1.5), (14.55, 1.85)], color=GREY, ls="--")
arrow(ax, [(12.4, 1.5), (12.4, 1.85)], color=GREY, ls="--")
arrow(ax, [(15.6, 0.9), (16.0, 0.9)], color=GREY, ls="--")

# ---- evaluation
box(ax, 16.0, 0.35, 17.8, 3.6, "Test-year metrics",
    "point matching\n|p − c| ≤ max(w, h)\n\nFROC: Pd, Pdc\nat 0.25–8 FA/frame\n\nframe bootstrap\n95 % CI, paired Δ\n\nval → test\nthreshold transfer",
    kind="eval", bsize=8.4)

fig.savefig("../docs/figures/pipeline.png", dpi=170, bbox_inches="tight", facecolor="white")
plt.close(fig)


# =============================================================== Fig. 2 architecture
from ultralytics import YOLO  # noqa: E402

net = YOLO("weights/yolov8n_aug40.pt").model
npar = [sum(p.numel() for p in l.parameters()) for l in net.model]
P_BB, P_NECK, P_HEAD = sum(npar[:10]), sum(npar[10:22]), sum(npar[22:])
P_ALL = sum(npar)

fig = plt.figure(figsize=(15, 10.2))
gs = fig.add_gridspec(2, 1, height_ratios=[2.55, 1], hspace=0.08)
ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 15.15); ax.set_ylim(0.0, 8.1); ax.axis("off")
ax.text(0.0, 7.95, "(a)", fontsize=14, fontweight="bold")
ax.text(0.45, 7.95, f"YOLOv8n detector · {P_ALL / 1e6:.2f} M parameters · input 320 × 320 tile",
        fontsize=12, color=INK, va="baseline")
for x, t, n in [(2.6, "Backbone", P_BB), (6.5, "Neck: top-down (FPN)", None), (10.3, "Neck: bottom-up (PAN)", P_NECK),
                (13.7, "Head", P_HEAD)]:
    ax.text(x, 7.42, t, ha="center", fontsize=11, fontweight="bold", color=GREY)
    if n:
        ax.text(x if t == "Backbone" or t == "Head" else 8.4, 7.12, f"{n / 1e6:.2f} M" if t != "Neck: bottom-up (PAN)" else f"neck {n / 1e6:.2f} M",
                ha="center", fontsize=9, color=GREY)
ax.plot([0.2, 14.9], [7.0, 7.0], color="#cccccc", lw=0.8)

ROW = {"in": 6.45, 160: 5.6, 80: 4.75, 40: 3.75, 20: 2.5, 10: 1.25}
H = 0.33
for r, lab in [(160, "160²"), (80, "80²"), (40, "P3 · 40²\nstride 8"), (20, "P4 · 20²\nstride 16"), (10, "P5 · 10²\nstride 32")]:
    ax.text(0.55, ROW[r], lab, ha="center", va="center", fontsize=8.8, color=GREY, linespacing=1.2)


def blk(x0, x1, r, title, body, kind="base"):
    box(ax, x0, ROW[r] - H, x1, ROW[r] + H, title, body, kind=kind, tsize=9.8, bsize=8.6)


box(ax, 1.25, ROW["in"] - 0.26, 3.95, ROW["in"] + 0.26, "SSS tile 320 × 320 (grey → 3 ch)", tsize=9.5)
blk(1.25, 3.95, 160, "Conv 3×3 s2", "16 ch")
blk(1.25, 3.95, 80, "Conv s2 → C2f ×1", "32 ch")
blk(1.25, 3.95, 40, "Conv s2 → C2f ×2", "64 ch")
blk(1.25, 3.95, 20, "Conv s2 → C2f ×2", "128 ch")
blk(1.25, 3.95, 10, "Conv s2 → C2f ×1 → SPPF", "256 ch")
for a, b in [("in", 160), (160, 80), (80, 40), (40, 20), (20, 10)]:
    arrow(ax, [(2.6, ROW[a] - (0.26 if a == "in" else H)), (2.6, ROW[b] + H)])

blk(5.1, 7.9, 20, "Up ×2 + P4 → C2f", "128 ch")
blk(5.1, 7.9, 40, "Up ×2 + P3 → C2f", "64 ch")
blk(8.9, 11.7, 20, "Conv s2 + FPN-P4 → C2f", "128 ch")
blk(8.9, 11.7, 10, "Conv s2 + SPPF → C2f", "256 ch")

# lateral and top-down connections
arrow(ax, [(3.95, ROW[40]), (5.1, ROW[40])])
arrow(ax, [(3.95, ROW[20]), (5.1, ROW[20])])
ax.plot([3.95, 6.5], [ROW[10], ROW[10]], color=INK, lw=1.3)
arrow(ax, [(6.5, ROW[10]), (6.5, ROW[20] - H)])
arrow(ax, [(6.5, ROW[10]), (8.9, ROW[10])])
arrow(ax, [(6.5, ROW[20] + H), (6.5, ROW[40] - H)])
# bottom-up
arrow(ax, [(7.9, ROW[40]), (12.2, ROW[40])])
arrow(ax, [(8.35, ROW[40]), (8.35, ROW[20] + 0.12), (8.9, ROW[20] + 0.12)])
arrow(ax, [(7.9, ROW[20] - 0.12), (8.9, ROW[20] - 0.12)])
arrow(ax, [(10.3, ROW[20] - H), (10.3, ROW[10] + H)])
arrow(ax, [(11.7, ROW[20]), (12.2, ROW[20])])
arrow(ax, [(11.7, ROW[10]), (12.2, ROW[10])])

for r, s, g in [(40, 8, 40), (20, 16, 20), (10, 32, 10)]:
    blk(12.2, 15.0, r, f"Detect · stride {s}", f"box 4 × 16 (DFL) · class 2", kind="eval")
ax.text(13.6, 0.12, f"decoupled, anchor-free · {40 * 40 + 20 * 20 + 10 * 10:,} points per tile\n→ score, box, MILCO / NOMBO → NMS",
        ha="center", fontsize=8.8, color=INK, linespacing=1.35)
ax.text(1.25, 0.32, "+ = concatenation · C2f: CSP block with 2 convolutions and n bottlenecks · SPPF: spatial pyramid pooling (fast)",
        ha="left", fontsize=8.6, color=GREY)
ax.annotate("30–50 px object ≈ 4–6 cells on P3", xy=(12.2, ROW[40] + 0.2), xytext=(9.0, 5.2), fontsize=9, color=AMBER,
            arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.0, mutation_scale=10), ha="left", style="italic")

# ---- (b) re-check classifier
bx = fig.add_subplot(gs[1]); bx.set_xlim(0, 15.15); bx.set_ylim(0, 3.1); bx.axis("off")
bx.text(0.0, 2.92, "(b)", fontsize=14, fontweight="bold")
bx.text(0.45, 2.92, "Cascade re-check: physical-feature classifier (64 features, logistic regression)", fontsize=12, color=INK,
        va="baseline")
Y0, Y1 = 0.2, 2.55


def bbox(x0, x1, title, body, kind="base"):
    box(bx, x0, Y0, x1, Y1, title, body, kind=kind, tsize=10, bsize=8.6)


bbox(0.2, 2.5, "Candidate", "(x, y, w, h)\non the 1024-px frame\nside of nadir known")
bx.add_patch(FancyBboxPatch((2.9, Y0), 4.3, Y1 - Y0, boxstyle="round,pad=0,rounding_size=0.08", fc=LIGHT, ec=INK, lw=1.3))
bx.text(5.05, 2.3, "Across-track profile", ha="center", fontsize=10, fontweight="bold", color=INK)
bx.text(5.05, 2.05, "61 samples, −1.5 w … +1.5 w (away from nadir)\nmean over ±h/4 rows · dB re seabed median",
        ha="center", va="top", fontsize=8.4, color=INK, linespacing=1.3)
ins = bx.inset_axes([3.3, 0.62, 3.5, 0.82], transform=bx.transData)
ins.set_facecolor("none")
import numpy as np  # noqa: E402
u = np.linspace(-1.5, 1.5, 61)
prof = 7 * np.exp(-((u + 0.05) / 0.18) ** 2) - 7 * (1 / (1 + np.exp(-(u - 0.35) / 0.06))) * (1 / (1 + np.exp((u - 1.3) / 0.06)))
prof += 0.6 * np.sin(9 * u) * np.exp(-u ** 2)
ins.plot(u, prof, color=INK, lw=1.4)
ins.axhline(0, color=GREY, lw=0.7, ls=":")
ins.set_xticks([-1.5, 0, 1.5]); ins.set_xticklabels(["−1.5w", "0", "+1.5w"], fontsize=7.5)
ins.set_yticks([]); ins.tick_params(length=2)
for s in ("top", "right", "left"):
    ins.spines[s].set_visible(False)
ins.text(0.0, 7.6, "highlight", fontsize=7.8, color=AMBER, ha="center")
ins.text(0.85, -4.6, "shadow", fontsize=7.8, color=AMBER, ha="center", va="bottom")
ins.text(-1.45, 6.0, "illustrative", fontsize=7.2, color=GREY, style="italic")
ins.set_ylim(-9.5, 10)
bbox(7.6, 9.8, "Feature vector", "61 profile values\n+ log w, log h\n+ range from nadir\n= 64")
bbox(10.2, 12.4, "Logistic regression", "standardised inputs\nC = 0.5\nclass-balanced\n3 classes")
bbox(12.8, 14.95, "Decision", "MILCO · NOMBO · seabed\nkeep if\n1 − p(seabed) ≥ t\n(t fixed on val)", kind="eval")
for xa, xb in [(2.5, 2.9), (7.2, 7.6), (9.8, 10.2), (12.4, 12.8)]:
    arrow(bx, [(xa, 1.37), (xb, 1.37)])

fig.savefig("../docs/figures/architecture.png", dpi=170, bbox_inches="tight", facecolor="white")
plt.close(fig)
