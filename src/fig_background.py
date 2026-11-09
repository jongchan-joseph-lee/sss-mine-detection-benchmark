"""Research background timeline (few boxes, large text).  python fig_background.py -> docs/figures/research_background.png"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"]})
INK, GREY, LIGHT, AMBER, AMBER_BG = "#333333", "#8a8a8a", "#f2f2f2", "#c47f12", "#fbeedb"

ERAS = [
    ("Physics-based ATR", "1960 – 2010s", [
        ("1960", "Matched filter", "Turin"),
        ("2003", "Highlight–shadow segmentation", "Reed et al."),
        ("2011", "Shadow as key cue", "Mukherjee et al.")]),
    ("Deep learning for sonar", "2016 –", [
        ("2016", "CNN target classification", "Williams"),
        ("2017", "Pretrained CNN transfer", "McKay et al."),
        ("2019", "Small-object copy-paste", "Kisantal et al.")]),
    ("Current issues", "2020 –", [
        ("2020", "Seabed shift degrades ATR", "de Bodinat et al."),
        ("2024", "Scarce public data", "Santos · Agrawal et al."),
        ("2024", "YOLO vs Transformer on SSS", "Steiniger")]),
]
USED = ["matched filter\nhighlight–shadow re-check", "YOLOv8 · RT-DETR · DINOv2\nshadow-aware copy-paste",
        "year split · size diagnosis\nlatency on CPU / GPU"]

fig, ax = plt.subplots(figsize=(14, 6.2))
ax.set_xlim(0, 14); ax.set_ylim(0, 6.2); ax.axis("off")
W, X0 = 4.3, [0.1, 4.85, 9.6]
for x, (title, span, items), used in zip(X0, ERAS, USED):
    ax.add_patch(FancyBboxPatch((x, 1.75), W, 4.2, boxstyle="round,pad=0,rounding_size=0.12", fc=LIGHT, ec=INK, lw=1.6))
    ax.text(x + W / 2, 5.55, title, ha="center", fontsize=16, fontweight="bold", color=INK)
    ax.text(x + W / 2, 5.15, span, ha="center", fontsize=13, color=GREY)
    for k, (yr, what, who) in enumerate(items):
        y = 4.45 - k * 0.9
        ax.text(x + 0.25, y, yr, fontsize=13.5, fontweight="bold", color=INK, va="center")
        ax.text(x + 1.05, y + 0.14, what, fontsize=13.5, color=INK, va="center")
        ax.text(x + 1.05, y - 0.22, who, fontsize=12, color=GREY, va="center", style="italic")
    ax.add_patch(FancyBboxPatch((x, 0.2), W, 1.05, boxstyle="round,pad=0,rounding_size=0.12", fc=AMBER_BG, ec=AMBER, lw=1.6))
    ax.text(x + W / 2, 1.0, "used in this project", ha="center", fontsize=12, color=AMBER, style="italic")
    ax.text(x + W / 2, 0.55, used, ha="center", va="center", fontsize=12.5, color=INK, linespacing=1.3)
    ax.annotate("", xy=(x + W / 2, 1.27), xytext=(x + W / 2, 1.73),
                arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.5, mutation_scale=15))
for x in X0[:2]:
    ax.annotate("", xy=(x + W + 0.43, 3.85), xytext=(x + W + 0.02, 3.85),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6, mutation_scale=16))
fig.savefig("../docs/figures/research_background.png", dpi=160, bbox_inches="tight", facecolor="white")
