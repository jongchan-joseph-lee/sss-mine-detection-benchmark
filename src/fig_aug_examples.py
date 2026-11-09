"""Examples of the size-matched augmentation (prep_aug.py) on real training frames.

  python fig_aug_examples.py   ->  docs/figures/augmentation_examples.png
Same operations as prep_aug.py, with fixed parameters so the figure is reproducible.
"""
import io
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image, ImageFilter

plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "font.size": 12})
AMBER = "#e39a2a"
SRC = tuple(os.environ.get("AUG_SRC", "2015,0059_2015.jpg,0").split(",")); SRC = (SRC[0], SRC[1], int(SRC[2]))   # training frame, object
DST = tuple(os.environ.get("AUG_DST", "2021,0022_2021.jpg").split(","))                                     # object-free frame, other year
S = json.load(open("screen.json"))
REC = {(r["year"], r["file"]): r for r in S["records"]}
T = 320
Z = 160                                  # zoom window shown around the object (px of the 1024 frame scale)
TARGET = 30                              # object size after scale-down (prep_aug draws 26-50 px)


def frame(y, f):
    im = Image.open(f"raw/{y}/{y}/{f}").convert("L")
    return np.asarray(im.resize((1024, 1024), Image.BICUBIC) if im.size[0] != 1024 else im, np.float32)


def nadir(a):
    col = a.mean(0); c0, c1 = 307, 716
    idx = np.where(col[c0:c1] < np.percentile(col, 15) + 5)[0] + c0
    return int(idx.mean()) if len(idx) else c0 + int(np.argmin(col[c0:c1]))


def crop(a, cx, cy, size=T):
    x0 = int(np.clip(cx - size / 2, 0, a.shape[1] - size)); y0 = int(np.clip(cy - size / 2, 0, a.shape[0] - size))
    return a[y0:y0 + size, x0:x0 + size], x0, y0


def show(ax, img, box=None, title="", sub=""):
    ax.imshow(np.clip(img, 0, 255), cmap="gray", vmin=0, vmax=255)
    if box is not None:
        x, y, w, h = box
        ax.add_patch(Rectangle((x - w / 2 - 4, y - h / 2 - 4), w + 8, h + 8, fill=False, ec=AMBER, lw=1.8))
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_color("#999999")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=6)
    if sub:
        ax.text(0.5, -0.06, sub, transform=ax.transAxes, ha="center", va="top", fontsize=11.5, color="#333333")


a = frame(*SRC[:2])
c, x, y, w, h = REC[SRC[:2]]["labels"][SRC[2]]
x, y, w, h = x * 1024, y * 1024, w * 1024, h * 1024
side = max(w, h)

fig = plt.figure(figsize=(15, 5.6))
gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 0.5, 0.5], wspace=0.08, hspace=0.12)

# (a) original tile
t, x0, y0 = crop(a, x, y, Z)
show(fig.add_subplot(gs[:, 0]), t, (x - x0, y - y0, w, h), "(a) Original", f"{SRC[0]} · object {side:.0f} px")

# (b) scale-down to ~38 px
s = TARGET / side
n = int(round(1024 * s))
a2 = np.asarray(Image.fromarray(a.astype(np.uint8)).resize((n, n), Image.BICUBIC), np.float32)
t2, x0, y0 = crop(a2, x * s, y * s, Z)
show(fig.add_subplot(gs[:, 1]), t2, (x * s - x0, y * s - y0, w * s, h * s), "(b) Scale-down", f"same frame × {s:.2f} → {side * s:.0f} px")

# (c) shadow-aware copy-paste onto another year's seabed, same side of nadir
nd = nadir(a); out = 1 if x > nd else -1
xa, xb = sorted([int(x - out * 1.0 * w), int(x + out * 3.0 * w)]); ya, yb = int(y - 1.2 * h), int(y + 1.2 * h)
xa, xb, ya, yb = max(0, xa), min(1024, xb), max(0, ya), min(1024, yb)
patch = a[ya:yb, xa:xb]
ox, oy = x - xa, y - ya
ph, pw = patch.shape; nh, nw = int(ph * s), int(pw * s)
p = np.asarray(Image.fromarray(patch.astype(np.uint8)).resize((nw, nh), Image.BICUBIC), np.float32)
d = frame(*DST).copy(); ndd = nadir(d)
cx, cy = ndd + out * 0.25 * 1024, 560
px, py = int(cx - ox * s), int(cy - oy * s)
dst = d[py:py + nh, px:px + nw]
ring = np.concatenate([p[:2].ravel(), p[-2:].ravel(), p[:, :2].ravel(), p[:, -2:].ravel()])
ratio = np.clip((p + 1) / (np.median(ring) + 1), 0.05, 8.0)
yy, xx = np.mgrid[0:nh, 0:nw].astype(np.float32)
m = np.clip((1 - np.sqrt(((xx - (nw - 1) / 2) / (nw / 2)) ** 2 + ((yy - (nh - 1) / 2) / (nh / 2)) ** 2)) / 0.35, 0, 1)
d[py:py + nh, px:px + nw] = dst * ratio ** m
t3, x0, y0 = crop(d, cx, cy, Z)
show(fig.add_subplot(gs[:, 2]), t3, (cx - x0, cy - y0, w * s, h * s), "(c) Shadow-aware copy-paste",
     f"{SRC[0]} object + shadow → {DST[0]} seabed")

# (d) photometric variants of tile (b)
rng = np.random.default_rng(3)
base = t2.astype(np.float32)
speckle = base * rng.gamma(6, 1 / 6, base.shape)
gamma = 255 * (np.clip(base, 0, 255) / 255) ** 1.35
blur = np.asarray(Image.fromarray(base.astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.1)), np.float32)
buf = io.BytesIO(); Image.fromarray(base.astype(np.uint8)).save(buf, "JPEG", quality=35)
jpeg = np.asarray(Image.open(buf).convert("L"), np.float32)
for k, (img, lab) in enumerate([(speckle, "speckle"), (gamma, "gamma"), (blur, "blur"), (jpeg, "JPEG q35")]):
    ax = fig.add_subplot(gs[k // 2, 3 + k % 2])
    show(ax, img)
    ax.text(0.5, -0.04, lab, transform=ax.transAxes, ha="center", va="top", fontsize=11.5, color="#333333")
fig.text(0.855, 0.905, "(d) Photometric (on b)", ha="center", fontsize=13, fontweight="bold")

fig.savefig(os.environ.get("AUG_OUT", "../docs/figures/augmentation_examples.png"), dpi=150, bbox_inches="tight", facecolor="white")
