"""
Offline augmentation for the year-split detector (training scenes only; val/test untouched).

Motivation (miss diagnosis, src/b5_miss_diag.json): misses concentrate on 30-50 px objects
(1024-px scale), while training objects are dominated by 2015 scenes with larger targets.

  A. scale-down     : training frames resized by s so that objects land in 26-50 px, then
                      object-centred 320-px tiles. Per-object repeats are year-balanced.
  B. copy-paste     : object + its acoustic shadow cut from a training frame, rescaled to 26-50 px,
                      intensity-matched and pasted on the SAME side of nadir of another training
                      frame (shadow keeps pointing away from nadir). Destination frames are
                      mostly object-free 2017/2021 scenes, so the seabed context is diversified.
  C. photometric    : applied to every augmented tile with p=0.5 each -
                      multiplicative speckle (gamma, L=4-16), gamma/contrast, blur, JPEG re-encode.

The target size range (26-50 px) covers the 2017/2021 training objects (median 44-46 px); it was
chosen after looking at the test-year miss diagnosis, which is stated in the README.

Output: tiles/train_aug/{images,labels} and tiles/milco_tiles_aug.yaml (train = base + aug, val = base val).
"""
import io
import json
import os
import random

import numpy as np
from PIL import Image, ImageFilter

S = json.load(open("screen.json"))
TRAIN = ["2015", "2017", "2021"]
T = 320
TARGET = (26, 50)                        # object max side after rescale, px at 1024 scale
REPS_SCALE = {"2015": 1, "2017": 6, "2021": 3}
N_PASTE = 700
OUT = "tiles/train_aug"

# ---- identical scene list and val split as prep_tiles.py
random.seed(0)
rep = {}
for r in S["records"]:
    k = (r["year"], r["scene"])
    if k not in rep or r["size"] > rep[k]["size"]:
        rep[k] = r
scenes = sorted(rep.values(), key=lambda r: (r["year"], r["file"]))
tr = [r for r in scenes if r["year"] in TRAIN]
random.shuffle(tr)
val_ids = {(r["year"], r["file"]) for r in tr[: int(0.15 * len(tr))]}
train = [r for r in tr if (r["year"], r["file"]) not in val_ids]

rng = np.random.default_rng(1)
os.makedirs(f"{OUT}/images", exist_ok=True)
os.makedirs(f"{OUT}/labels", exist_ok=True)


def frame(r):
    im = Image.open(f"raw/{r['year']}/{r['year']}/{r['file']}").convert("L")
    if im.size[0] != 1024:
        im = im.resize((1024, 1024), Image.BICUBIC)
    return np.asarray(im, np.float32)


def boxes(r):
    return [(c, x * 1024, y * 1024, w * 1024, h * 1024) for c, x, y, w, h in r["labels"]
            if w * 1024 >= 4 and h * 1024 >= 4]


def nadir(a):                            # same rule as b2.nadir_band
    W = a.shape[1]
    col = a.mean(0)
    c0, c1 = int(W * 0.3), int(W * 0.7)
    dark = col[c0:c1] < np.percentile(col, 15) + 5
    idx = np.where(dark)[0] + c0
    return int(idx.mean()) if len(idx) else c0 + int(np.argmin(col[c0:c1]))


def photometric(a):
    a = a.astype(np.float32)
    if rng.random() < 0.5:                                    # multiplicative speckle
        L = rng.uniform(4, 16)
        a = a * rng.gamma(L, 1 / L, a.shape)
    if rng.random() < 0.5:                                    # gamma / contrast
        g = rng.uniform(0.7, 1.4)
        a = 255 * (np.clip(a, 0, 255) / 255) ** g * rng.uniform(0.8, 1.2)
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if rng.random() < 0.5:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.2)))
    if rng.random() < 0.5:                                    # JPEG re-encode
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=int(rng.integers(35, 85)))
        im = Image.open(buf).convert("L")
    return im


def crop_boxes(bx, x0, y0):
    out = []
    for c, x, y, w, h in bx:
        a0, a1 = max(x - w / 2, x0), min(x + w / 2, x0 + T)
        b0, b1 = max(y - h / 2, y0), min(y + h / 2, y0 + T)
        if a1 - a0 >= 0.6 * w and b1 - b0 >= 0.6 * h:
            out.append((c, (a0 + a1) / 2 - x0, (b0 + b1) / 2 - y0, a1 - a0, b1 - b0))
    return out


def write_tile(a, bx, x, y, name):
    H, W = a.shape
    x0 = int(np.clip(x - rng.uniform(0.2, 0.8) * T, 0, W - T))
    y0 = int(np.clip(y - rng.uniform(0.2, 0.8) * T, 0, H - T))
    lab = crop_boxes(bx, x0, y0)
    photometric(a[y0:y0 + T, x0:x0 + T]).convert("RGB").save(f"{OUT}/images/{name}.png")
    with open(f"{OUT}/labels/{name}.txt", "w") as f:
        for c, cx, cy, w, h in lab:
            f.write(f"{c} {cx / T:.6f} {cy / T:.6f} {w / T:.6f} {h / T:.6f}\n")
    return len(lab)


stats = {"scale": {}, "paste": {}, "size_after": []}
cache = {}


def get(r):
    k = (r["year"], r["file"])
    if k not in cache:
        cache[k] = frame(r)
    return cache[k]


# ---- A. year-balanced scale-down
for r in train:
    bx = boxes(r)
    if not bx:
        continue
    a = get(r)
    for i, (c, x, y, w, h) in enumerate(bx):
        for k in range(REPS_SCALE[r["year"]]):
            side = max(w, h)
            s = float(np.clip(rng.uniform(*TARGET) / side, 0.4, 1.0))
            n = int(round(1024 * s))
            if n < T:
                continue
            a2 = np.asarray(Image.fromarray(a.astype(np.uint8)).resize((n, n), Image.BICUBIC), np.float32)
            bx2 = [(cc, xx * s, yy * s, ww * s, hh * s) for cc, xx, yy, ww, hh in bx]
            stem = f"{r['year']}_{r['file'][:-4]}_s{i}_{k}"
            if write_tile(a2, bx2, x * s, y * s, stem):
                stats["scale"][r["year"]] = stats["scale"].get(r["year"], 0) + 1
                stats["size_after"].append(side * s)

# ---- B. shadow-aware copy-paste
src_objs = []
for r in train:
    a = get(r)
    nd = nadir(a)
    for c, x, y, w, h in boxes(r):
        out = 1 if x > nd else -1                           # outward direction (shadow side)
        x_in, x_out = x - out * 1.0 * w, x + out * 3.0 * w     # 1 w inward, 3 w outward (shadow)
        xa, xb = int(max(0, min(x_in, x_out))), int(min(1024, max(x_in, x_out)))
        ya, yb = int(max(0, y - 1.2 * h)), int(min(1024, y + 1.2 * h))
        if xb - xa < 8 or yb - ya < 8:
            continue
        src_objs.append(dict(r=r, c=c, out=out, patch=a[ya:yb, xa:xb].copy(),
                             ox=x - xa, oy=y - ya, w=w, h=h))
yr_n = {y: sum(o["r"]["year"] == y for o in src_objs) for y in TRAIN}
wts = np.array([1 / yr_n[o["r"]["year"]] for o in src_objs])
wts /= wts.sum()
dests = [r for r in train]
d_w = np.array([3.0 if not boxes(r) else 1.0 for r in dests])
d_w /= d_w.sum()

for j in range(N_PASTE):
    o = src_objs[rng.choice(len(src_objs), p=wts)]
    d = dests[rng.choice(len(dests), p=d_w)]
    a = get(d).copy()
    nd = nadir(a)
    side = max(o["w"], o["h"])
    s = float(np.clip(rng.uniform(*TARGET) / side, 0.4, 1.0))
    ph, pw = o["patch"].shape
    nh, nw = max(4, int(ph * s)), max(4, int(pw * s))
    p = np.asarray(Image.fromarray(o["patch"].astype(np.uint8)).resize((nw, nh), Image.BICUBIC), np.float32)
    ox, oy, w, h = o["ox"] * s, o["oy"] * s, o["w"] * s, o["h"] * s
    if rng.random() < 0.5:                                    # along-track flip keeps the physics
        p = p[::-1]
        oy = nh - oy
    bx = boxes(d)
    ok = False
    for _ in range(30):                                       # same side of nadir, clear of existing boxes
        cx = nd + o["out"] * rng.uniform(0.08, 0.42) * 1024
        cy = rng.uniform(0.06, 0.94) * 1024
        xa, ya = int(cx - ox), int(cy - oy)
        if xa < 0 or ya < 0 or xa + nw > 1024 or ya + nh > 1024:
            continue
        if np.median(a[ya:ya + nh, xa:xa + nw]) < max(12, 0.5 * np.median(a)):   # seabed, not water column
            continue
        if all(abs(cx - bx_) > (w + bw) and abs(cy - by_) > (h + bh) for _, bx_, by_, bw, bh in bx):
            ok = True
            break
    if not ok:
        continue
    dst = a[ya:ya + nh, xa:xa + nw]
    # multiplicative transfer: object/shadow contrast relative to the source seabed level is
    # imposed on the destination texture (highlight > 1, shadow < 1), so no source texture box remains
    ring = np.concatenate([p[:2].ravel(), p[-2:].ravel(), p[:, :2].ravel(), p[:, -2:].ravel()])
    ratio = np.clip((p + 1) / (np.median(ring) + 1), 0.05, 8.0)
    yy, xx = np.mgrid[0:nh, 0:nw].astype(np.float32)
    rr = np.sqrt(((xx - (nw - 1) / 2) / (nw / 2)) ** 2 + ((yy - (nh - 1) / 2) / (nh / 2)) ** 2)
    m = np.clip((1.0 - rr) / 0.35, 0, 1)                      # elliptical soft mask
    a[ya:ya + nh, xa:xa + nw] = dst * ratio ** m
    bx = bx + [(o["c"], cx, cy, w, h)]
    if write_tile(a, bx, cx, cy, f"paste_{j:04d}_{d['year']}"):
        y = o["r"]["year"]
        stats["paste"][y] = stats["paste"].get(y, 0) + 1
        stats["size_after"].append(max(w, h))

sz = np.array(stats["size_after"])
stats["size_after"] = dict(n=int(len(sz)), median=float(np.median(sz)), q25=float(np.percentile(sz, 25)),
                           q75=float(np.percentile(sz, 75)))
stats["train_objects_by_year"] = {y: sum(len(boxes(r)) for r in train if r["year"] == y) for y in TRAIN}
json.dump(stats, open("prep_aug.json", "w"), indent=1)
ab = os.path.abspath("tiles")
open("tiles/milco_tiles_aug.yaml", "w").write(
    f"path: {ab}\ntrain: [train/images, train_aug/images]\nval: val/images\nnames:\n  0: MILCO\n  1: NOMBO\n")
print(json.dumps(stats))
