"""
Scale-normalised frames + tiles for small-object detection.
  - one image per scene (duplicates removed), grey intensity (pseudo-colour removed)
  - 416 px frames upsampled to 1024 px so every object is on the same pixel scale
  - training tiles 320 px: object-centred (random offset) + background tiles
  - test = operational years (2010, 2018) as full 1024 px frames
"""
import json
import os
import random

import numpy as np
from PIL import Image

random.seed(0)
S = json.load(open("screen.json"))
TRAIN, OP = ["2015", "2017", "2021"], ["2010", "2018"]
T = 320
rep = {}
for r in S["records"]:
    k = (r["year"], r["scene"])
    if k not in rep or r["size"] > rep[k]["size"]:
        rep[k] = r
scenes = sorted(rep.values(), key=lambda r: (r["year"], r["file"]))
tr = [r for r in scenes if r["year"] in TRAIN]
random.shuffle(tr)
val_ids = {(r["year"], r["file"]) for r in tr[: int(0.15 * len(tr))]}


def frame(r):
    im = Image.open(f"raw/{r['year']}/{r['year']}/{r['file']}").convert("L")
    if im.size[0] != 1024:
        im = im.resize((1024, 1024), Image.BICUBIC)
    return im


def boxes(r):
    return [(c, x * 1024, y * 1024, w * 1024, h * 1024) for c, x, y, w, h in r["labels"]
            if w * 1024 >= 4 and h * 1024 >= 4]


def write(img, bx, d, name):
    os.makedirs(f"{d}/images", exist_ok=True)
    os.makedirs(f"{d}/labels", exist_ok=True)
    img.convert("RGB").save(f"{d}/images/{name}.png")
    W, H = img.size
    with open(f"{d}/labels/{name}.txt", "w") as f:
        for c, x, y, w, h in bx:
            f.write(f"{c} {x / W:.6f} {y / H:.6f} {w / W:.6f} {h / H:.6f}\n")


def crop_boxes(bx, x0, y0):
    out = []
    for c, x, y, w, h in bx:
        a0, a1 = max(x - w / 2, x0), min(x + w / 2, x0 + T)
        b0, b1 = max(y - h / 2, y0), min(y + h / 2, y0 + T)
        if a1 - a0 >= 0.6 * w and b1 - b0 >= 0.6 * h:          # keep if mostly inside
            out.append((c, (a0 + a1) / 2 - x0, (b0 + b1) / 2 - y0, a1 - a0, b1 - b0))
    return out


n = {"train": 0, "val": 0, "test": 0}
for r in scenes:
    im = frame(r)
    bx = boxes(r)
    key = (r["year"], r["file"])
    stem = f"{r['year']}_{r['file'][:-4]}"
    if r["year"] in OP:
        write(im, bx, "tiles/test_full", stem)
        n["test"] += 1
        continue
    split = "val" if key in val_ids else "train"
    reps = 3 if split == "train" else 1
    k = 0
    for c, x, y, w, h in bx:                                     # object-centred tiles
        for _ in range(reps):
            x0 = int(np.clip(x - random.uniform(0.2, 0.8) * T, 0, 1024 - T))
            y0 = int(np.clip(y - random.uniform(0.2, 0.8) * T, 0, 1024 - T))
            write(im.crop((x0, y0, x0 + T, y0 + T)), crop_boxes(bx, x0, y0), f"tiles/{split}", f"{stem}_o{k}")
            k += 1
            n[split] += 1
    for j in range(2 if bx else 1):                              # background tiles
        x0, y0 = random.randint(0, 1024 - T), random.randint(0, 1024 - T)
        write(im.crop((x0, y0, x0 + T, y0 + T)), crop_boxes(bx, x0, y0), f"tiles/{split}", f"{stem}_b{j}")
        n[split] += 1
    if split == "val":
        write(im, bx, "tiles/val_full", stem)

ab = os.path.abspath("tiles")
open("tiles/milco_tiles.yaml", "w").write(
    f"path: {ab}\ntrain: train/images\nval: val/images\nnames:\n  0: MILCO\n  1: NOMBO\n")
open("tiles/milco_test.yaml", "w").write(
    f"path: {ab}\ntrain: train/images\nval: test_full/images\nnames:\n  0: MILCO\n  1: NOMBO\n")
print(n)
