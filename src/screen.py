"""
B1 screening of the MILCO side-scan sonar dataset (Santos et al. 2024, Data in Brief 53:110132).

Checks
  1. near-duplicate scenes: same frame at 1024 px and 416 px, incl. flipped copies (Roboflow export)
     -> group images into 'scene' ids (connected components, corr > THR under 8 dihedral transforms)
  2. label statistics per year: images, empty images, objects per class, box size in metres-agnostic
     relative units and in pixels of the 1024 frame, objects per image
  3. colour map: images are pseudo-coloured; check how much information is in luminance
"""
import glob
import json

import numpy as np
from PIL import Image

THR = 0.95
YEARS = ["2010", "2015", "2017", "2018", "2021"]
CLS = {0: "MILCO", 1: "NOMBO"}


from scipy.ndimage import gaussian_filter


def small(f):
    a = np.asarray(Image.open(f).convert("L").resize((160, 160), Image.BILINEAR), float)
    return a - gaussian_filter(a, 3)          # high-pass: keep seabed texture, drop common layout


def z(a):
    a = a - a.mean()
    return a / (np.linalg.norm(a) + 1e-9)


def dihedral(a, k):
    b = np.rot90(a, k % 4)
    return np.fliplr(b) if k >= 4 else b


records = []
for y in YEARS:
    fs = sorted(glob.glob(f"raw/{y}/{y}/*.jpg"))
    for f in fs:
        im = Image.open(f)
        lab = []
        try:
            for line in open(f[:-4] + ".txt").read().strip().splitlines():
                p = line.split()
                if len(p) == 5:
                    lab.append([int(p[0])] + [float(v) for v in p[1:]])
        except FileNotFoundError:
            pass
        records.append(dict(year=y, file=f.split("/")[-1], path=f, size=im.size[0], labels=lab))

# ---- near-duplicate grouping within each year (and across, to be safe)
feats = np.array([z(small(r["path"])).ravel() for r in records])
T = {k: np.array([z(dihedral(small(r["path"]), k)).ravel() for r in records]) for k in range(1, 8)}
n = len(records)
parent = list(range(n))


def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


pair_tf = {}
for k in range(8):
    G = feats @ (feats if k == 0 else T[k]).T
    ii, jj = np.where(np.triu(G, 1 if k == 0 else 0) > THR) if k == 0 else np.where(G > THR)
    for i, j in zip(ii, jj):
        if i == j:
            continue
        a, b = find(i), find(j)
        if a != b:
            parent[a] = b
        pair_tf.setdefault(k, 0)
        pair_tf[k] += 1
scene = [find(i) for i in range(n)]
uniq = {s: idx for idx, s in enumerate(sorted(set(scene)))}
for r, s in zip(records, scene):
    r["scene"] = uniq[s]

# ---- statistics
out = {"years": {}, "thr": THR}
for y in YEARS:
    R = [r for r in records if r["year"] == y]
    scenes = {r["scene"] for r in R}
    cross = {r["scene"] for r in records if r["year"] != y} & scenes
    objs = [(l[0], l[3], l[4], r["size"]) for r in R for l in r["labels"]]
    # unique objects: count labels only from one representative per scene (prefer 1024)
    rep = {}
    for r in R:
        if r["scene"] not in rep or r["size"] > rep[r["scene"]]["size"]:
            rep[r["scene"]] = r
    uobjs = [l for r in rep.values() for l in r["labels"]]
    out["years"][y] = dict(
        n_images=len(R), n_1024=sum(r["size"] == 1024 for r in R), n_416=sum(r["size"] == 416 for r in R),
        n_scenes=len(scenes), n_scenes_shared_other_year=len(cross),
        n_empty_images=sum(len(r["labels"]) == 0 for r in R),
        n_empty_scenes=sum(len(r["labels"]) == 0 for r in rep.values()),
        obj_all={CLS[c]: sum(o[0] == c for o in objs) for c in CLS},
        obj_unique={CLS[c]: sum(l[0] == c for l in uobjs) for c in CLS},
        box_w_rel={CLS[c]: [round(l[3], 4) for l in uobjs if l[0] == c] for c in CLS},
        box_h_rel={CLS[c]: [round(l[4], 4) for l in uobjs if l[0] == c] for c in CLS},
        objs_per_scene=[len(r["labels"]) for r in rep.values()],
    )
    s = out["years"][y]
    print(y, {k: v for k, v in s.items() if not k.startswith(("box_", "objs_per"))})

out["pair_transforms"] = {str(k): int(v) for k, v in pair_tf.items()}
out["records"] = [{k: r[k] for k in ("year", "file", "size", "scene", "labels")} for r in records]
json.dump(out, open("screen.json", "w"))
print("transforms among duplicate pairs:", pair_tf)
print("total images", n, "scenes", len(uniq))
