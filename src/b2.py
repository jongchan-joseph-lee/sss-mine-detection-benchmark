"""
B2: image-domain processing for side-scan sonar (MILCO) and a classical matched-filter detector.

  1. pseudo-colour -> intensity (luminance; PCA shows a 1-D colour map)
  2. nadir (water-column) localisation and across-track gain normalisation (per-column median)
  3. highlight-shadow signature of MILCO / NOMBO / background along the outward across-track axis
  4. classical detector: 2-D matched filter "highlight then shadow, away from nadir", multi-scale,
     NMS, FROC (recall vs false alarms per image) on unique scenes of each year split
"""
import json

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, maximum_filter, median_filter, uniform_filter

S = json.load(open("screen.json"))
YEARS_TRAIN, YEARS_OP = ["2015", "2017", "2021"], ["2010", "2018"]
rng = np.random.default_rng(0)

# one representative image per scene, prefer the 1024 px frame
rep = {}
for r in S["records"]:
    k = (r["year"], r["scene"])
    if k not in rep or r["size"] > rep[k]["size"]:
        rep[k] = r
scenes = sorted(rep.values(), key=lambda r: (r["year"], r["file"]))


def load(r):
    a = np.asarray(Image.open(f"raw/{r['year']}/{r['year']}/{r['file']}").convert("L"), float)
    return a


def nadir_band(a):
    W = a.shape[1]
    col = a.mean(0)
    c0, c1 = int(W * 0.3), int(W * 0.7)
    dark = col[c0:c1] < np.percentile(col, 15) + 5
    idx = np.where(dark)[0] + c0
    if len(idx) == 0:
        m = c0 + int(np.argmin(col[c0:c1]))
        return m, m - W // 40, m + W // 40
    return int(idx.mean()), int(idx.min()), int(idx.max())


def normalise(a):
    """across-track gain normalisation: divide by per-column median (range/beam-pattern TVG proxy)"""
    med = np.median(a, axis=0) + 4.0
    return a / med[None, :]


# ---------------------------------------------------------------- 3. signature profiles
def profile(n, cx, cy, w, h, nad, L=4.0, npts=81):
    """mean intensity along outward across-track axis through the box centre, in units of box width"""
    H, W = n.shape
    out_sign = 1 if cx > nad else -1
    u = np.linspace(-L, L, npts)
    xs = np.clip((cx + out_sign * u * w).round().astype(int), 0, W - 1)
    y0, y1 = int(max(0, cy - h / 2)), int(min(H, cy + h / 2 + 1))
    return n[y0:y1, :][:, xs].mean(0)


prof = {"MILCO": [], "NOMBO": [], "BG": []}
for r in scenes:
    a = load(r)
    n = normalise(a)
    W = a.shape[1]
    nad, n0, n1 = nadir_band(a)
    for c, x, y, w, h in r["labels"]:
        if w * W < 3 or h * W < 3:
            continue
        prof["MILCO" if c == 0 else "NOMBO"].append(profile(n, x * W, y * W, w * W, h * W, nad))
    if r["labels"] and len(prof["BG"]) < 1500:
        for _ in range(4):
            side = rng.choice([-1, 1])
            cx = nad + side * rng.uniform(0.08, 0.45) * W
            cy = rng.uniform(0.05, 0.95) * W
            ww = np.median([l[3] for l in r["labels"]]) * W
            hh = np.median([l[4] for l in r["labels"]]) * W
            prof["BG"].append(profile(n, cx, cy, ww, hh, nad))
sig = {k: dict(mean=np.mean(v, 0).round(4).tolist(), q25=np.percentile(v, 25, 0).round(4).tolist(),
               q75=np.percentile(v, 75, 0).round(4).tolist(), n=len(v)) for k, v in prof.items()}
print({k: v["n"] for k, v in sig.items()})


# ---------------------------------------------------------------- 4. matched-filter detector
def mf_kernel(s):
    """highlight (+) of width s, then shadow (-) of width 2s on the outward side; height s.
    zero-mean, unit-energy. Returned for the starboard (outward = +x) direction."""
    k = np.zeros((s, 4 * s))
    k[:, s:2 * s] = 1.0
    k[:, 2 * s:4 * s] = -0.5
    k -= k.mean()
    return k / np.linalg.norm(k)


def detect(a, scale_px=(6, 10, 16), k_top=40):
    H, W = a.shape
    f = W / 1024.0
    n = np.log(normalise(median_filter(a, 3)) + 1e-3)
    n = (n - n.mean()) / (n.std() + 1e-9)
    nad, n0, n1 = nadir_band(a)
    best = np.full_like(n, -np.inf)
    from scipy.signal import fftconvolve
    for s in scale_px:
        s = max(2, int(round(s * f)))
        k = mf_kernel(s)
        right = fftconvolve(n, k[::-1, ::-1], mode="same")              # starboard: outward = +x
        left = fftconvolve(n, k[::-1, :], mode="same")                   # port: outward = -x (mirrored)
        sc = np.where(np.arange(W)[None, :] > nad, right, left)
        best = np.maximum(best, sc)
    margin = int(0.02 * W)
    best[:, max(0, n0 - margin):min(W, n1 + margin)] = -np.inf          # water column
    best[:margin, :] = best[-margin:, :] = -np.inf
    best[:, :margin] = best[:, -margin:] = -np.inf
    pk = (best == maximum_filter(best, size=int(24 * f) | 1)) & np.isfinite(best)
    ys, xs = np.where(pk)
    sc = best[ys, xs]
    o = np.argsort(-sc)[:k_top]
    return [(float(xs[i] / W), float(ys[i] / H), float(sc[i])) for i in o]


def match(dets, labels, tol=1.0):
    """greedy matching of point detections to GT boxes: hit if the point is within tol*max(w,h) of centre"""
    gts = [l for l in labels if l[3] > 0 and l[4] > 0]
    used = set()
    out = []
    for x, y, s in sorted(dets, key=lambda d: -d[2]):
        hit = None
        for gi, (c, gx, gy, gw, gh) in enumerate(gts):
            if gi in used:
                continue
            if np.hypot(x - gx, y - gy) <= tol * max(gw, gh, 0.012):
                hit = gi
                break
        if hit is not None:
            used.add(hit)
            out.append((s, 1, int(gts[hit][0])))
        else:
            out.append((s, 0, -1))
    return out, len(gts), [int(g[0]) for g in gts]


def froc(rows, n_gt, n_img, fa_grid=(0.1, 0.25, 0.5, 1, 2, 4, 8)):
    rows = sorted(rows, key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in rows])
    fp = np.cumsum([1 - r[1] for r in rows])
    rec = tp / max(n_gt, 1)
    fa = fp / n_img
    curve = [(float(f), float(rec[fa <= f].max()) if (fa <= f).any() else 0.0) for f in fa_grid]
    return curve, dict(fa=fa[::max(1, len(fa) // 300)].round(4).tolist(), rec=rec[::max(1, len(rec) // 300)].round(4).tolist())


det_res = {}
per_scene = []
for split, years in [("train_years", YEARS_TRAIN), ("operational", YEARS_OP)]:
    rows, n_gt, n_img = [], 0, 0
    rows_cls = {0: [], 1: []}
    for r in [s for s in scenes if s["year"] in years]:
        a = load(r)
        d = detect(a)
        m, g, gcls = match(d, r["labels"])
        rows += m
        n_gt += g
        n_img += 1
        per_scene.append(dict(year=r["year"], file=r["file"], n_gt=g, dets=d[:10]))
    curve, raw = froc(rows, n_gt, n_img)
    det_res[split] = dict(n_img=n_img, n_gt=n_gt, froc=curve, raw=raw)
    print(split, n_img, n_gt, curve)

json.dump(dict(signature=sig, mf=det_res, per_scene=per_scene), open("b2_results.json", "w"))
print("saved")
