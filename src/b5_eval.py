"""
B5: test-year evaluation for the augmentation / architecture comparison.

For each trained detector: point FROC on the 578 test frames (2010, 2018) with the same matching rule
as b2/b3, full-frame (1024 px) and sliced (320-px tiles, stride 256) inference, Pd by object size at the
4 FA/frame operating point, and frame-bootstrap 95% CIs incl. paired differences against the reference.

  python b5_eval.py ref=runs/detect/runs/yolo_tiles/weights/best.pt n_aug40=runs/detect/runs/exp/n_aug40/weights/best.pt ...
Detections are cached in b5_dets/<name>_<mode>.json so the bootstrap can be re-run without inference.
"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

src = open("b2.py").read()
exec(src[src.index("def match("):src.index("det_res = {}")])          # match(), froc()
FILES = sorted(glob.glob("tiles/test_full/images/*.png"))
GRID = (0.25, 0.5, 1, 2, 4, 8)
BANDS = [(0, 30), (30, 50), (50, 1e9)]
os.makedirs("b5_dets", exist_ok=True)
DEV = os.environ.get("B5_DEVICE", "cpu")               # "0" on a GPU runtime


def labels(f):
    lf = f.replace("/images/", "/labels/")[:-4] + ".txt"
    if not os.path.exists(lf):
        return []
    return [[int(p.split()[0])] + [float(x) for x in p.split()[1:]] for p in open(lf).read().split("\n") if p.strip()]


def nms_pts(d, r=0.02):
    keep = []
    for x, y, c in sorted(d, key=lambda t: -t[2]):
        if all(np.hypot(x - a, y - b) > r for a, b, _ in keep):
            keep.append((x, y, c))
    return keep


def detect_all(weights, mode):
    from ultralytics import RTDETR, YOLO
    m = (RTDETR if "rtdetr" in weights else YOLO)(weights)
    out = []
    for f in FILES:
        if mode == "full":
            b = m.predict(f, imgsz=1024, conf=0.01, device=DEV, verbose=False)[0].boxes
            d = [(float(x), float(y), float(c)) for (x, y, _, _), c in zip(b.xywhn.cpu().numpy(), b.conf.cpu().numpy())]
        else:
            im = np.asarray(Image.open(f).convert("RGB"))
            T, S = 320, 256
            xs = list(range(0, 1024 - T + 1, S))
            xs = xs if xs[-1] == 1024 - T else xs + [1024 - T]
            tiles = [(x0, y0, im[y0:y0 + T, x0:x0 + T]) for y0 in xs for x0 in xs]
            res = m.predict([t[2] for t in tiles], imgsz=320, conf=0.01, device=DEV, verbose=False)
            d = []
            for (x0, y0, _), r in zip(tiles, res):
                for (x, y, _, _), c in zip(r.boxes.xywh.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    d.append(((x0 + float(x)) / 1024, (y0 + float(y)) / 1024, float(c)))
            d = nms_pts(d)
        out.append(d)
    return out


def per_image(dets):
    """rows per image + per-GT best score (for size analysis)"""
    rows, gts = [], []
    for f, d in zip(FILES, dets):
        lab = labels(f)
        mm, g, _ = match(d, lab)
        rows.append((mm, g))
        for c, gx, gy, gw, gh in lab:
            r = max(gw, gh, 0.012)
            s = [dc for dx, dy, dc in d if np.hypot(dx - gx, dy - gy) <= r]
            gts.append(dict(year=os.path.basename(f)[:4], cls=c, size=max(gw, gh) * 1024, best=max(s) if s else 0.0))
    return rows, gts


def pd_at(rows, idx, grid=GRID):
    allr = [r for i in idx for r in rows[i][0]]
    ng = sum(rows[i][1] for i in idx)
    c, _ = froc(allr, ng, len(idx), grid)
    return np.array([p for _, p in c])


def thr_at(rows, fa=4.0):
    sc = sorted([r for mm, _ in rows for r in mm], key=lambda r: -r[0])
    fp = np.cumsum([1 - r[1] for r in sc]) / len(rows)
    ok = np.where(fp <= fa)[0]
    return sc[ok[-1]][0] if len(ok) else 1.0


def main():
    specs = [a.split("=", 1) for a in sys.argv[1:]]
    rng = np.random.default_rng(0)
    B = 2000
    boot_idx = [rng.integers(0, len(FILES), len(FILES)) for _ in range(B)]
    res, rows_all = {}, {}
    for name, w in specs:
        for mode in ("full", "sliced"):
            key = f"{name}_{mode}"
            cache = f"b5_dets/{key}.json"
            if os.path.exists(cache):
                dets = json.load(open(cache))
            else:
                dets = detect_all(w, mode)
                json.dump(dets, open(cache, "w"))
            rows, gts = per_image(dets)
            rows_all[key] = rows
            pd = pd_at(rows, range(len(FILES)))
            bs = np.array([pd_at(rows, ix) for ix in boot_idx])
            thr = thr_at(rows)
            by_size = {}
            for lo, hi in BANDS:
                g = [x for x in gts if lo <= x["size"] < hi]
                by_size[f"{lo}-{hi if hi < 1e9 else 'inf'}"] = dict(n=len(g), pd=float(np.mean([x["best"] >= thr for x in g])) if g else None)
            by_year = {y: float(np.mean([x["best"] >= thr for x in gts if x["year"] == y])) for y in ("2010", "2018")}
            res[key] = dict(weights=w, mode=mode, fa_grid=list(GRID), pd=pd.round(4).tolist(),
                            ci95=[np.percentile(bs[:, k], [2.5, 97.5]).round(4).tolist() for k in range(len(GRID))],
                            thr_at_4fa=float(thr), pd_by_size_at_4fa=by_size, pd_by_year_at_4fa=by_year)
            print(key, dict(zip(GRID, pd.round(3))), by_size, flush=True)
    ref = "ref_full"
    if ref in rows_all:
        for key in rows_all:
            if key == ref:
                continue
            d = np.array([pd_at(rows_all[key], ix) - pd_at(rows_all[ref], ix) for ix in boot_idx[:1000]])
            res[key]["diff_vs_ref_full"] = dict(mean=(np.array(res[key]["pd"]) - np.array(res[ref]["pd"])).round(4).tolist(),
                                                ci95=[np.percentile(d[:, k], [2.5, 97.5]).round(4).tolist() for k in range(len(GRID))])
    out = os.environ.get("B5_OUT", "b5_eval.json")
    json.dump(res, open(out, "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
