"""
B4: detection -> classification cascade.
  1. YOLOv8n candidates on full 1024-px frames (weights from train_yolo.py)
  2. each candidate re-scored by the highlight-shadow (acoustic) patch classifier: MILCO / NOMBO / background
  3. candidates judged as background are dropped

Thresholds are fixed on the held-out training-year scenes (tiles/val_full, never used for YOLO training)
and applied once to the test years (2010, 2018).
  - detector threshold: highest recall on val scenes at <= 4 false alarms per frame
  - classifier threshold: highest p(object) cut that keeps >= 95 % of val true detections

Classifier training data (training-year scenes, val scenes excluded)
  A: ground-truth objects + random seabed patches (same as b3_patch.py)
  B: A + YOLO false alarms on the training scenes as extra background (hard negatives)

Run from src/ after prep_tiles.py:   python b4_cascade.py   ->  b4_cascade.json
"""
import glob
import json
import os
import random

import numpy as np
from PIL import Image
from scipy.ndimage import median_filter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO

random.seed(0)
np.random.seed(0)
src = open("b2.py").read()
exec(src[src.index("def match("):src.index("det_res = {}")])            # match(), froc()
src = open("b3_patch.py").read()
P = 112
exec(src[src.index("def nadir("):src.index("# ---------------------------------------------------------------- build patches")])

S = json.load(open("screen.json"))
TRAIN, OP = ["2015", "2017", "2021"], ["2010", "2018"]
FA_DET, KEEP_TP = 4.0, 0.95

# ---- same scene selection and val split as prep_tiles.py (random.seed(0) -> one shuffle)
rep = {}
for r in S["records"]:
    k = (r["year"], r["scene"])
    if k not in rep or r["size"] > rep[k]["size"]:
        rep[k] = r
scenes = sorted(rep.values(), key=lambda r: (r["year"], r["file"]))
tr = [r for r in scenes if r["year"] in TRAIN]
random.shuffle(tr)
val_ids = {(r["year"], r["file"]) for r in tr[: int(0.15 * len(tr))]}
train_sc = [r for r in scenes if r["year"] in TRAIN and (r["year"], r["file"]) not in val_ids]
val_sc = [r for r in scenes if (r["year"], r["file"]) in val_ids]
test_sc = [r for r in scenes if r["year"] in OP]
random.seed(1)


def labels(r):
    return [[c, x, y, w, h] for c, x, y, w, h in r["labels"] if w * 1024 >= 4 and h * 1024 >= 4]


m = YOLO("runs/detect/runs/yolo_tiles/weights/best.pt")


def detect(r):
    a = load1024(r)
    res = m.predict(np.stack([a.astype(np.uint8)] * 3, -1), imgsz=1024, conf=0.01, device="cpu", verbose=False)[0]
    b = res.boxes
    return a, [(float(x), float(y), float(w), float(h), float(c), int(k))
               for (x, y, w, h), c, k in zip(b.xywhn.numpy(), b.conf.numpy(), b.cls.numpy())]


def feats(a, nad, x, y, w, h):
    return acoustic_feat(a, x * 1024, y * 1024, max(w * 1024, 4), max(h * 1024, 4), nad)


def run_split(sc, keep_frames=False):
    out = []
    for r in sc:
        a, d = detect(r)
        af = median_filter(a, 3)
        nad = nadir(af)[0]
        lab = labels(r)
        mm, _, _ = match([(x, y, c) for x, y, w, h, c, k in d], lab)
        # match() sorts by score; map back by score order
        order = sorted(range(len(d)), key=lambda i: -d[i][4])
        hit = np.zeros(len(d), int); gcls = -np.ones(len(d), int)
        for i, (s, t, g) in zip(order, mm):
            hit[i], gcls[i] = t, g
        F = np.array([feats(af, nad, x, y, w, h) for x, y, w, h, c, k in d], np.float32) if d else np.zeros((0, 64), np.float32)
        out.append(dict(r=r, d=d, hit=hit, gcls=gcls, F=F, n_gt=len(lab), af=af if keep_frames else None, nad=nad))
    return out


print("detecting val / test / train scenes ...", flush=True)
VAL, TEST, TRN = run_split(val_sc), run_split(test_sc), run_split(train_sc, keep_frames=True)

# ---- sanity: detector-alone FROC on test should match b3_yolo.json
rows = [(d[4], h, g) for s in TEST for d, h, g in zip(s["d"], s["hit"], s["gcls"])]
curve, _ = froc(rows, sum(s["n_gt"] for s in TEST), len(TEST))
print("test FROC (detector alone)", curve, flush=True)

# ---- classifier training sets
XA, yA, XB, yB = [], [], [], []
sizes = [max(l[3], l[4]) * 1024 for s in TRN for l in labels(s["r"])]
for s in TRN:
    af, nad = s["af"], s["nad"]
    _, n0, n1 = nadir(af)
    bx = labels(s["r"])
    for c, x, y, w, h in bx:
        XA.append(feats(af, nad, x, y, w, h)); yA.append(c)
    for _ in range(2):
        for _try in range(30):
            side = max(2 * random.choice(sizes), 48)
            opts = [(0.04 * 1024, n0 - side), (n1 + side, 0.96 * 1024)]
            opts = [o for o in opts if o[1] > o[0]] or [(side, 1024 - side)]
            lo_, hi_ = random.choice(opts)
            cx, cy = random.uniform(lo_, hi_), random.uniform(side, 1024 - side)
            if all(abs(cx - b[1] * 1024) > side and abs(cy - b[2] * 1024) > side for b in bx):
                break
        XA.append(acoustic_feat(af, cx, cy, side / 2, side / 2, nad)); yA.append(2)
    for (x, y, w, h, c, k), hit, f in zip(s["d"], s["hit"], s["F"]):
        if not hit and c >= 0.05:
            XB.append(f); yB.append(2)
XB, yB = XA + XB, yA + yB


def fit(X, y):
    X = np.nan_to_num(np.array(X, np.float32), nan=0.0, posinf=0.0, neginf=-30.0)
    sc = StandardScaler().fit(X)
    return sc, LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced").fit(sc.transform(X), np.array(y))


def p_obj(model, F):
    sc, lr = model
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=-30.0)
    p = lr.predict_proba(sc.transform(F)) if len(F) else np.zeros((0, 3))
    return 1 - p[:, list(lr.classes_).index(2)], p


def pool(split, thr_d):
    hit, conf, F, cls, gcls = [], [], [], [], []
    for s in split:
        for (x, y, w, h, c, k), t, g, f in zip(s["d"], s["hit"], s["gcls"], s["F"]):
            if c >= thr_d:
                hit.append(t); conf.append(c); F.append(f); cls.append(k); gcls.append(g)
    return np.array(hit), np.array(conf), (np.array(F, np.float32) if F else np.zeros((0, 64), np.float32)), np.array(cls), np.array(gcls)


def op(split, thr):
    n_img, n_gt = len(split), sum(s["n_gt"] for s in split)
    rows = [(d[4], h) for s in split for d, h in zip(s["d"], s["hit"]) if d[4] >= thr]
    tp = sum(h for _, h in rows)
    return tp / n_gt, (len(rows) - tp) / n_img


# ---- detector threshold on val
cands = sorted({round(d[4], 4) for s in VAL for d in s["d"]})
best = None
for t in cands:
    rec, fa = op(VAL, t)
    if fa <= FA_DET and (best is None or rec > best[1] or (rec == best[1] and t > best[0])):
        best = (t, rec, fa)
thr_d = best[0]
out = {"n_val_frames": len(VAL), "n_val_gt": sum(s["n_gt"] for s in VAL), "n_test_frames": len(TEST),
       "n_test_gt": sum(s["n_gt"] for s in TEST), "detector_threshold": thr_d,
       "val_detector": {"recall": best[1], "fa_per_frame": best[2]}, "test_froc_detector_alone": curve}
rec, fa = op(TEST, thr_d)
out["test_detector_alone"] = {"recall": rec, "fa_per_frame": fa}

for name, (X, y) in {"A_random_background": (XA, yA), "B_plus_hard_negatives": (XB, yB)}.items():
    model = fit(X, y)
    vh, vc, vF, _, _ = pool(VAL, thr_d)
    vp, _ = p_obj(model, vF)
    tp_scores = np.sort(vp[vh == 1])
    thr_c = float(tp_scores[int(np.floor((1 - KEEP_TP) * len(tp_scores)))]) if len(tp_scores) else 0.0
    th, tc, tF, tcls, tg = pool(TEST, thr_d)
    tp_, pr = p_obj(model, tF)
    keep = tp_ >= thr_c
    n_gt = out["n_test_gt"]
    # class agreement on true detections: YOLO class vs classifier class (MILCO=0, NOMBO=1)
    m_tp = th == 1
    clf_cls = pr[:, :2].argmax(1)
    def bacc(pred, true):
        return float(np.mean([np.mean(pred[true == c] == c) for c in (0, 1) if (true == c).any()]))
    out[name] = {"n_train": len(y), "n_train_hard_neg": int(len(y) - len(yA)) if name.startswith("B") else 0,
                 "classifier_threshold": thr_c,
                 "val_true_detections_kept": float(np.mean(vp[vh == 1] >= thr_c)) if (vh == 1).any() else None,
                 "val_false_alarms_removed": float(np.mean(vp[vh == 0] < thr_c)) if (vh == 0).any() else None,
                 "test_recall": float(th[keep].sum() / n_gt), "test_fa_per_frame": float((keep & (th == 0)).sum() / len(TEST)),
                 "test_false_alarms_removed": float(np.mean(~keep[th == 0])) if (th == 0).any() else None,
                 "test_true_detections_lost": float(np.mean(~keep[th == 1])) if (th == 1).any() else None,
                 "class_ba_yolo_on_true_detections": bacc(tcls[m_tp], tg[m_tp]),
                 "class_ba_classifier_on_true_detections": bacc(clf_cls[m_tp], tg[m_tp])}
# ---- second operating point: detector threshold set by a recall target on val (>= 90 %), then the same cascade
# (added after the first result showed 53 % test recall at the false-alarm-capped point)
best90 = max([t for t in cands if op(VAL, t)[0] >= 0.90] or [min(cands)])
out["detector_threshold_recall90"] = best90
r_, f_ = op(VAL, best90); out["val_detector_recall90"] = {"recall": r_, "fa_per_frame": f_}
r_, f_ = op(TEST, best90); out["test_detector_alone_recall90"] = {"recall": r_, "fa_per_frame": f_}
modelA = fit(XA, yA)
vh, vc, vF, _, _ = pool(VAL, best90); vp, _ = p_obj(modelA, vF)
ts = np.sort(vp[vh == 1]); thr_c90 = float(ts[int(np.floor((1 - KEEP_TP) * len(ts)))]) if len(ts) else 0.0
th, tc, tF, _, _ = pool(TEST, best90); tp_, _ = p_obj(modelA, tF); keep = tp_ >= thr_c90
out["A_random_background_recall90"] = {"classifier_threshold": thr_c90,
    "test_recall": float(th[keep].sum() / out["n_test_gt"]), "test_fa_per_frame": float((keep & (th == 0)).sum() / len(TEST)),
    "test_false_alarms_removed": float(np.mean(~keep[th == 0])), "test_true_detections_lost": float(np.mean(~keep[th == 1]))}
# ---- FROC of the cascade on test (classifier threshold fixed from val at the recall-target point)
th, tc, tF, _, _ = pool(TEST, 0.0); tp_, _ = p_obj(modelA, tF); keep = tp_ >= thr_c90
c_alone, _ = froc(list(zip(tc, th, [0] * len(th))), out["n_test_gt"], len(TEST))
c_casc, _ = froc(list(zip(tc[keep], th[keep], [0] * int(keep.sum()))), out["n_test_gt"], len(TEST))
out["test_froc_alone"], out["test_froc_cascade_A"] = c_alone, c_casc
# ---- Pdc: detected AND object type (MILCO / NOMBO) correct, as a function of false alarms per frame
th, tc, tF, tcls, tg = pool(TEST, 0.0); tp_, pr = p_obj(modelA, tF); keep = tp_ >= thr_c90
clf_cls = pr[:, :2].argmax(1)


def pdc_curve(score, hit, correct, n_gt, n_img, grid=(0.1, 0.25, 0.5, 1, 2, 4, 8)):
    o = np.argsort(-score); hit, correct = hit[o], correct[o]
    fa = np.cumsum(1 - hit) / n_img; pdc = np.cumsum(hit * correct) / n_gt
    return [(float(f), float(pdc[fa <= f].max()) if (fa <= f).any() else 0.0) for f in grid]


out["test_pdc_alone"] = pdc_curve(tc, th, (tcls == tg).astype(int), out["n_test_gt"], len(TEST))
out["test_pdc_cascade_A"] = pdc_curve(tc[keep], th[keep], (clf_cls[keep] == tg[keep]).astype(int), out["n_test_gt"], len(TEST))
json.dump(out, open("b4_cascade.json", "w"), indent=2)
print(json.dumps(out, indent=2))
