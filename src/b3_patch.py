"""
B3-2: 3-class patch classification (MILCO / NOMBO / background), cf. Mine-JEPA protocol.

Patches: square crop around each box (side = max(2*max(w,h), 48) px on the 1024-px scale),
oriented so that the outward (shadow) direction points right, resized to 112 px.
Background: 2 random seabed crops per scene, away from boxes and the water column.

Splits
  year  : train 2015/2017/2021 -> test 2010/2018 (official 'train' / 'operational')
  scene : 5-fold CV over all years, grouped by scene (no duplicate leakage)
  naive : 5-fold CV over ALL images incl. the 416-px copies, grouped by nothing (leakage demo)
Models
  acoustic : B2 highlight-shadow profile (61 pts) + box size -> logistic regression
  dino_lp  : DINOv2-S frozen (CLS + mean token) -> logistic regression
  dino_ft  : DINOv2-S, last 4 blocks + head fine-tuned (year split only)
"""
import json
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.ndimage import median_filter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel

torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
random.seed(0)
np.random.seed(0)
S = json.load(open("screen.json"))
TRAIN, OP = ["2015", "2017", "2021"], ["2010", "2018"]
P = 112


def nadir(a):
    W = a.shape[1]
    col = a.mean(0)
    c0, c1 = int(W * .3), int(W * .7)
    idx = np.where(col[c0:c1] < np.percentile(col, 15) + 5)[0] + c0
    if len(idx) == 0:
        m = c0 + int(np.argmin(col[c0:c1]))
        return m, m - 25, m + 25
    return int(idx.mean()), int(idx.min()), int(idx.max())


def load1024(r):
    im = Image.open(f"raw/{r['year']}/{r['year']}/{r['file']}").convert("L")
    if im.size[0] != 1024:
        im = im.resize((1024, 1024), Image.BICUBIC)
    return np.asarray(im, float)


def crop(a, cx, cy, side, flip):
    H, W = a.shape
    h = side / 2
    x0, x1, y0, y1 = int(cx - h), int(cx + h), int(cy - h), int(cy + h)
    pad = np.pad(a, ((300, 300), (300, 300)), mode="reflect")
    x0, x1 = np.clip([x0, x1], -290, W + 290)
    y0, y1 = np.clip([y0, y1], -290, H + 290)
    c = pad[y0 + 300:y1 + 300, x0 + 300:x1 + 300]
    if c.size == 0:
        c = a[:48, :48]
    if flip:
        c = c[:, ::-1]
    return np.asarray(Image.fromarray(c.astype(np.uint8)).resize((P, P), Image.BILINEAR), float)


def acoustic_feat(a, cx, cy, w, h, nad):
    H, W = a.shape
    sgn = 1 if cx > nad else -1
    u = np.linspace(-1.5, 1.5, 61)
    xs = np.clip((cx + sgn * u * w).round().astype(int), 0, W - 1)
    y0, y1 = int(max(0, cy - h / 4)), int(min(H, cy + h / 4 + 1))
    p = a[y0:y1][:, xs].mean(0)
    ys = np.r_[max(0, int(cy - 5 * h)):max(0, int(cy - 3 * h)), min(H, int(cy + 3 * h)):min(H, int(cy + 5 * h))]
    bg = np.median(a[ys][:, xs]) if len(ys) else np.median(p)
    prof = 10 * np.log10(p / (bg + 1) + 1e-3)
    return np.r_[prof, np.log(w), np.log(h), abs(cx - nad) / W]


# ---------------------------------------------------------------- build patches
def build(records, tag):
    X, F, y, grp, yr = [], [], [], [], []
    sizes = [max(l[3], l[4]) * 1024 for r in records for l in r["labels"]]
    for r in records:
        a = median_filter(load1024(r), 3)
        nad, n0, n1 = nadir(a)
        g = f"{r['year']}_{r['scene']}" if tag != "naive" else f"{r['year']}_{r['file']}"
        bx = [(c, x * 1024, yy * 1024, w * 1024, h * 1024) for c, x, yy, w, h in r["labels"]
              if w * 1024 >= 4 and h * 1024 >= 4]
        for c, cx, cy, w, h in bx:
            side = max(2 * max(w, h), 48)
            X.append(crop(a, cx, cy, side, cx < nad)); F.append(acoustic_feat(a, cx, cy, w, h, nad))
            y.append(c); grp.append(g); yr.append(r["year"])
        for _ in range(2):                                     # background
            for _try in range(30):
                side = max(2 * random.choice(sizes), 48)
                opts = [(0.04 * 1024, n0 - side), (n1 + side, 0.96 * 1024)]
                opts = [o for o in opts if o[1] > o[0]] or [(side, 1024 - side)]
                lo_, hi_ = random.choice(opts)
                cx = random.uniform(lo_, hi_)
                cy = random.uniform(side, 1024 - side)
                if all(abs(cx - b[1]) > side and abs(cy - b[2]) > side for b in bx):
                    break
            w = h = side / 2
            X.append(crop(a, cx, cy, side, cx < nad)); F.append(acoustic_feat(a, cx, cy, w, h, nad))
            y.append(2); grp.append(g); yr.append(r["year"])
    return np.array(X, np.float32), np.array(F, np.float32), np.array(y), np.array(grp), np.array(yr)


rep = {}
for r in S["records"]:
    k = (r["year"], r["scene"])
    if k not in rep or r["size"] > rep[k]["size"]:
        rep[k] = r
t0 = time.time()
import os
if os.path.exists("b3_cache.npz"):
    z = np.load("b3_cache.npz", allow_pickle=True)
    U = tuple(z[f"U{i}"] for i in range(5)); N = tuple(z[f"N{i}"] for i in range(5))
else:
    U = build(sorted(rep.values(), key=lambda r: (r["year"], r["file"])), "scene")
    N = build(S["records"], "naive")
U = (U[0], np.nan_to_num(U[1], nan=0.0, posinf=0.0, neginf=-30.0), *U[2:])
N = (N[0], np.nan_to_num(N[1], nan=0.0, posinf=0.0, neginf=-30.0), *N[2:])
print("patches unique", len(U[2]), np.bincount(U[2]), "naive", len(N[2]), round(time.time() - t0), flush=True)

# ---------------------------------------------------------------- DINOv2 features
dino = AutoModel.from_pretrained("facebook/dinov2-small").eval()
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def to_tensor(X):
    x = (X / 255.0)[:, None].repeat(3, 1)
    x = (x - MEAN[None, :, None, None]) / STD[None, :, None, None]
    return torch.tensor(x, dtype=torch.float32)


@torch.no_grad()
def embed(X, bs=64):
    out = []
    for i in range(0, len(X), bs):
        h = dino(pixel_values=to_tensor(X[i:i + bs])).last_hidden_state
        out.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).numpy())
    return np.concatenate(out)


if os.path.exists("b3_cache.npz") and "EU" in np.load("b3_cache.npz").files:
    EU, EN = np.load("b3_cache.npz")["EU"], np.load("b3_cache.npz")["EN"]
else:
    EU = embed(U[0]); EN = embed(N[0])
    np.savez("b3_cache.npz", **{f"U{i}": U[i] for i in range(5)}, **{f"N{i}": N[i] for i in range(5)}, EU=EU, EN=EN)
print("embedded", round(time.time() - t0), flush=True)


def lr_fit_pred(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    return m.predict(sc.transform(Xte))


def scores(yt, yp):
    return dict(f1_3=float(f1_score(yt, yp, average="macro")),
                f1_bin=float(f1_score(yt == 0, yp == 0)),
                recall=[float(((yp == c) & (yt == c)).sum() / max((yt == c).sum(), 1)) for c in range(3)],
                cm=confusion_matrix(yt, yp, labels=[0, 1, 2]).tolist(), n=[int((yt == c).sum()) for c in range(3)])


res = {}
# year split
tr = np.isin(U[4], TRAIN)
te = ~tr
for name, feats in [("acoustic", U[1]), ("dino_lp", EU)]:
    res[f"year|{name}"] = scores(U[2][te], lr_fit_pred(feats[tr], U[2][tr], feats[te]))
    print("year", name, {k: v for k, v in res[f"year|{name}"].items() if k != "cm"}, flush=True)


# grouped 5-fold CV
def cv(feats, y, groups, k=5):
    ug = np.unique(groups)
    rng = np.random.default_rng(0)
    rng.shuffle(ug)
    folds = np.array_split(ug, k)
    yp = np.empty_like(y)
    for f in folds:
        m = np.isin(groups, f)
        yp[m] = lr_fit_pred(feats[~m], y[~m], feats[m])
    return scores(y, yp)


for name, (fu, fn) in {"acoustic": (U[1], N[1]), "dino_lp": (EU, EN)}.items():
    res[f"scene|{name}"] = cv(fu, U[2], U[3])
    res[f"naive|{name}"] = cv(fn, N[2], N[3])
    print(name, "scene", round(res[f"scene|{name}"]["f1_3"], 3), "naive", round(res[f"naive|{name}"]["f1_3"], 3), flush=True)


# ---------------------------------------------------------------- DINOv2 fine-tune (year split)
class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = AutoModel.from_pretrained("facebook/dinov2-small")
        for p in self.b.parameters():
            p.requires_grad = False
        for blk in self.b.encoder.layer[-4:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in self.b.layernorm.parameters():
            p.requires_grad = True
        self.fc = nn.Linear(768, 3)

    def forward(self, x):
        h = self.b(pixel_values=x).last_hidden_state
        return self.fc(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1))


def finetune(Xtr, ytr, Xte, epochs=8, seed=0):
    torch.manual_seed(seed)
    net = Head()
    w = torch.tensor(len(ytr) / (3 * np.bincount(ytr, minlength=3)), dtype=torch.float32)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=5e-5, weight_decay=0.05)
    Xt, yt = to_tensor(Xtr), torch.tensor(ytr)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(yt))
        for i in range(0, len(perm), 32):
            b = perm[i:i + 32]
            xb = Xt[b]
            if random.random() < 0.5:
                xb = torch.flip(xb, [2])                         # along-track flip (keeps shadow side)
            xb = xb + 0.05 * torch.randn_like(xb)
            loss = nn.functional.cross_entropy(net(xb), yt[b], weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
        print("ft epoch", ep, round(time.time() - t0), flush=True)
    net.eval()
    with torch.no_grad():
        Xe = to_tensor(Xte)
        return np.concatenate([net(Xe[i:i + 64]).argmax(1).numpy() for i in range(0, len(Xe), 64)])


res["year|dino_ft"] = scores(U[2][te], finetune(U[0][tr], U[2][tr], U[0][te]))
print("year dino_ft", {k: v for k, v in res["year|dino_ft"].items() if k != "cm"}, flush=True)

# examples for the report (grey patches, 48 px thumbnails)
ex = {}
for c in range(3):
    idx = np.where((U[2] == c) & te)[0][:8].tolist() + np.where((U[2] == c) & tr)[0][:8].tolist()
    ex[c] = [np.asarray(Image.fromarray(U[0][i].astype(np.uint8)).resize((56, 56))).astype(int).tolist() for i in idx]
res["examples"] = ex
res["counts"] = dict(unique=np.bincount(U[2]).tolist(), naive=np.bincount(N[2]).tolist(),
                     train=np.bincount(U[2][tr]).tolist(), test=np.bincount(U[2][te]).tolist())
json.dump(res, open("b3_patch.json", "w"))
print("saved", round(time.time() - t0))
