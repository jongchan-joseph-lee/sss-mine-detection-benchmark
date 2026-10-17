import base64
import io
import json

import numpy as np
from PIL import Image
from ultralytics import YOLO

S = json.load(open("screen.json"))
years = {y: {k: v for k, v in d.items() if not k.startswith(("box_", "objs_per"))} for y, d in S["years"].items()}

rep = {}
for r in S["records"]:
    k = (r["year"], r["scene"])
    if k not in rep or r["size"] > rep[k]["size"]:
        rep[k] = r
sizes = {"MILCO": {"w": [], "h": []}, "NOMBO": {"w": [], "h": []}}
for r in rep.values():
    for c, x, y, w, h in r["labels"]:
        n = "MILCO" if c == 0 else "NOMBO"
        sizes[n]["w"].append(round(w * 1024, 1)); sizes[n]["h"].append(round(h * 1024, 1))

b2 = json.load(open("b2_results.json"))
mf = {k: dict(froc=v["froc"], raw=v["raw"], n_img=v["n_img"], n_gt=v["n_gt"]) for k, v in b2["mf"].items()}
sig = json.load(open("b2_signature.json"))
P = json.load(open("b3_patch.json"))
examples = {int(k): v for k, v in P.pop("examples").items()}
Yo = json.load(open("b3_yolo.json"))
Yo = {k: v for k, v in Yo.items() if not k.startswith("per_img")}

# example operational frames: pick frames with GT objects, show GT + top YOLO boxes
m = YOLO("runs/detect/runs/yolo_tiles/weights/best.pt")
cands = []
import glob, os
for f in sorted(glob.glob("tiles/test_full/images/*.png")):
    lf = f.replace("/images/", "/labels/")[:-4] + ".txt"
    lab = [list(map(float, l.split())) for l in open(lf).read().strip().splitlines() if l.strip()] if os.path.exists(lf) else []
    if lab:
        cands.append((f, lab))
frames = []
picked = 0
for f, lab in cands:
    r = m.predict(f, imgsz=1024, conf=0.25, device="cpu", verbose=False)[0]
    det = [[*map(float, b), float(c)] for b, c in zip(r.boxes.xywhn.numpy(), r.boxes.conf.numpy())]
    hit = any(np.hypot(d[0] - l[1], d[1] - l[2]) < max(l[3], l[4], 0.012) for d in det for l in lab)
    if hit and len(det) <= 6:
        im = Image.open(f).convert("L").resize((512, 512), Image.BILINEAR)
        buf = io.BytesIO(); im.save(buf, "PNG", optimize=True)
        frames.append(dict(name=os.path.basename(f), png=base64.b64encode(buf.getvalue()).decode(),
                           gt=[l[1:5] for l in lab], det=[[round(v, 4) for v in d] for d in det[:6]]))
        picked += 1
    if picked == 2:
        break
cap = (f"시험 연도 장면 {', '.join(fr['name'][:-4] for fr in frames)}. 실선 녹색은 정답, 주황 점선은 신뢰도 0.25 이상 YOLO 탐지와 그 신뢰도입니다. "
       "표적은 화면 전체에서 수 픽셀 크기라 상자를 표적보다 조금 크게 그렸습니다.")

out = dict(years=years, sizes=sizes, mf=mf, sig=sig, patch=P, examples=examples, yolo=Yo, frames=frames, frame_cap=cap)
s = json.dumps(out).replace("</", "<\\/")
t = open("report_B_template.html").read()
open("milco-report.html", "w").write(t.replace("__DATA__", s))
print("built", len(s), [f["name"] for f in frames])
