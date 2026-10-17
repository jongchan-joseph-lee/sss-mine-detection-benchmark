"""
B3-1: evaluate the tile-trained YOLOv8n on full 1024-px frames.
  - standard mAP@0.5 (ultralytics val) on operational years and on held-out train-year scenes
  - point-based FROC (same matching rule as the B2 matched filter) for a like-for-like comparison
"""
import glob
import json
import os
import sys

import numpy as np
from ultralytics import YOLO

W = sys.argv[1] if len(sys.argv) > 1 else "runs/detect/runs/yolo_tiles/weights/best.pt"
m = YOLO(W)
src = open("b2.py").read()
exec(src[src.index("def match("):src.index("det_res = {}")])
out = {}

for name, yaml in [("operational", "tiles/milco_test.yaml")]:
    v = m.val(data=yaml, imgsz=1024, batch=4, device="cpu", conf=0.001, iou=0.6, plots=False, verbose=False,
              project="runs", name=f"val_{name}", exist_ok=True)
    out[f"map_{name}"] = dict(map50=float(v.box.map50), map50_95=float(v.box.map),
                             per_class_ap50=[float(x) for x in v.box.ap50], p=float(v.box.mp), r=float(v.box.mr))
    print(name, out[f"map_{name}"], flush=True)


def froc_dir(d):
    rows, rows_m, ng, ngm, ni = [], [], 0, 0, 0
    per_img = []
    for f in sorted(glob.glob(f"{d}/images/*.png")):
        lab = []
        lf = f.replace("/images/", "/labels/")[:-4] + ".txt"
        if os.path.exists(lf):
            lab = [[int(p.split()[0])] + [float(x) for x in p.split()[1:]] for p in open(lf).read().strip().splitlines() if p.strip()]
        r = m.predict(f, imgsz=1024, conf=0.01, device="cpu", verbose=False)[0]
        b = r.boxes
        xywhn = b.xywhn.numpy()
        cls = b.cls.numpy().astype(int)
        conf = b.conf.numpy()
        dets = [(float(x), float(y), float(c)) for (x, y, _, _), c in zip(xywhn, conf)]
        mm, g, _ = match(dets, lab)
        rows += mm; ng += g; ni += 1
        # MILCO-only: detections predicted as MILCO vs MILCO ground truth
        dm = [(float(x), float(y), float(c)) for (x, y, _, _), c, k in zip(xywhn, conf, cls) if k == 0]
        lm = [l for l in lab if l[0] == 0]
        m2, g2, _ = match(dm, lm)
        rows_m += m2; ngm += g2
        per_img.append(dict(f=os.path.basename(f), n_gt=g, top=[(round(x, 4), round(y, 4), round(c, 3)) for x, y, c in dets[:5]]))
    c1, raw1 = froc(rows, ng, ni)
    c2, raw2 = froc(rows_m, ngm, ni)
    return dict(n_img=ni, n_gt=ng, froc_obj=c1, raw_obj=raw1, n_gt_milco=ngm, froc_milco=c2, raw_milco=raw2), per_img


for name, d in [("operational", "tiles/test_full"), ("val_scenes", "tiles/val_full")]:
    out[f"froc_{name}"], pi = froc_dir(d)
    out[f"per_img_{name}"] = pi
    print(name, out[f"froc_{name}"]["froc_obj"], out[f"froc_{name}"]["froc_milco"], flush=True)

json.dump(out, open("b3_yolo.json", "w"))
print("saved")
