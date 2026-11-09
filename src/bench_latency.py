"""
Inference latency per 1024-px frame for every trained detector, on GPU and CPU.

  python src/bench_latency.py            # writes src/gpu_runs/latency.json (about 5-10 min)

End-to-end time of one frame as used in evaluation (pre-processing, model, post-processing):
  full    : one 1024-px pass
  sliced  : sixteen 320-px tiles (stride 256) in one batch + point merge
Warm-up frames are discarded; the median and 90th percentile over the measured frames are reported.
"""
import glob
import json
import os
import platform
import statistics
import sys
import time

import numpy as np
import torch
from PIL import Image
from ultralytics import RTDETR, YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
RUNS = {"ref_n25_cpu": "runs/detect/runs/yolo_tiles/weights/best.pt"}
for n in ("n_base60", "n_aug60", "s_base60", "s_aug60", "rtdetr_base60", "rtdetr_aug60"):
    p = os.path.join("runs_gpu", n, "weights", "last.pt")
    if os.path.exists(p):
        RUNS[n] = p
FRAMES = sorted(glob.glob(os.path.join("tiles", "test_full", "images", "*.png")))
N = {"cuda": (10, 100), "cpu": (3, 30)}           # (warm-up, measured) frames per device
T, S = 320, 256
XS = list(range(0, 1024 - T + 1, S)); XS = XS if XS[-1] == 1024 - T else XS + [1024 - T]


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def one_frame(m, im, mode, dev):
    d = 0 if dev == "cuda" else "cpu"
    if mode == "full":
        m.predict(im, imgsz=1024, conf=0.01, device=d, verbose=False)
    else:
        tiles = [im[y:y + T, x:x + T] for y in XS for x in XS]
        res = m.predict(tiles, imgsz=320, conf=0.01, device=d, verbose=False)
        pts = sorted(((float(c), float(b[0])) for r in res for b, c in zip(r.boxes.xywh.cpu().numpy(), r.boxes.conf.cpu().numpy())),
                     reverse=True)                # same work order as the point merge in b5_eval
        _ = pts[:300]


def main():
    devices = (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]
    out = dict(machine=dict(os=platform.platform(), cpu=platform.processor(), cpu_threads=torch.get_num_threads(),
                            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                            python=platform.python_version(), torch=torch.__version__),
               frames=dict(cuda=N["cuda"][1], cpu=N["cpu"][1]), runs={})
    imgs = [np.asarray(Image.open(f).convert("RGB")) for f in FRAMES[: max(w + n for w, n in N.values())]]
    for name, w in RUNS.items():
        m = (RTDETR if "rtdetr" in name else YOLO)(w)
        params = sum(p.numel() for p in m.model.parameters())
        out["runs"][name] = dict(weights=w, parameters=params)
        for dev in devices:
            for mode in ("full", "sliced"):
                warm, n = N[dev]
                ts = []
                for i, im in enumerate(imgs[: warm + n]):
                    sync(dev); t0 = time.perf_counter()
                    one_frame(m, im, mode, dev)
                    sync(dev); dt = (time.perf_counter() - t0) * 1000
                    if i >= warm:
                        ts.append(dt)
                q = statistics.quantiles(ts, n=10)
                out["runs"][name][f"{dev}_{mode}_ms"] = dict(median=round(statistics.median(ts), 2), p90=round(q[-1], 2))
                print(f"{name:15s} {dev:4s} {mode:6s} median {statistics.median(ts):8.1f} ms  p90 {q[-1]:8.1f} ms", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    os.makedirs("gpu_runs", exist_ok=True)
    json.dump(out, open(os.path.join("gpu_runs", "latency.json"), "w"), indent=1)
    print("saved", os.path.abspath(os.path.join("gpu_runs", "latency.json")), flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
