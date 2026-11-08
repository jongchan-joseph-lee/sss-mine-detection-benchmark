"""
Run the GPU detector comparison on a local NVIDIA GPU (Windows / Linux), end to end:
data download -> tiles + augmentation -> 6 training runs -> test-year evaluation -> results zip.

  cd src
  python run_local.py              # everything (about 2-3 h on an RTX 3070 Ti)
  python run_local.py --epochs 60  # same as the Colab notebook (default)

Safe to stop and restart: downloaded data and tiles are reused, finished runs are skipped,
interrupted runs resume from their last epoch, cached detections are reused.
Output: results_gpu.zip (b5_eval_gpu.json, per-frame detections, final weights, training logs).
"""
import argparse
import os
import subprocess
import sys
import urllib.request
import zipfile

FIGSHARE = {"2010": 43169008, "2015": 43169002, "2017": 43169005, "2018": 43169011, "2021": 43168999}
GRID = [("rtdetr", "rtdetr-l.pt", "aug"), ("rtdetr", "rtdetr-l.pt", "base"),
        ("s", "yolov8s.pt", "aug"), ("s", "yolov8s.pt", "base"),
        ("n", "yolov8n.pt", "aug"), ("n", "yolov8n.pt", "base")]
DATA = {"base": "tiles/milco_tiles.yaml", "aug": "tiles/milco_tiles_aug.yaml"}
ENV = dict(os.environ, PYTHONUTF8="1", B5_DEVICE="0")


def step(msg):
    print(f"\n=== {msg}", flush=True)


def run(args, env=ENV):
    print(">", " ".join(args), flush=True)
    r = subprocess.run([sys.executable] + args, env=env)
    if r.returncode != 0:
        sys.exit(f"failed: {' '.join(args)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    step("GPU check")
    import torch
    if not torch.cuda.is_available():
        sys.exit("CUDA is not available. Install the CUDA build of PyTorch (see the README / pytorch.org) and retry.")
    print(torch.__version__, torch.cuda.get_device_name(0))

    step("data (Santos et al. 2024, figshare, CC BY 4.0)")
    for y, fid in FIGSHARE.items():
        d = os.path.join("raw", y, y)
        if os.path.isdir(d) and os.listdir(d):
            print(y, "ok")
            continue
        os.makedirs(os.path.join("raw", y), exist_ok=True)
        z = os.path.join("raw", f"{y}.zip")
        print(y, "downloading ...", flush=True)
        urllib.request.urlretrieve(f"https://api.figshare.com/v2/file/download/{fid}", z)
        zipfile.ZipFile(z).extractall(os.path.join("raw", y))

    step("tiles and augmentation")
    if not os.path.isdir("tiles/test_full/images"):
        run(["prep_tiles.py"])
    if not os.path.isdir("tiles/train_aug/images"):
        run(["prep_aug.py"])

    step(f"training grid ({len(GRID)} runs, {a.epochs} epochs)")
    project = os.path.abspath("runs_gpu")
    runs = {}
    for mk, w, dk in GRID:
        name = f"{mk}_{dk}{a.epochs}"
        run(["train_exp.py", "--auto", "--data", DATA[dk], "--model", w, "--epochs", str(a.epochs), "--name", name,
             "--device", "0", "--workers", str(a.workers), "--batch", str(a.batch), "--project", project])
        runs[name] = os.path.join(project, name, "weights", "last.pt")

    step("test-year evaluation")
    args = ["b5_eval.py", "ref=runs/detect/runs/yolo_tiles/weights/best.pt"] + [f"{k}={v}" for k, v in runs.items()]
    run(args, env=dict(ENV, B5_OUT="b5_eval_gpu.json"))

    step("results_gpu.zip")
    with zipfile.ZipFile("results_gpu.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write("b5_eval_gpu.json")
        for f in os.listdir("b5_dets"):
            if any(f.startswith(k + "_") for k in runs):
                z.write(os.path.join("b5_dets", f))
        for k in runs:
            for f in ("results.csv", os.path.join("weights", "last.pt")):
                p = os.path.join(project, k, f)
                if os.path.exists(p):
                    z.write(p, os.path.join("runs_gpu", k, f))
    print("done ->", os.path.abspath("results_gpu.zip"))


if __name__ == "__main__":
    main()
