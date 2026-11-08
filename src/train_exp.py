"""
Detector training for the augmentation / architecture comparison.
Same hyper-parameters as train_yolo.py except epochs and patience, so runs differ only in data and model.

  python train_exp.py --data tiles/milco_tiles.yaml     --model yolov8n.pt  --epochs 40 --name n_base40
  python train_exp.py --data tiles/milco_tiles_aug.yaml --model yolov8n.pt  --epochs 40 --name n_aug40
  python train_exp.py --data tiles/milco_tiles_aug.yaml --model rtdetr-l.pt --epochs 60 --name rtdetr_aug60 --device 0   # GPU
"""
import argparse
import os

from ultralytics import RTDETR, YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--project", default="runs/exp", help="absolute path (e.g. on Google Drive) keeps runs across disconnects")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--auto", action="store_true", help="skip a finished run, resume an interrupted one, else start")
    a = ap.parse_args()

    run_dir = os.path.join(a.project if os.path.isabs(a.project) else os.path.join("runs/detect", a.project), a.name)
    last = os.path.join(run_dir, "weights", "last.pt")
    if a.auto and os.path.exists(last):
        csv = os.path.join(run_dir, "results.csv")
        done = sum(1 for _ in open(csv)) - 1 if os.path.exists(csv) else 0
        if done >= a.epochs:
            print("SKIP finished", a.name)
            return
        a.resume = True

    if a.resume:
        m = (RTDETR if "rtdetr" in a.name else YOLO)(last)
        m.train(resume=True)
    else:
        m = (RTDETR if "rtdetr" in a.model else YOLO)(a.model)
        m.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device, workers=a.workers,
                project=a.project, name=a.name, exist_ok=True, plots=False, flipud=0.5, fliplr=0.5,
                hsv_h=0.0, hsv_s=0.0, hsv_v=0.3, seed=a.seed, deterministic=True, patience=100, verbose=False)
    print("TRAIN_DONE", a.name)


if __name__ == "__main__":
    main()
