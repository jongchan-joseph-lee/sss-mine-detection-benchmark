"""
Detector training for the augmentation / architecture comparison.
Same hyper-parameters as train_yolo.py except epochs and patience, so runs differ only in data and model.

  python train_exp.py --data tiles/milco_tiles.yaml     --model yolov8n.pt  --epochs 40 --name n_base40
  python train_exp.py --data tiles/milco_tiles_aug.yaml --model yolov8n.pt  --epochs 40 --name n_aug40
  python train_exp.py --data tiles/milco_tiles_aug.yaml --model rtdetr-l.pt --epochs 60 --name rtdetr_aug60 --device 0   # GPU
"""
import argparse

from ultralytics import RTDETR, YOLO

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
ap.add_argument("--resume", action="store_true")
a = ap.parse_args()

if a.resume:
    m = (RTDETR if "rtdetr" in a.name else YOLO)(f"runs/detect/runs/exp/{a.name}/weights/last.pt")
    m.train(resume=True)
else:
    m = (RTDETR if "rtdetr" in a.model else YOLO)(a.model)
    m.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device, workers=a.workers,
            project="runs/exp", name=a.name, exist_ok=True, plots=False, flipud=0.5, fliplr=0.5,
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.3, seed=a.seed, deterministic=True, patience=100, verbose=False)
print("TRAIN_DONE", a.name)
