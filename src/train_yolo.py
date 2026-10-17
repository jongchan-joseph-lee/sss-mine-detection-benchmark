from ultralytics import YOLO
m = YOLO('yolov8n.pt')
m.train(data='tiles/milco_tiles.yaml', epochs=25, imgsz=320, batch=16, device='cpu', workers=2,
        project='runs', name='yolo_tiles', exist_ok=True, plots=False, flipud=0.5, fliplr=0.5,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.3, seed=0, deterministic=True, patience=10, verbose=False)
print('TRAIN_DONE')
