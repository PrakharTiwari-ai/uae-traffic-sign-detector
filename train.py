from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.train(
    data="uae_traffic.yaml",
    epochs=30,
    imgsz=640,
    batch=8,
    name="uae_signs"
)
