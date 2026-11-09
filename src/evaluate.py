from ultralytics import YOLO

model = YOLO(
    "results/crack_detection/weights/best.pt"
)

metrics = model.val(
    data="dataset/data.yaml",
    split="test"
)

print(metrics)
