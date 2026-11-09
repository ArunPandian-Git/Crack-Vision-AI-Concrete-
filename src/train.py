from ultralytics import YOLO

# Load pretrained segmentation model
model = YOLO("yolov8n-seg.pt")

# Train the model
results = model.train(

    data="dataset/data.yaml",

    epochs=50,

    imgsz=640,

    batch=8,

    project="results",

    name="crack_detection",

    patience=10,

    device="cpu"
)

print("Training Completed!")
