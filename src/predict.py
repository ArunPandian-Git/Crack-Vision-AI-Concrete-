from ultralytics import YOLO

model = YOLO(
    "results/crack_detection/weights/best.pt"
)

def predict_crack(image_path):

    results = model.predict(
        source=image_path,
        conf=0.25
    )

    return results
