from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")

# Train on your pothole dataset
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    workers=4
)

# Save trained model
model.export(format="onnx")  # Export for deployment if needed
