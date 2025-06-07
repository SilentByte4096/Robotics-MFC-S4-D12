from ultralytics import YOLO
import os
import numpy as np

# Load the trained model
model_path = "runs\\detect\\train2\\weights\\best.pt"  # Replace with your .pt file path
model = YOLO(model_path)

# Define the validation dataset
data_yaml = "data.yaml"  # Replace with your data.yaml path

# Check if files exist
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found.")
    exit()  
if not os.path.exists(data_yaml):
    print(f"Error: Data file '{data_yaml}' not found.")
    exit()

# Evaluate the model
results = model.val(data=data_yaml, imgsz=640, batch=16, device='cpu')

# Overall metrics (averaged across classes)
print("\nOverall Model Metrics (Averaged):")
print(f"Precision: {np.mean(results.box.p):.4f}")
print(f"Recall: {np.mean(results.box.r):.4f}")
print(f"mAP@0.5: {results.box.map50:.4f}")  # Scalar, no indexing needed
print(f"mAP@0.5:0.95: {np.mean(results.box.map):.4f}")

# Per-class metrics
print("\nPer-Class Metrics:")
for i, name in enumerate(results.names.values()):
    print(f"Class '{name}':")
    print(f"  Precision: {results.box.p[i]:.4f}")
    print(f"  Recall: {results.box.r[i]:.4f}")
    # Use results.box.ap50 for per-class mAP@0.5 (if available in your ultralytics version)
    try:
        print(f"  mAP@0.5: {results.box.ap50[i]:.4f}")
    except AttributeError:
        # Fallback: Approximate from maps (not exact, but close)
        print(f"  mAP@0.5 (approx): {results.box.maps[i]:.4f} (Note: This is mAP@0.5:0.95)")
    print(f"  mAP@0.5:0.95: {results.box.maps[i]:.4f}")
