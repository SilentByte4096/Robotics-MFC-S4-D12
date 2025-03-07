import cv2
import torch
import numpy as np

# Load the MiDaS model
model_type = "MiDaS_small"
model = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.eval()

# Load the appropriate transform based on model type
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform if model_type == "MiDaS_small" else midas_transforms.dpt_transform

# Load and preprocess the image
img_path = "pothole.jpg"
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Image file '{img_path}' not found.")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

input_batch = transform(img).to(device)

with torch.no_grad():
    prediction = model(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

depth_map = prediction.cpu().numpy()
depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
depth_map_color = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_INFERNO)

# Save the images instead of displaying them
cv2.imwrite("original_image.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
cv2.imwrite("depth_map.jpg", depth_map_color)