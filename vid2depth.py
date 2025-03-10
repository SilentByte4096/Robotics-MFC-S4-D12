import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bar

# Load MiDaS model
model_type = "DPT_Large"  # You can switch to "DPT_Hybrid" or others
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Load the appropriate transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# Input and output video paths
input_video_path = "demo.avi"  # Replace with your video file
output_video_path = "depth_video.avi"  # Output video file

# Open the input video
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), isColor=False)

# Process video frames
for _ in tqdm(range(total_frames), desc="Processing frames"):
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB (MiDaS expects RGB)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply MiDaS transform and move to device
    input_batch = transform(img).to(device)

    # Perform depth prediction
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    # Convert prediction to numpy and normalize for visualization
    depth_map = prediction.cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Write the depth map frame to the output video
    out.write(depth_map)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Depth video saved as {output_video_path}")