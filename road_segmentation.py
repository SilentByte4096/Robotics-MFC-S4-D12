import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from ultralytics import YOLO  # Assuming your pothole model is YOLO-based


# Load the pre-trained SegFormer model for semantic segmentation
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

token = "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"  # Replace with the token you copied
feature_extractor = SegformerFeatureExtractor.from_pretrained(
    "nvidia/segformer-b0-finetuned-cityscapes-512-1024",  # Corrected model name
    token=token
)
segmentation_model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-cityscapes-512-1024",
    token=token
)
segmentation_model.eval()

# Load your pothole detection model
# Replace with the path to your model file
pothole_model = YOLO('C:\\Users\\nandu\\Downloads\\Robotics-MFC-S4-D12\\runs\\detect\\train2\\weights\\best.pt')  # Adjust the path as necessary

# Define video input and output paths
video_path = 'demo.mp4'  # Replace with your video file path
output_path = 'segmented_road_output.mp4'

# Initialize video capture
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize video writer
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Define visualization colors
TRAVERSABLE_COLOR = (0, 255, 0)  # Green for traversable (road without potholes)
NON_TRAVERSABLE_COLOR = (0, 0, 255)  # Red for non-traversable (sky, grass, potholes, etc.)

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    # Prepare frame for segmentation
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    inputs = feature_extractor(images=frame_pil, return_tensors="pt")

    # Perform semantic segmentation
    with torch.no_grad():
        outputs = segmentation_model(**inputs)
        logits = outputs.logits
        # Resize logits to match original frame size
        logits = torch.nn.functional.interpolate(
            logits, size=(frame_height, frame_width), mode='bilinear', align_corners=False
        )
        pred = logits.argmax(dim=1)[0].cpu().numpy()  # Class indices per pixel

    # Create binary mask: 0 for traversable (road), 1 for non-traversable
    # In Cityscapes, class 0 is 'road'
    mask = (pred != 0).astype(np.uint8)

    # Integrate pothole detection
    pothole_results = pothole_model(frame)
    for box in pothole_results[0].boxes:
        x1, y1, x2, y2 = map(int,box.xyxy[0])
        # Mark pothole areas as non-traversable
        mask[y1:y2, x1:x2] = 1

    # Visualize the mask
    overlay = frame.copy()
    overlay[mask == 0] = TRAVERSABLE_COLOR  # Traversable areas
    overlay[mask == 1] = NON_TRAVERSABLE_COLOR  # Non-traversable areas

    # Blend overlay with original frame
    alpha = 0.4  # Transparency
    output_frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0.0)

    # Display the frame
    cv2.imshow('Traversable Area Segmentation', output_frame)

    # Write to output video
    out.write(output_frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processing complete. Output saved to '{output_path}'.")