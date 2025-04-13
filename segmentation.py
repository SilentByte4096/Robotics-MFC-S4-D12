import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from ultralytics import YOLO

# Load the pre-trained SegFormer model for semantic segmentation
feature_extractor = SegformerFeatureExtractor.from_pretrained(
    "nvidia/segformer-b0-finetuned-cityscapes-512-1024",
)
segmentation_model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-cityscapes-512-1024",
)
segmentation_model.eval()

# Load your pothole detection model
pothole_model = YOLO('C:\\Users\\Srikrishna\\Documents\\GitHub\\Sem_4\\Robotics-MFC-S4-D12\\runs\\detect\\train2\\weights\\best.pt')  # Adjust path as necessary

# Define video input and output paths
video_path = 'demo.mp4'  # Replace with your video file path
output_path = 'segmented_road_only_output.mp4'

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
TRAVERSABLE_COLOR = (0, 255, 0)  # Green for traversable road
POTHOLE_COLOR = (0, 0, 255)      # Red for potholes

# Process video frames
frame_count = 0
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

    # Create road mask: 1 for road (class 0), 0 for non-road
    road_mask = (pred == 0).astype(np.uint8)

    # Initialize output frame as black
    output_frame = np.zeros_like(frame)

    # Create pothole mask
    pothole_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    pothole_results = pothole_model(frame)
    for box in pothole_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        pothole_mask[y1:y2, x1:x2] = 1

    # Apply colors only to road areas
    output_frame[(road_mask == 1) & (pothole_mask == 0)] = TRAVERSABLE_COLOR  # Traversable road
    output_frame[(road_mask == 1) & (pothole_mask == 1)] = POTHOLE_COLOR      # Potholes on road
    # Non-road areas remain black (already initialized as zeros)

    # Write to output video
    out.write(output_frame)

    # Optional: Display the frame (comment out if not needed)
    # cv2.imshow('Segmented Road Only', output_frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    frame_count += 1
    print(f"Processed frame {frame_count}")

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processing complete. Output saved to '{output_path}'.")