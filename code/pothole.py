import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from ultralytics import YOLO
import socket
import threading
import sys
import time

# Check if Pi's IP address is provided as a command-line argument
if len(sys.argv) != 2:
    print("Usage: python pothole.py 192.168.58.146")
    sys.exit(1)

# Configuration
PI_IP = sys.argv[1]  # Raspberry Pi's IP address from command line
PI_PORT = 65432      # Port for socket communication
MJPG_URL = f'http://192.168.58.146:8080/?action=stream'  # MJPG stream URL from Pi

# Load the pre-trained SegFormer model for road segmentation
feature_extractor = SegformerFeatureExtractor.from_pretrained(
    "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
)
segmentation_model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
)
segmentation_model.eval()

# Load the pothole detection model (update this path to your YOLOv8 weights file)
pothole_model = YOLO('best.pt')  # Replace with your actual path

# Initialize video capture from the Pi's MJPG stream
cap = cv2.VideoCapture(MJPG_URL)
if not cap.isOpened():
    print("Error: Could not open MJPG stream from the Raspberry Pi.")
    sys.exit(1)

# Get frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define visualization colors
TRAVERSABLE_COLOR = (0, 255, 0)  # Green for traversable road
POTHOLE_COLOR = (0, 0, 255)      # Red for potholes

# Shared resources
stop_event = threading.Event()  # Event to signal threads to stop

def handle_socket():
    """Thread to manage socket communication with the Raspberry Pi."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((PI_IP, PI_PORT))
        print(f"Connected to Raspberry Pi at {PI_IP}:{PI_PORT}")
    except Exception as e:
        print(f"Failed to connect to Pi: {e}")
        stop_event.set()
        return

    while not stop_event.is_set():
        try:
            # Receive data from the Pi (customize this based on your needs)
            data = s.recv(1024).decode()
            if not data:
                break
            print(f"Received from Pi: {data}")
            # Add logic here to process Pi data or send commands if needed
        except Exception as e:
            print(f"Socket communication error: {e}")
            break

    s.close()

def process_video():
    """Thread to process the MJPG stream, detect potholes, segment roads, and display results."""
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from MJPG stream.")
            stop_event.set()
            break

        # Convert frame to RGB for segmentation
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        inputs = feature_extractor(images=frame_pil, return_tensors="pt")

        # Perform road segmentation with SegFormer
        with torch.no_grad():
            outputs = segmentation_model(**inputs)
            logits = outputs.logits
            logits = torch.nn.functional.interpolate(
                logits, size=(frame_height, frame_width), mode='bilinear', align_corners=False
            )
            pred = logits.argmax(dim=1)[0].cpu().numpy()  # Predicted class per pixel

        # Create road mask (class 0 assumed as road)
        road_mask = (pred == 0).astype(np.uint8)

        # Initialize output frame as black
        output_frame = np.zeros_like(frame)

        # Detect potholes with YOLOv8
        pothole_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        pothole_results = pothole_model(frame)
        for box in pothole_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            pothole_mask[y1:y2, x1:x2] = 1

        # Apply colors based on masks
        output_frame[(road_mask == 1) & (pothole_mask == 0)] = TRAVERSABLE_COLOR  # Traversable road
        output_frame[(road_mask == 1) & (pothole_mask == 1)] = POTHOLE_COLOR      # Potholes

        # Display the processed frame
        cv2.imshow('Live Feed with Segmentation and Detection', output_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    """Main function to start socket and video processing threads."""
    # Start socket thread
    socket_thread = threading.Thread(target=handle_socket)
    socket_thread.start()

    # Start video processing thread
    video_thread = threading.Thread(target=process_video)
    video_thread.start()

    # Wait for threads to complete
    video_thread.join()
    socket_thread.join()
    print("Program terminated.")

if __name__ == "__main__":
    main()