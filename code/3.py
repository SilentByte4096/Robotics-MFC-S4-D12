import socket
import cv2
import threading
import os
from datetime import datetime

# Socket client setup
HOST = '192.168.58.146'  # Pi's IP address
PORT = 65432            # Port to connect to

# Webcam stream URL
STREAM_URL = 'http://192.168.58.146:8080/?action=stream'

# Create recordings directory if it doesn't exist
if not os.path.exists('recordings'):
    os.makedirs('recordings')

# Video recording setup with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"recordings/recording_{timestamp}.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_filename, fourcc, 20.0, (640, 480))

# Function to record webcam stream
def record_stream():
    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        print("Error: Could not open video stream")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        out.write(frame)
    cap.release()

# Start recording in a separate thread
recording_thread = threading.Thread(target=record_stream, daemon=True)
recording_thread.start()
print(f"Started recording webcam stream to {output_filename}...")

# Main client logic
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    print("Connected to Raspberry Pi. Enter commands (e.g., 'forward 100', 'right 90', 'stop') or 'exit' to quit.")
    while True:
        command = input("Command: ").strip().lower()
        if command == 'exit':
            break
        s.sendall(command.encode())

# Stop recording when program ends
out.release()
print("Program ended. Recording stopped.")