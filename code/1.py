import cv2
import numpy as np
import threading
import queue
import socket
import json
import sys
import time
import os
import datetime
from ultralytics import YOLO
import msvcrt  # For non-blocking input on Windows

# Configuration
PI_IP = "192.168.58.146"
PI_PORT = 65432
MJPG_URL = f"http://192.168.58.146:8080/?action=stream"
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
COMMAND_INTERVAL = 0.5

# Shared resources
command_queue = queue.Queue()
stop_event = threading.Event()
recording_event = threading.Event()
last_command_time = time.time()

# Load YOLOv8 model for pothole detection
pothole_model = YOLO('C:\\Users\\Srikrishna\\Documents\\GitHub\\Sem_4\\Robotics-MFC-S4-D12\\runs\\detect\\train2\\weights\\best.pt')

# Initialize video capture
cap = cv2.VideoCapture(MJPG_URL)
if not cap.isOpened():
    print("Error: Could not open MJPG stream from the Raspberry Pi.")
    sys.exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
frame_width = FRAME_WIDTH
frame_height = FRAME_HEIGHT

def pothole_in_path(pothole_boxes):
    center_x = frame_width // 2
    bot_width_pixels = frame_width // 4
    for x1, y1, x2, y2 in pothole_boxes:
        if x1 <= center_x + bot_width_pixels and x2 >= center_x - bot_width_pixels:
            return True
    return False

def detect_potholes(frame):
    pothole_boxes = []
    results = pothole_model(frame)
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        pothole_boxes.append((x1, y1, x2, y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return pothole_boxes

def display_video():
    cap = cv2.VideoCapture(MJPG_URL)
    if not cap.isOpened():
        print("Failed to open video stream")
        stop_event.set()
        return
    
    height, width = FRAME_HEIGHT, FRAME_WIDTH
    video_writer = None
    prev_time = time.time()
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (width, height))
            pothole_boxes = detect_potholes(frame)
            if pothole_in_path(pothole_boxes):
                command = "stop"
            else:
                command = "forward"
            current_time = time.time()
            if current_time - last_command_time >= COMMAND_INTERVAL:
                command_queue.put({"action": command})
                globals()['last_command_time'] = current_time
            # Start recording if not already started
            if recording_event.is_set() and video_writer is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join("recordings", f"recording_{timestamp}.avi")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(filename, fourcc, 30, (width, height))
            # Write frame if recording
            if recording_event.is_set() and video_writer is not None:
                video_writer.write(frame)
            fps = 1 / (current_time - prev_time) if current_time != prev_time else 0
            prev_time = current_time
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Video Feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
        else:
            print("Failed to get frame")
            time.sleep(0.1)
    
    # Release video writer at the end
    if video_writer is not None:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

def handle_socket():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((PI_IP, PI_PORT))
        print(f"Connected to Pi at {PI_IP}:{PI_PORT}")
    except Exception as e:
        print(f"Failed to connect to Pi: {e}")
        stop_event.set()
        return
    
    while not stop_event.is_set():
        try:
            cmd = command_queue.get(timeout=0.1)
            s.sendall(json.dumps(cmd).encode() + b'\n')
            ack = s.recv(1024).decode().strip()
            if not ack:
                print("Connection closed by server")
                break
            elif ack != "ACK":
                print("Did not receive ACK")
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Socket error: {e}")
            break
    
    s.close()

def main():
    os.makedirs("recordings", exist_ok=True)
    recording_event.set()  # Start recording immediately
    
    video_thread = threading.Thread(target=display_video)
    video_thread.start()
    
    socket_thread = threading.Thread(target=handle_socket)
    socket_thread.start()
    
    try:
        while not stop_event.is_set():
            # Check for user input without blocking
            if msvcrt.kbhit():
                user_input = msvcrt.getch().decode().lower()
                if user_input == "q":
                    stop_event.set()
                    print("Quitting program")
                    break
            time.sleep(0.1)  # Brief sleep to avoid busy-waiting
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        stop_event.set()
    except Exception as e:
        print(f"Error: {e}")
        stop_event.set()
    
    video_thread.join()
    socket_thread.join()
    print("Program terminated")

if __name__ == "__main__":
    main()