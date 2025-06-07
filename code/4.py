import cv2
import numpy as np
import socket
import json
import time
import os
import queue
import threading
import datetime
from path_planning import create_occupancy_grid, a_star_planning, process_path  # Assuming these are in a separate module

# Constants
MJPG_URL = "http://192.168.58.146:8080/?action=stream"  # Replace with actual IP
PI_IP = "192.168.58.146"  # Replace with actual IP
PI_PORT = 65432
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
GRID_WIDTH = 20
GRID_HEIGHT = 20
CELL_SIZE = FRAME_WIDTH / GRID_WIDTH  # Adjust based on your needs
FRAME_TIME = 0.2  # 5 FPS

# Global variables
stop_event = threading.Event()
recording_event = threading.Event()
command_queue = queue.Queue()
command_list = []  # Global command list for path execution

def draw_path(frame, path, grid_height, grid_width, frame_height, frame_width):
    """Draw the planned path on the frame."""
    if not path:
        return frame
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        y1 = int((r1 / grid_height) * frame_height)
        x1 = int((c1 / grid_width) * frame_width)
        y2 = int((r2 / grid_height) * frame_height)
        x2 = int((c2 / grid_width) * frame_width)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame

def detect_potholes(frame):
    """Placeholder for pothole detection; replace with your implementation."""
    # Example: return [] if no potholes, else list of bounding boxes [(x, y, w, h), ...]
    return []  # Adjust based on actual detection logic

def display_video():
    cap = cv2.VideoCapture(MJPG_URL)
    if not cap.isOpened():
        print("Failed to open video stream")
        stop_event.set()
        return

    height, width = FRAME_HEIGHT, FRAME_WIDTH
    video_writer = None
    frame_count = 0
    pothole_detected_time = 0
    is_stopped = False
    prev_time = time.time()

    while not stop_event.is_set():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Failed to get frame")
            time.sleep(0.1)
            continue

        frame_count += 1
        frame = cv2.resize(frame, (width, height))
        pothole_boxes = detect_potholes(frame)
        current_time = time.time()

        # Always plan a path (active even without potholes)
        occupancy_grid = create_occupancy_grid(pothole_boxes, GRID_HEIGHT, GRID_WIDTH, height, width)
        start = (GRID_HEIGHT - 1, GRID_WIDTH // 2)
        goal = (0, GRID_WIDTH // 2)
        path = a_star_planning(occupancy_grid, start, goal)
        frame = draw_path(frame, path, GRID_HEIGHT, GRID_WIDTH, height, width)  # Display path

        # Handle pothole avoidance and movement
        if pothole_boxes and not is_stopped and not command_list:
            command_queue.put("stop")
            pothole_detected_time = current_time
            is_stopped = True
        elif is_stopped and (current_time - pothole_detected_time) >= 3 and not command_list:
            if path:
                command_list.extend(process_path(path, CELL_SIZE))
            else:
                command_queue.put("stop")
        elif command_list:
            command = command_list.pop(0)
            command_queue.put(command)
            print(f"Sending command: {command}")
        elif not pothole_boxes:
            command_queue.put("forward 10")
            is_stopped = False
            print("Sending command: forward 10")

        # Calculate and display FPS
        fps = 1 / (current_time - prev_time) if current_time != prev_time else 0
        prev_time = current_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Recording logic
        if recording_event.is_set() and video_writer is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join("recordings", f"recording_{timestamp}.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(filename, fourcc, 30, (width, height))
        if recording_event.is_set() and video_writer is not None:
            video_writer.write(frame)

        # Display frame
        print("Displaying frame")  # Debug to confirm display
        cv2.imshow('Video Feed', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):  # Increased wait time for visibility
            stop_event.set()
            break

        # Maintain 5 FPS
        elapsed_time = time.time() - start_time
        if elapsed_time < FRAME_TIME:
            time.sleep(FRAME_TIME - elapsed_time)

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
            s.sendall(cmd.encode() + b'\n')
            ack = s.recv(1024).decode().strip()
            if not ack:
                print("Connection closed by server")
                break
            elif ack != "ACK":
                print(f"Did not receive ACK, got: {ack}")
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Socket error: {e}")
            break

    s.close()

if __name__ == "__main__":
    os.makedirs("recordings", exist_ok=True)
    video_thread = threading.Thread(target=display_video)
    socket_thread = threading.Thread(target=handle_socket)
    video_thread.start()
    socket_thread.start()
    video_thread.join()
    socket_thread.join()