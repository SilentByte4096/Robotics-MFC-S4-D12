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
import select
import heapq

# Configuration
PI_IP = "192.168.58.189"          # Raspberry Pi IP
PI_PORT = 65432                   # Socket port
MJPG_URL = f"http://10.12.209.62:8080/?action=stream"  # MJPG stream URL
FRAME_WIDTH = 320                 # Reduced for performance
FRAME_HEIGHT = 240                # Reduced for performance
COMMAND_INTERVAL = 0.5            # Seconds between commands
GRID_RESOLUTION = 20              # Grid size for path planning
COMMAND_EXECUTION_TIME = 1.0      # Time to execute each motor command

# Shared resources
command_queue = queue.Queue()     # Queue for commands
stop_event = threading.Event()    # Event to stop threads
recording_event = threading.Event()  # Event for recording control
last_command_time = time.time()   # Tracks last command sent
path_execution_state = {
    'executing': False,
    'current_path': None,
    'path_index': 0
}  # State for path execution

# Load YOLOv8 model for pothole detection
try:
    pothole_model = YOLO('C:\\Users\\srikr\\Documents\\GitHub\\Robotics-MFC-S4-D12-1\\best.pt')
    print("YOLOv8 model loaded successfully")
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
    sys.exit(1)

# Initialize video capture
cap = cv2.VideoCapture(MJPG_URL)
if not cap.isOpened():
    print("Error: Could not open MJPG stream from the Raspberry Pi.")
    sys.exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
frame_width = FRAME_WIDTH
frame_height = FRAME_HEIGHT

def create_occupancy_grid(pothole_mask):
    """Create occupancy grid based on pothole mask."""
    grid_height = frame_height // GRID_RESOLUTION
    grid_width = frame_width // GRID_RESOLUTION
    occupancy_grid = np.zeros((grid_height, grid_width), dtype=np.uint8)
    for i in range(grid_height):
        for j in range(grid_width):
            y_start = i * GRID_RESOLUTION
            y_end = (i + 1) * GRID_RESOLUTION
            x_start = j * GRID_RESOLUTION
            x_end = (j + 1) * GRID_RESOLUTION
            pothole_patch = pothole_mask[y_start:y_end, x_start:x_end]
            if np.any(pothole_patch > 0):
                occupancy_grid[i, j] = 1
    return occupancy_grid, grid_height, grid_width

def heuristic(a, b):
    """Manhattan distance heuristic for A*."""
    return abs(b[0] - a[0]) + abs(b[1] - a[1])

def a_star(grid, start, goal):
    """A* algorithm for path planning."""
    open_set = {start}
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    heap = [(f_score[start], start)]
    while heap:
        current = heapq.heappop(heap)[1]
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        open_set.remove(current)
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = current[0] + di, current[1] + dj
            if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1] and grid[ni, nj] == 0:
                tentative_g = g_score[current] + 1
                if (ni, nj) not in g_score or tentative_g < g_score[(ni, nj)]:
                    came_from[(ni, nj)] = current
                    g_score[(ni, nj)] = tentative_g
                    f_score[(ni, nj)] = tentative_g + heuristic((ni, nj), goal)
                    if (ni, nj) not in open_set:
                        open_set.add((ni, nj))
                        heapq.heappush(heap, (f_score[(ni, nj)], (ni, nj)))
    return None

def pothole_in_path(pothole_boxes):
    """Check if a pothole is in the bot's center path."""
    center_x = frame_width // 2
    bot_width_pixels = frame_width // 4
    for x1, y1, x2, y2 in pothole_boxes:
        if x1 <= center_x + bot_width_pixels and x2 >= center_x - bot_width_pixels:
            return True, y1, y2
    return False, None, None

def get_next_command(current_pos, next_pos):
    """Convert grid movement to a motor command."""
    di = next_pos[0] - current_pos[0]
    dj = next_pos[1] - current_pos[1]
    if di == -1 and dj == 0:
        return "forward"
    elif di == 0 and dj == -1:
        return "left"
    elif di == 0 and dj == 1:
        return "right"
    return "stop"

def detect_potholes(frame):
    """Detect potholes using YOLOv8."""
    pothole_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    pothole_boxes = []
    try:
        results = pothole_model(frame)
        print(f"Pothole detection ran: {len(results[0].boxes)} potholes detected")
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            pothole_mask[y1:y2, x1:x2] = 255
            pothole_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    except Exception as e:
        print(f"Pothole detection error: {e}")
    return pothole_mask, pothole_boxes

def draw_path(frame, path, grid_resolution=GRID_RESOLUTION):
    """Draw the planned path on the frame."""
    if path:
        for i in range(len(path) - 1):
            row1, col1 = path[i]
            row2, col2 = path[i + 1]
            pt1 = (col1 * grid_resolution + grid_resolution // 2,
                   row1 * grid_resolution + grid_resolution // 2)
            pt2 = (col2 * grid_resolution + grid_resolution // 2,
                   row2 * grid_resolution + grid_resolution // 2)
            cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
        cv2.circle(frame, pt1, 5, (0, 255, 0), -1)  # Start (green)
        cv2.circle(frame, pt2, 5, (0, 0, 255), -1)  # Goal (red)
    return frame

def display_video():
    """Thread for video display, pothole detection, path planning, and recording."""
    cap = cv2.VideoCapture(MJPG_URL)
    if not cap.isOpened():
        print("Failed to open video stream")
        stop_event.set()
        return
    
    height, width = FRAME_HEIGHT, FRAME_WIDTH
    video_writer = None
    prev_time = time.time()
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        client.connect((PI_IP, PI_PORT))
        print(f"Connected to Pi at {PI_IP}:{PI_PORT}")
    except Exception as e:
        print(f"Failed to connect to Pi: {e}")
        stop_event.set()
        return
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (width, height))
            
            if path_execution_state['executing']:
                if path_execution_state['path_index'] < len(path_execution_state['current_path']) - 1:
                    current_pos = path_execution_state['current_path'][path_execution_state['path_index']]
                    next_pos = path_execution_state['current_path'][path_execution_state['path_index'] + 1]
                    command = get_next_command(current_pos, next_pos)
                    current_time = time.time()
                    if current_time - last_command_time >= COMMAND_EXECUTION_TIME:
                        cmd_data = {"action": command}
                        client.sendall(json.dumps(cmd_data).encode() + b'\n')
                        print(f"Laptop: Sent command - {command}")  # Log on laptop
                        ack = client.recv(1024).decode().strip()
                        if ack == "ACK":
                            path_execution_state['path_index'] += 1
                            globals()['last_command_time'] = current_time
                        else:
                            print("Did not receive ACK")
                else:
                    cmd_data = {"action": "stop"}
                    client.sendall(json.dumps(cmd_data).encode() + b'\n')
                    print(f"Laptop: Sent command - stop")  # Log on laptop
                    path_execution_state['executing'] = False
                    path_execution_state['current_path'] = None
                    path_execution_state['path_index'] = 0
            else:
                pothole_mask, pothole_boxes = detect_potholes(frame)
                pothole_detected, pothole_y1, pothole_y2 = pothole_in_path(pothole_boxes)
                if pothole_detected:
                    cmd_data = {"action": "stop"}
                    client.sendall(json.dumps(cmd_data).encode() + b'\n')
                    print(f"Laptop: Sent command - stop")  # Log on laptop
                    occupancy_grid, grid_height, grid_width = create_occupancy_grid(pothole_mask)
                    center_col = grid_width // 2
                    start_row = min(grid_height - 1, (pothole_y2 // GRID_RESOLUTION) + 1)
                    goal_row = max(0, (pothole_y1 // GRID_RESOLUTION) - 1)
                    start = (start_row, center_col)
                    goal = (goal_row, center_col)
                    path = a_star(occupancy_grid, start, goal)
                    if path:
                        path_execution_state['executing'] = True
                        path_execution_state['current_path'] = path
                        path_execution_state['path_index'] = 0
                    else:
                        print("No path found around pothole")
                else:
                    current_time = time.time()
                    if current_time - last_command_time >= COMMAND_INTERVAL:
                        cmd_data = {"action": "forward"}
                        client.sendall(json.dumps(cmd_data).encode() + b'\n')
                        print(f"Laptop: Sent command - forward")  # Log on laptop
                        globals()['last_command_time'] = current_time
            
            frame = draw_path(frame, path_execution_state['current_path'])
            
            if recording_event.is_set() and video_writer is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join("recordings", f"recording_{timestamp}.avi")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(filename, fourcc, 30, (width, height))
            if not recording_event.is_set() and video_writer is not None:
                video_writer.release()
                video_writer = None
            
            if recording_event.is_set() and video_writer is not None:
                video_writer.write(frame)
            
            current_time = time.time()
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
    
    if video_writer is not None:
        video_writer.release()
    cap.release()
    client.close()
    cv2.destroyAllWindows()

def main():
    """Main function to start threads and handle user input."""
    os.makedirs("recordings", exist_ok=True)
    recording_event.set()  # Start recording immediately
    
    video_thread = threading.Thread(target=display_video)
    video_thread.start()
    
    try:
        while not stop_event.is_set():
            if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                user_input = sys.stdin.readline().strip().lower()
                if user_input == "quit":
                    stop_event.set()
                    print("Quitting program")
                    break
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        stop_event.set()
    except Exception as e:
        print(f"Error: {e}")
        stop_event.set()
    
    video_thread.join()
    print("Program terminated")

if __name__ == "__main__":
    main()