import cv2
import numpy as np
import heapq
import time

# Load video
video_path = "output.avi"
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define video writer
out = cv2.VideoWriter("output_video.avi", cv2.VideoWriter_fourcc(*"XVID"), fps, (frame_width, frame_height))

# A* Pathfinding Algorithm
def astar(grid, start, goal):
    rows, cols = grid.shape
    open_set = [(0, start)]
    heapq.heapify(open_set)
    came_from = {}
    g_score = {start: 0}
    f_score = {start: np.linalg.norm(np.array(start) - np.array(goal))}
    
    directions = [(-1,0), (1,0), (0,-1), (0,1)]  # Up, Down, Left, Right

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor] == 1:
                temp_g_score = g_score[current] + 1
                if neighbor not in g_score or temp_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = temp_g_score + np.linalg.norm(np.array(neighbor) - np.array(goal))
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No path found

# Path memory and navigation history
path_memory = []
frame_counter = 0
last_direction = None  # To prevent immediate reversals

# Safety Margin
buffer_size = 10  # Extra buffer for potholes
extra_safe_distance = 5  # Additional safe distance for wider turns

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Detect red pothole regions
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    pothole_mask = mask1 | mask2  # Final pothole mask

    # Find contours and draw filled bounding boxes
    contours, _ = cv2.findContours(pothole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), -1)  # Filled red box

    # Create navigation grid (10x10 pixel blocks)
    grid_size = 10
    grid_rows, grid_cols = frame_height // grid_size, frame_width // grid_size
    grid = np.ones((grid_rows, grid_cols), dtype=np.uint8)

    pothole_positions = []
    for i in range(grid_rows):
        for j in range(grid_cols):
            x1, y1 = j * grid_size, i * grid_size
            x2, y2 = x1 + grid_size, y1 + grid_size  # Fixed y2 issue
            if np.any(pothole_mask[y1:y2, x1:x2] > 0):
                pothole_positions.append((i, j))
                grid[i, j] = 0

    # Expand danger zone
    for r, c in pothole_positions:
        for dr in range(-buffer_size - extra_safe_distance, buffer_size + extra_safe_distance + 1):
            for dc in range(-buffer_size - extra_safe_distance, buffer_size + extra_safe_distance + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_rows and 0 <= nc < grid_cols:
                    grid[nr, nc] = 0

    start = (grid_rows - 1, grid_cols // 2)
    goal = (0, grid_cols // 2)

    if frame_counter % 5 == 0 or not path_memory:
        path_memory = astar(grid, start, goal)
        time.sleep(0.5)

    movement_commands = []
    if path_memory:
        step_count = min(9, len(path_memory))  # Extended path display
        for idx in range(step_count):
            r, c = path_memory[idx]
            x, y = c * grid_size + grid_size // 2, r * grid_size + grid_size // 2
            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)  # Ensure blue path is visible

        path_memory = path_memory[step_count:]
        
        # Display the next decision at the top of the frame
        if movement_commands:
            cv2.putText(frame, movement_commands[0], (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    out.write(frame)
    cv2.imshow("Pothole Avoidance", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
