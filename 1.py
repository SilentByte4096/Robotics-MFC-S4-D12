import numpy as np
import cv2
from heapq import heappush, heappop
from tqdm import tqdm
import os
import time

# A* Path Planning Class
class AStarPlanner:
    def __init__(self, grid):
        self.grid = grid.copy()
        self.rows, self.cols = grid.shape

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(self, pos):
        pos = tuple(pos)
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in directions:
            new_x, new_y = pos[0] + dx, pos[1] + dy
            if 0 <= new_x < self.cols and 0 <= new_y < self.rows and self.grid[new_y, new_x]:
                neighbors.append((new_x, new_y))
        return neighbors

    def find_path(self, start, goal):
        start = tuple(start)
        goal = tuple(goal)

        if not (0 <= start[0] < self.cols and 0 <= start[1] < self.rows and 
                0 <= goal[0] < self.cols and 0 <= goal[1] < self.rows):
            print(f"Out of bounds: start={start}, goal={goal}")
            return []

        if not self.grid[start[1], start[0]]:
            start = self.find_nearest_valid(start)
            if not start:
                print(f"No valid start found near {start}")
                return []

        if not self.grid[goal[1], goal[0]]:
            goal = self.find_nearest_valid(goal)
            if not goal:
                print(f"No valid goal found near {goal}")
                return []

        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            current = heappop(open_set)[1]

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return [list(p) for p in path[::-1]]

            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))

        print(f"No path found from {start} to {goal}")
        return []

    def find_nearest_valid(self, point):
        point = tuple(point)
        for r in range(1, 50):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    x, y = point[0] + dx, point[1] + dy
                    if 0 <= x < self.cols and 0 <= y < self.rows and self.grid[y, x]:
                        return (x, y)
        return None

# Utility Functions
def create_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask = mask_green > 0
    mask[mask_red > 0] = 0
    return mask.astype(np.uint8)

def detect_pothole(frame, robot_pos):
    # Placeholder: Detect pothole within 10 pixels of robot
    # Replace with your actual pothole detection logic
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_gray = np.array([0, 0, 50])
    upper_gray = np.array([180, 50, 150])  # Example range for potholes (adjust)
    pothole_mask = cv2.inRange(hsv, lower_gray, upper_gray)
    
    x, y = map(int, robot_pos)
    detection_radius = 10
    y_min, y_max = max(0, y - detection_radius), min(frame.shape[0], y + detection_radius)
    x_min, x_max = max(0, x - detection_radius), min(frame.shape[1], x + detection_radius)
    
    if np.any(pothole_mask[y_min:y_max, x_min:x_max]):
        # Return center of detected pothole (simplified)
        pothole_y, pothole_x = np.where(pothole_mask[y_min:y_max, x_min:x_max])
        return [x_min + pothole_x[0], y_min + pothole_y[0]]
    return None

def draw_path(frame, path, color=(255, 0, 0)):
    if not path or len(path) < 2:
        return
    for i in range(len(path) - 1):
        start_pt = tuple(map(int, path[i]))
        end_pt = tuple(map(int, path[i + 1]))
        if (0 <= start_pt[0] < frame.shape[1] and 0 <= start_pt[1] < frame.shape[0] and 
            0 <= end_pt[0] < frame.shape[1] and 0 <= end_pt[1] < frame.shape[0]):
            cv2.line(frame, start_pt, end_pt, color, 2)

def draw_robot(frame, position, color=(0, 255, 0), radius=5):
    position = tuple(map(int, position))
    if 0 <= position[0] < frame.shape[1] and 0 <= position[1] < frame.shape[0]:
        cv2.circle(frame, position, radius, color, -1)
        cv2.circle(frame, position, int(radius + 10), (0, 255, 255), 1)

def find_goal_position(start_pos, distance_y, mask):
    start_pos = tuple(start_pos)
    goal = [start_pos[0], max(0, start_pos[1] - distance_y)]
    planner = AStarPlanner(mask)
    if not mask[int(goal[1]), int(goal[0])]:
        nearest = planner.find_nearest_valid(goal)
        if nearest:
            return list(nearest)
        return list(start_pos)
    return goal

def plan_path_segment(start_pos, frame_mask, distance_y):
    planner = AStarPlanner(frame_mask)
    goal = find_goal_position(start_pos, distance_y, frame_mask)
    path = planner.find_path(start_pos, goal)
    return path

# Main Pipeline
def main_pipeline(video_path, output_video_path="output_video.avi", output_frames_dir="output_frames", 
                  distance_y=100, steps_per_frame=1, delay_per_frame=0.5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file at {video_path}")

    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Cannot read first frame")

    height, width = first_frame.shape[:2]
    start_pos = (width // 2, height // 2)

    first_mask = create_mask(first_frame)
    planner = AStarPlanner(first_mask)
    if not first_mask[start_pos[1], start_pos[0]]:
        start_pos = planner.find_nearest_valid(start_pos)
        if not start_pos:
            raise ValueError("No traversable starting point found")
        print(f"Adjusted start position to {start_pos}")

    current_pos = np.array(start_pos)
    current_path = plan_path_segment(current_pos, first_mask, distance_y)
    path_index = 0

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    if not os.path.exists(output_frames_dir):
        os.makedirs(output_frames_dir)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    with tqdm(total=total_frames, desc="Processing and Saving") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            curr_mask = create_mask(frame)

            # Simulate pothole detection
            pothole = detect_pothole(frame, current_pos)
            if pothole:
                # Mark pothole as non-traversable in mask
                px, py = map(int, pothole)
                curr_mask[max(0, py-5):min(height, py+5), max(0, px-5):min(width, px+5)] = 0
                # Replan path to avoid pothole
                current_path = plan_path_segment(current_pos, curr_mask, distance_y)
                path_index = 0

            if current_path and path_index < len(current_path):
                steps_to_take = min(steps_per_frame, len(current_path) - path_index)
                for _ in range(steps_to_take):
                    next_pos = current_path[path_index]
                    draw_path(frame, current_path, color=(255, 0, 0))  # Blue path
                    draw_robot(frame, next_pos, radius=5)
                    current_pos = np.array(next_pos)
                    path_index += 1
            else:
                draw_robot(frame, current_pos, radius=5)
                # Replan if path is exhausted
                current_path = plan_path_segment(current_pos, curr_mask, distance_y)
                path_index = 0

            frame_filename = os.path.join(output_frames_dir, f"frame_{frame_count:06d}.png")
            cv2.imwrite(frame_filename, frame)
            out.write(frame)

            time.sleep(delay_per_frame)  # Simulate pothole detection delay

            frame_count += 1
            pbar.update(1)

    cap.release()
    out.release()
    print(f"Video and frames saved. Video: {output_video_path}, Frames: {output_frames_dir}")

if __name__ == "__main__":
    VIDEO_PATH = "seg.mp4"
    OUTPUT_VIDEO_PATH = "output_video.avi"
    OUTPUT_FRAMES_DIR = "output_frames"
    DISTANCE_Y = 100
    STEPS_PER_FRAME = 1  # Slow movement
    DELAY_PER_FRAME = 0.5  # Adjust based on pothole detection speed

    try:
        main_pipeline(VIDEO_PATH, OUTPUT_VIDEO_PATH, OUTPUT_FRAMES_DIR, DISTANCE_Y, STEPS_PER_FRAME, DELAY_PER_FRAME)
    except Exception as e:
        print(f"An error occurred: {e}")