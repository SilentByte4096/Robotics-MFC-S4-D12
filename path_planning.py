import cv2
import numpy as np
import heapq

# Define constants
GRID_RESOLUTION = 20  # Downsample factor for the occupancy grid
TOP_CUTOFF_RATIO = 0.1  # Ignore top 10% of the frame
BOTTOM_CUTOFF_RATIO = 0.7  # Finalize paths crossing bottom 30%

# Colors for paths (BGR format)
GLOBAL_PATH_COLOR = (255, 255, 0)  # Yellow
LOCAL_PATH_COLOR = (0, 255, 255)   # Cyan
FINAL_PATH_COLOR = (255, 0, 255)   # Magenta

# Function to extract road mask (green areas)
def extract_road_mask(frame):
    """Extract a binary mask where green pixels (road) are 1."""
    return ((frame[:, :, 0] == 0) & 
            (frame[:, :, 1] == 255) & 
            (frame[:, :, 2] == 0)).astype(np.uint8)

# Function to extract pothole mask (red areas)
def extract_pothole_mask(frame):
    """Extract a binary mask where red pixels (potholes) are 1."""
    return ((frame[:, :, 0] == 0) & 
            (frame[:, :, 1] == 0) & 
            (frame[:, :, 2] == 255)).astype(np.uint8)

# Create a downsampled occupancy grid
def create_occupancy_grid(road_mask, pothole_mask, grid_resolution):
    """Convert road and pothole masks into a coarse grid for A* planning."""
    height, width = road_mask.shape
    grid_height = height // grid_resolution
    grid_width = width // grid_resolution
    occupancy_grid = np.zeros((grid_height, grid_width), dtype=np.uint8)
    for i in range(grid_height):
        for j in range(grid_width):
            region_road = road_mask[i*grid_resolution:(i+1)*grid_resolution, 
                                   j*grid_resolution:(j+1)*grid_resolution]
            region_pothole = pothole_mask[i*grid_resolution:(i+1)*grid_resolution, 
                                         j*grid_resolution:(j+1)*grid_resolution]
            # Mark as obstacle (1) if not all pixels are road or any are potholes
            if not np.all(region_road == 1) or np.any(region_pothole == 1):
                occupancy_grid[i, j] = 1
    return occupancy_grid

# Find the nearest free cell in the grid
def find_nearest_free_cell(occupancy_grid, cell):
    """Find the closest traversable cell using BFS if the given cell is occupied."""
    grid_height, grid_width = occupancy_grid.shape
    i, j = cell
    if occupancy_grid[i, j] == 0:
        return cell
    queue = [(i, j)]
    visited = {(i, j)}
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    while queue:
        current_i, current_j = queue.pop(0)
        for di, dj in directions:
            ni, nj = current_i + di, current_j + dj
            if (0 <= ni < grid_height and 0 <= nj < grid_width and 
                    (ni, nj) not in visited):
                visited.add((ni, nj))
                if occupancy_grid[ni, nj] == 0:
                    return (ni, nj)
                queue.append((ni, nj))
    return cell  # Fallback to original cell if no free cell is found

# Heuristic function for A* (Manhattan distance)
def heuristic(a, b):
    """Calculate the Manhattan distance between two points."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# A* path planning algorithm
def a_star_planning(start, goal, grid):
    """Find the shortest path from start to goal on the grid using A*."""
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    open_heap = [(fscore[start], start)]
    heapq.heapify(open_heap)
    
    while open_heap:
        current = heapq.heappop(open_heap)[1]
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        close_set.add(current)
        for di, dj in neighbors:
            neighbor = (current[0] + di, current[1] + dj)
            if (0 <= neighbor[0] < grid.shape[0] and 
                    0 <= neighbor[1] < grid.shape[1]):
                if grid[neighbor[0], neighbor[1]] == 1:
                    continue
                tentative_g_score = gscore[current] + 1
                if (neighbor in close_set and 
                        tentative_g_score >= gscore.get(neighbor, float('inf'))):
                    continue
                if (tentative_g_score < gscore.get(neighbor, float('inf'))):
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (fscore[neighbor], neighbor))
    return None  # No path found

# Convert pixel coordinates to grid coordinates
def pixel_to_grid(x, y, grid_resolution):
    """Map pixel (x, y) to grid (row, col)."""
    return (y // grid_resolution, x // grid_resolution)

# Convert grid coordinates to pixel coordinates
def grid_to_pixel(i, j, grid_resolution):
    """Map grid (row, col) to pixel (x, y) at the center of the cell."""
    return (j * grid_resolution + grid_resolution // 2, 
            i * grid_resolution + grid_resolution // 2)

# Main processing function
def main():
    # Input and output video setup
    video_path = 'seg.mp4'  # Pre-segmented video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out_combined = cv2.VideoWriter('path_planning_combined.mp4', 
                                  cv2.VideoWriter_fourcc(*'mp4v'), 
                                  fps, (frame_width, frame_height))
    out_path_only = cv2.VideoWriter('final_path_only.mp4', 
                                   cv2.VideoWriter_fourcc(*'mp4v'), 
                                   fps, (frame_width, frame_height))

    # Define frame cutoffs
    TOP_CUTOFF = int(frame_height * TOP_CUTOFF_RATIO)
    BOTTOM_CUTOFF = int(frame_height * BOTTOM_CUTOFF_RATIO)

    # Compute global path from the first frame
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        exit()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start

    road_mask = extract_road_mask(first_frame)
    global_path = []
    for y in range(TOP_CUTOFF, frame_height):
        road_indices = np.where(road_mask[y] == 1)[0]
        if len(road_indices) > 0:
            center_x = int(np.mean(road_indices))
            global_path.append((center_x, y))

    # Process each frame
    final_path_segments = []
    active_local_path = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        # Extract masks from the pre-segmented frame
        road_mask = extract_road_mask(frame)
        pothole_mask = extract_pothole_mask(frame)
        occupancy_grid = create_occupancy_grid(road_mask, pothole_mask, GRID_RESOLUTION)

        # Check if the global path is blocked by a pothole in the middle section
        path_blocked = False
        pothole_position = None
        for idx, (x, y) in enumerate(global_path):
            if TOP_CUTOFF <= y < BOTTOM_CUTOFF:
                grid_i, grid_j = pixel_to_grid(x, y, GRID_RESOLUTION)
                if occupancy_grid[grid_i, grid_j] == 1:
                    path_blocked = True
                    pothole_position = idx
                    break

        # Plan a local path if the global path is blocked and no active local path exists
        if path_blocked and active_local_path is None:
            start_idx = max(0, pothole_position - 10)  # Start slightly before pothole
            start_pixel = global_path[start_idx]
            start_grid = pixel_to_grid(start_pixel[0], start_pixel[1], GRID_RESOLUTION)
            if occupancy_grid[start_grid[0], start_grid[1]] == 1:
                start_grid = find_nearest_free_cell(occupancy_grid, start_grid)

            goal_idx = min(len(global_path) - 1, pothole_position + 10)  # End after pothole
            goal_pixel = global_path[goal_idx]
            goal_grid = pixel_to_grid(goal_pixel[0], goal_pixel[1], GRID_RESOLUTION)
            if occupancy_grid[goal_grid[0], goal_grid[1]] == 1:
                goal_grid = find_nearest_free_cell(occupancy_grid, goal_grid)

            local_path_grid = a_star(start_grid, goal_grid, occupancy_grid)
            if local_path_grid:
                active_local_path = [grid_to_pixel(i, j, GRID_RESOLUTION) 
                                    for i, j in local_path_grid]

        # Finalize the local path when it crosses the bottom cutoff
        if active_local_path:
            last_y = active_local_path[-1][1]
            if last_y >= BOTTOM_CUTOFF:
                final_path_segments.append(active_local_path)
                active_local_path = None

        # Visualize combined output (segmented frame with paths)
        output_frame = frame.copy()
        for i in range(len(global_path) - 1):
            cv2.line(output_frame, global_path[i], global_path[i + 1], 
                    GLOBAL_PATH_COLOR, 5)
        if active_local_path:
            for i in range(len(active_local_path) - 1):
                cv2.line(output_frame, active_local_path[i], active_local_path[i + 1], 
                        LOCAL_PATH_COLOR, 5)
        for segment in final_path_segments:
            for i in range(len(segment) - 1):
                cv2.line(output_frame, segment[i], segment[i + 1], 
                        FINAL_PATH_COLOR, 5)
        cv2.line(output_frame, (0, BOTTOM_CUTOFF), (frame_width, BOTTOM_CUTOFF), 
                (255, 255, 255), 2)  # Bottom cutoff line

        # Visualize path-only output (black background with paths)
        path_only_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        for i in range(len(global_path) - 1):
            cv2.line(path_only_frame, global_path[i], global_path[i + 1], 
                    GLOBAL_PATH_COLOR, 5)
        if active_local_path:
            for i in range(len(active_local_path) - 1):
                cv2.line(path_only_frame, active_local_path[i], active_local_path[i + 1], 
                        LOCAL_PATH_COLOR, 5)
        for segment in final_path_segments:
            for i in range(len(segment) - 1):
                cv2.line(path_only_frame, segment[i], segment[i + 1], 
                        FINAL_PATH_COLOR, 5)

        # Write frames to output videos
        out_combined.write(output_frame)
        out_path_only.write(path_only_frame)

    # Cleanup
    cap.release()
    out_combined.release()
    out_path_only.release()
    print("Processing complete. Outputs saved as 'path_planning_combined.mp4' and 'final_path_only.mp4'.")

if __name__ == "__main__":
    main()