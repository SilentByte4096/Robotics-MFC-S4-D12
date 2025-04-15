import cv2
import numpy as np
import heapq
import time
import argparse
import os

class PotholeAvoidanceSystem:
    def __init__(self, grid_resolution=20):
        """
        Initialize the pothole avoidance system.
        
        Args:
            grid_resolution: Resolution reduction factor for the occupancy grid
        """
        self.grid_resolution = grid_resolution
        self.current_path = None
        self.last_plan_time = 0
        self.replan_interval = 1.0  # Replan every 1 second
        self.traversal_distance = 10  # Distance to commit to before replanning
        self.distance_traveled = 0
        self.occupancy_grid = None
        self.grid_shape = None
        
    def detect_potholes(self, frame):
        """
        Detect potholes in the input frame using color thresholding for red outlines.
        
        Args:
            frame: Input video frame
            
        Returns:
            Binary mask where 1 represents pothole areas
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.dilate(red_mask, kernel, iterations=2)
        red_mask = cv2.erode(red_mask, kernel, iterations=1)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pothole_mask = np.zeros_like(red_mask)
        for contour in contours:
            if cv2.contourArea(contour) > 50:
                cv2.drawContours(pothole_mask, [contour], -1, 255, -1)
        return pothole_mask
    
    def create_occupancy_grid(self, frame, pothole_mask):
        """
        Create a lower resolution occupancy grid from the pothole mask.
        
        Args:
            frame: Original frame for size reference
            pothole_mask: Binary mask of pothole areas
            
        Returns:
            Occupancy grid where 1 represents obstacles/potholes
        """
        height, width = pothole_mask.shape
        grid_height = height // self.grid_resolution
        grid_width = width // self.grid_resolution
        self.grid_shape = (grid_height, grid_width)
        occupancy_grid = np.zeros((grid_height, grid_width), dtype=np.uint8)
        for i in range(grid_height):
            for j in range(grid_width):
                region = pothole_mask[i*self.grid_resolution:(i+1)*self.grid_resolution, 
                                      j*self.grid_resolution:(j+1)*self.grid_resolution]
                if np.any(region > 0):
                    occupancy_grid[i, j] = 1
        return occupancy_grid
    
    def get_pothole_size(self, i, j):
        region = self.pothole_mask[i*self.grid_resolution:(i+1)*self.grid_resolution, 
                                 j*self.grid_resolution:(j+1)*self.grid_resolution]
        return np.sum(region > 0)
    
    def heuristic(self, a, b):
        return abs(b[0] - a[0]) + abs(b[1] - a[1])
    
    def a_star_planning(self, start, goal):
        if self.occupancy_grid[start[0], start[1]] == 1:
            start = self.find_nearest_free_cell(start)
        if self.occupancy_grid[goal[0], goal[1]] == 1:
            goal = self.find_nearest_free_cell(goal)
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        close_set = set()
        came_from = {}
        gscore = {start: 0}
        fscore = {start: self.heuristic(start, goal)}
        oheap = []
        heapq.heappush(oheap, (fscore[start], start))
        pothole_sizes = {}
        while oheap:
            current = heapq.heappop(oheap)[1]
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            close_set.add(current)
            for i, j in neighbors:
                neighbor = (current[0] + i, current[1] + j)
                if (neighbor[0] < 0 or neighbor[0] >= self.grid_shape[0] or
                    neighbor[1] < 0 or neighbor[1] >= self.grid_shape[1] or
                    neighbor in close_set):
                    continue
                tentative_g_score = gscore[current] + 1
                if self.occupancy_grid[neighbor[0], neighbor[1]] == 1:
                    if neighbor not in pothole_sizes:
                        pothole_sizes[neighbor] = self.get_pothole_size(neighbor[0], neighbor[1])
                    tentative_g_score += 100 + pothole_sizes[neighbor]
                if neighbor not in gscore or tentative_g_score < gscore[neighbor]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
        if not self.current_path:
            print("No completely safe path found. Finding path with smallest pothole.")
            return self.find_path_with_smallest_pothole(start, goal)
        return self.current_path
    
    def find_nearest_free_cell(self, cell):
        i, j = cell
        max_distance = max(self.grid_shape)
        for d in range(1, max_distance):
            for di in range(-d, d+1):
                for dj in range(-d, d+1):
                    if abs(di) + abs(dj) == d:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < self.grid_shape[0] and 
                            0 <= nj < self.grid_shape[1] and
                            self.occupancy_grid[ni, nj] == 0):
                            return (ni, nj)
        return cell
    
    def find_path_with_smallest_pothole(self, start, goal):
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        close_set = set()
        came_from = {}
        gscore = {start: 0}
        fscore = {start: self.heuristic(start, goal)}
        oheap = []
        heapq.heappush(oheap, (fscore[start], start))
        pothole_sizes = {}
        while oheap:
            current = heapq.heappop(oheap)[1]
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            close_set.add(current)
            for i, j in neighbors:
                neighbor = (current[0] + i, current[1] + j)
                if (neighbor[0] < 0 or neighbor[0] >= self.grid_shape[0] or
                    neighbor[1] < 0 or neighbor[1] >= self.grid_shape[1] or
                    neighbor in close_set):
                    continue
                tentative_g_score = gscore[current] + 1
                if self.occupancy_grid[neighbor[0], neighbor[1]] == 1:
                    if neighbor not in pothole_sizes:
                        pothole_sizes[neighbor] = self.get_pothole_size(neighbor[0], neighbor[1])
                    tentative_g_score += pothole_sizes[neighbor]
                if neighbor not in gscore or tentative_g_score < gscore[neighbor]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
        return None
    
    def draw_path(self, frame, path):
        """
        Draw the computed path on the frame.
        
        Args:
            frame: Input video frame
            path: List of grid positions forming the path
            
        Returns:
            Frame with the path drawn on it
        """
        result_frame = frame.copy()
        if path:
            pixel_path = []
            for point in path:
                pixel_y = (point[0] * self.grid_resolution) + (self.grid_resolution // 2)
                pixel_x = (point[1] * self.grid_resolution) + (self.grid_resolution // 2)
                pixel_path.append((pixel_x, pixel_y))
            for i in range(len(pixel_path) - 1):
                cv2.line(result_frame, pixel_path[i], pixel_path[i+1], 
                        (255, 0, 0), 2)  # Blue line
            cv2.circle(result_frame, pixel_path[0], 5, (0, 255, 0), -1)  # Green start
            cv2.circle(result_frame, pixel_path[-1], 5, (0, 0, 255), -1)  # Red goal
        return result_frame
    
    def visualize_occupancy_grid(self, frame):
        result_frame = frame.copy()
        if self.occupancy_grid is not None:
            for i in range(self.grid_shape[0]):
                for j in range(self.grid_shape[1]):
                    x = j * self.grid_resolution
                    y = i * self.grid_resolution
                    if self.occupancy_grid[i, j] == 1:
                        cv2.rectangle(result_frame, (x, y), 
                                     (x + self.grid_resolution, y + self.grid_resolution), 
                                     (0, 0, 255), 1)
                        overlay = result_frame.copy()
                        cv2.rectangle(overlay, (x, y), 
                                     (x + self.grid_resolution, y + self.grid_resolution), 
                                     (0, 0, 255), -1)
                        cv2.addWeighted(overlay, 0.3, result_frame, 0.7, 0, result_frame)
        return result_frame
    
    def process_frame(self, frame):
        """
        Process a single video frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (full_visualization_frame, path_only_frame)
        """
        self.pothole_mask = self.detect_potholes(frame)
        self.occupancy_grid = self.create_occupancy_grid(frame, self.pothole_mask)
        start = (self.grid_shape[0] - 5, self.grid_shape[1] // 2)
        goal = (5, self.grid_shape[1] // 2)
        current_time = time.time()
        new_plan_needed = (
            self.current_path is None or
            (current_time - self.last_plan_time) > self.replan_interval or
            self.distance_traveled >= self.traversal_distance
        )
        if new_plan_needed:
            self.current_path = self.a_star_planning(start, goal)
            self.last_plan_time = current_time
            self.distance_traveled = 0
        else:
            self.distance_traveled += 1
        
        # Create path-only frame
        path_only_frame = self.draw_path(frame.copy(), self.current_path)
        
        # Create full visualization frame
        result_frame = self.visualize_occupancy_grid(frame)
        result_frame = self.draw_path(result_frame, self.current_path)
        full_vis_frame = result_frame.copy()
        contours, _ = cv2.findContours(self.pothole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(full_vis_frame, contours, -1, (0, 0, 255), 2)
        info_text1 = f"Grid Resolution: {self.grid_resolution}px"
        info_text2 = f"Replanning: {'Yes' if new_plan_needed else 'No'}"
        info_text3 = f"Distance Traveled: {self.distance_traveled}/{self.traversal_distance}"
        cv2.putText(full_vis_frame, info_text1, (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(full_vis_frame, info_text2, (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(full_vis_frame, info_text3, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return full_vis_frame, path_only_frame

def main():
    video_file = "seg.mp4"
    grid_resolution = 20
    if not os.path.exists(video_file):
        print(f"Error: Video file '{video_file}' not found")
        print("Falling back to webcam")
        cap = cv2.VideoCapture(0)
    else:
        print(f"Using video file: {video_file}")
        cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    print(f"Video dimensions: {frame_width}x{frame_height}, FPS: {fps}")
    
    # Initialize two video writers
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_full = cv2.VideoWriter("full_vis.avi", fourcc, fps, (frame_width, frame_height))
    out_path = cv2.VideoWriter("path_only.avi", fourcc, fps, (frame_width, frame_height))
    
    avoidance_system = PotholeAvoidanceSystem(grid_resolution=grid_resolution)
    cv2.namedWindow('Pothole Avoidance System', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Pothole Avoidance System', frame_width, frame_height)
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame")
            break
        frame_count += 1
        
        # Process frame to get both full visualization and path-only frames
        full_vis_frame, path_only_frame = avoidance_system.process_frame(frame)
        
        # Write frames to their respective videos
        out_full.write(full_vis_frame)
        out_path.write(path_only_frame)
        
        # Calculate and display FPS on full visualization frame
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            processing_fps = frame_count / elapsed_time
            cv2.putText(full_vis_frame, f"Processing FPS: {processing_fps:.1f}", 
                       (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Pothole Avoidance System', full_vis_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting")
            break
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"pothole_frame_{timestamp}.jpg"
            cv2.imwrite(filename, full_vis_frame)
            print(f"Saved frame as {filename}")
    
    total_time = time.time() - start_time
    average_fps = frame_count / total_time if total_time > 0 else 0
    print(f"Processed {frame_count} frames in {total_time:.2f} seconds ({average_fps:.2f} FPS)")
    
    cap.release()
    out_full.release()
    out_path.release()
    cv2.destroyAllWindows()
    
    print("Output videos saved: full_vis.avi and path_only.avi")

if __name__ == "__main__":
    main()