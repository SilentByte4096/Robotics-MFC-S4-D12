import numpy as np
import cv2
import math
import time
from queue import PriorityQueue
import threading
from ultralytics import YOLO

class RealTimePotholeAvoidance:
    def __init__(self, vehicle_width=2.0, safety_margin=0.3, grid_size=0.1, 
                 lookahead_distance=15.0, path_update_freq=10):
        """
        Initialize the real-time pothole avoidance system.
        
        Args:
            vehicle_width: Width of the vehicle in meters
            safety_margin: Additional safety margin around obstacles in meters
            grid_size: Size of each grid cell in meters for path planning
            lookahead_distance: How far ahead to plan paths in meters
            path_update_freq: How many times per second to update the path
        """
        self.vehicle_width = vehicle_width
        self.safety_margin = safety_margin
        self.grid_size = grid_size
        self.total_margin = self.vehicle_width/2 + self.safety_margin
        self.lookahead_distance = lookahead_distance
        self.path_update_interval = 1.0 / path_update_freq
        
        # Current state variables
        self.current_frame = None
        self.current_position = (0, 0)  # Current position in pixel coordinates
        self.current_heading = 0  # Current heading in radians (0 is forward)
        self.current_speed = 0  # Current speed in m/s
        
        # Path planning variables
        self.obstacle_map = None
        self.current_paths = []
        self.selected_path = None
        self.path_lock = threading.Lock()
        
        # Start path planning thread
        self.running = True
        self.path_thread = threading.Thread(target=self._path_planning_loop)
        self.path_thread.daemon = True
        self.path_thread.start()
        
        # For performance monitoring
        self.frame_times = []
        self.planning_times = []
    
    def set_current_position(self, x, y, heading, speed):
        """Update the current position, heading and speed of the vehicle."""
        self.current_position = (x, y)
        self.current_heading = heading
        self.current_speed = speed
    
    def process_frame(self, frame, pothole_detector):
        frame_start = time.time()
        
        # Store the current frame
        self.current_frame = frame.copy()
        
        # Detect potholes in real-time
        results = pothole_detector(frame)
        
        # Create obstacle map from detected potholes
        height, width = frame.shape[:2]
        obstacle_map = np.zeros((height, width), dtype=np.uint8)
        
        # Draw detected potholes on the frame and mark them in the obstacle map
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                # Draw bounding box on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Mark the pothole and safety margin in the obstacle map
                margin_pixels = int(self.total_margin / self.grid_size)
                cv2.rectangle(obstacle_map, 
                             (max(0, x1 - margin_pixels), max(0, y1 - margin_pixels)),
                             (min(width, x2 + margin_pixels), min(height, y2 + margin_pixels)),
                             255, -1)
        
        # Update obstacle map for path planning thread
        self.obstacle_map = obstacle_map
        
        # Get the latest planned paths (thread-safe)
        with self.path_lock:
            current_paths = self.current_paths.copy()
            selected_path = self.selected_path
        
        # Visualize the paths on the frame
        result_frame = self.visualize_paths(frame, current_paths, selected_path)
        
        # Calculate and display frame processing time
        frame_time = time.time() - frame_start
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 100:
            self.frame_times.pop(0)
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        cv2.putText(result_frame, f"FPS: {fps:.1f}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # If we have a path, execute it
        if selected_path:
            self.execute_path_step(selected_path)
        
        return result_frame
    
    def _path_planning_loop(self):
        """Background thread for continuous path planning."""
        while self.running:
            loop_start = time.time()
            
            # Skip if we don't have obstacle data yet
            if self.obstacle_map is None or self.current_frame is None:
                time.sleep(0.01)
                continue
            
            # Make local copies to avoid race conditions
            obstacle_map = self.obstacle_map.copy()
            current_position = self.current_position
            
            # Calculate goal point based on current position and heading
            height, width = obstacle_map.shape
            lookahead_pixels = int(self.lookahead_distance / self.grid_size)
            
            # Calculate the goal point (directly ahead of current position based on heading)
            goal_x = int(current_position[0] + lookahead_pixels * math.cos(self.current_heading))
            goal_y = int(current_position[1] + lookahead_pixels * math.sin(self.current_heading))
            
            # Ensure goal is within frame boundaries
            goal_x = max(0, min(width - 1, goal_x))
            goal_y = max(0, min(height - 1, goal_y))
            goal_point = (goal_x, goal_y)
            
            # Plan multiple paths
            try:
                paths = self.plan_paths(obstacle_map, current_position, goal_point, num_paths=3)
                
                # Update the paths (thread-safe)
                with self.path_lock:
                    self.current_paths = paths
                    self.selected_path = paths[0][0] if paths else None
                
                # Track planning time
                planning_time = time.time() - loop_start
                self.planning_times.append(planning_time)
                if len(self.planning_times) > 20:
                    self.planning_times.pop(0)
            except Exception as e:
                print(f"Path planning error: {e}")
            
            # Sleep until next update is needed, ensuring minimum frequency
            elapsed = time.time() - loop_start
            sleep_time = max(0, self.path_update_interval - elapsed)
            time.sleep(sleep_time)
        
    def plan_paths(self, obstacle_map, start_point, goal_point, num_paths=3):
        """
        Plan multiple possible paths to avoid potholes in real-time.
        Uses efficient A* implementation with optimizations for speed.
        """
        paths = []
        
        # Create a distance transform for cost calculation
        dist_transform = cv2.distanceTransform(255 - obstacle_map, cv2.DIST_L2, 3)
        dist_transform = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
        
        # Only search in a limited region to improve performance
        height, width = obstacle_map.shape
        search_radius = int(1.5 * self.lookahead_distance / self.grid_size)
        
        # Create a mask for the search region
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, start_point, search_radius, 255, -1)
        
        # A* algorithm with slight randomness to generate diverse paths
        for i in range(num_paths):
            path = self._a_star_search(obstacle_map, dist_transform, start_point, goal_point, 
                                      mask=mask, randomness=0.1 * i)
            if path:
                # Calculate path cost
                cost = self._calculate_path_cost(path, dist_transform)
                paths.append((path, cost))
        
        # Sort paths by cost
        paths.sort(key=lambda x: x[1])
        return paths
    
    def _a_star_search(self, obstacle_map, dist_transform, start, goal, mask=None, randomness=0.0):
        """
        Optimized A* search algorithm for real-time path planning.
        
        Args:
            obstacle_map: Binary map where obstacles are marked as 255
            dist_transform: Distance transform for cost calculation
            start: (x, y) tuple of start position
            goal: (x, y) tuple of goal position
            mask: Optional mask to limit the search area
            randomness: Factor of randomness to generate diverse paths
            
        Returns:
            List of (x, y) coordinates forming a path
        """
        # Define movement directions: 8-connected neighborhood
        # Optimization: pre-compute the movement costs
        directions = [
            ((0, 1), 1.0), ((1, 0), 1.0), ((0, -1), 1.0), ((-1, 0), 1.0),
            ((1, 1), 1.414), ((1, -1), 1.414), ((-1, 1), 1.414), ((-1, -1), 1.414)
        ]
        
        height, width = obstacle_map.shape
        
        # Initialize open and closed sets
        open_set = PriorityQueue()
        open_set.put((0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}
        open_set_hash = {start}
        
        # Optimization: maximum number of nodes to explore
        max_nodes = 1000
        nodes_explored = 0
        
        while not open_set.empty() and nodes_explored < max_nodes:
            current = open_set.get()[1]
            open_set_hash.remove(current)
            nodes_explored += 1
            
            # Early termination if we're close enough to the goal
            if self._heuristic(current, goal) < 5.0:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                
                # Optimization: simplify the path to reduce waypoints
                if len(path) > 3:
                    path = self._simplify_path(path, obstacle_map)
                
                path.reverse()
                return path
            
            for (dx, dy), move_cost in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check if the neighbor is within bounds
                if (neighbor[0] < 0 or neighbor[0] >= width or 
                    neighbor[1] < 0 or neighbor[1] >= height):
                    continue
                    
                # Check if the neighbor is an obstacle
                if obstacle_map[neighbor[1], neighbor[0]] == 255:
                    continue
                
                # Check if the neighbor is in the search mask (if provided)
                if mask is not None and mask[neighbor[1], neighbor[0]] == 0:
                    continue
                
                # Add cost based on proximity to obstacles (inverse of distance transform)
                proximity_cost = (1.0 - dist_transform[neighbor[1], neighbor[0]]) * 5.0
                
                # Add slight randomness to generate diverse paths
                random_cost = np.random.random() * randomness
                
                tentative_g_score = g_score.get(current, float('inf')) + move_cost + proximity_cost + random_cost
                
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self._heuristic(neighbor, goal)
                    if neighbor not in open_set_hash:
                        open_set.put((f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)
        
        # If we didn't find a path to the goal, return the best partial path
        if came_from:
            # Find the node closest to the goal
            best_node = min(came_from.keys(), key=lambda node: self._heuristic(node, goal))
            
            # Reconstruct the partial path
            path = []
            current = best_node
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        
        # No path found at all
        return None
    
    def _simplify_path(self, path, obstacle_map):
        """
        Simplify a path by removing unnecessary waypoints.
        Uses the Ramer-Douglas-Peucker algorithm adapted for obstacle avoidance.
        """
        if len(path) <= 2:
            return path
        
        # Find the point with the maximum deviation
        max_distance = 0
        index = 0
        
        start_point = path[0]
        end_point = path[-1]
        
        for i in range(1, len(path) - 1):
            # Calculate distance from point to line
            distance = self._point_line_distance(path[i], start_point, end_point)
            
            # Check if the direct line crosses obstacles
            if self._line_intersects_obstacle(start_point, end_point, obstacle_map):
                distance = float('inf')  # Force subdivision if obstacles are in the way
            
            if distance > max_distance:
                max_distance = distance
                index = i
        
        # If the maximum distance is greater than our threshold, recursively simplify
        if max_distance > 2.0:
            # Recursively simplify the two segments
            first_segment = self._simplify_path(path[:index + 1], obstacle_map)
            second_segment = self._simplify_path(path[index:], obstacle_map)
            
            # Concatenate the segments, avoiding duplicating the connection point
            return first_segment[:-1] + second_segment
        else:
            # The path is simple enough, just use the endpoints
            return [start_point, end_point]
    
    def _point_line_distance(self, point, line_start, line_end):
        """Calculate the perpendicular distance from a point to a line."""
        # Convert to numpy arrays for vectorized operations
        point = np.array(point)
        line_start = np.array(line_start)
        line_end = np.array(line_end)
        
        # Handle case where line points are the same
        if np.all(line_start == line_end):
            return np.linalg.norm(point - line_start)
        
        # Calculate distance
        line_vec = line_end - line_start
        point_vec = point - line_start
        line_len = np.linalg.norm(line_vec)
        line_unitvec = line_vec / line_len
        point_vec_scaled = point_vec / line_len
        
        t = np.dot(line_unitvec, point_vec_scaled)
        t = max(0, min(1, t))  # Clamp to [0, 1]
        
        nearest = line_start + t * line_vec
        return np.linalg.norm(point - nearest)
    
    def _line_intersects_obstacle(self, p1, p2, obstacle_map):
        """Check if a line between two points intersects any obstacle."""
        # Use Bresenham's line algorithm to check cells along the line
        x1, y1 = p1
        x2, y2 = p2
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        
        err = dx - dy
        
        height, width = obstacle_map.shape
        
        while True:
            # Check if current point is an obstacle
            if (0 <= x1 < width and 0 <= y1 < height and 
                obstacle_map[y1, x1] == 255):
                return True
            
            if x1 == x2 and y1 == y2:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
        
        return False
    
    def _heuristic(self, a, b):
        """Calculate the Euclidean distance heuristic."""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def _calculate_path_cost(self, path, dist_transform):
        """Calculate the cost of a path based on length and proximity to obstacles."""
        if not path or len(path) < 2:
            return float('inf')
            
        cost = 0
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i+1]
            
            # Distance cost
            distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            cost += distance
            
            # Proximity to obstacles cost (higher penalty for being close to obstacles)
            proximity = 1.0 - dist_transform[p2[1], p2[0]]
            cost += proximity * 3.0
            
        return cost
    
    def visualize_paths(self, frame, paths, selected_path=None):
        result_frame = frame.copy()
        
        # Draw the current position and heading
        current_position_int = (int(self.current_position[0]), int(self.current_position[1]))
        cv2.circle(result_frame, current_position_int, 5, (255, 0, 0), -1)
        
        # Calculate endpoint of heading indicator
        heading_length = 20
        heading_end = (
            int(self.current_position[0] + heading_length * math.cos(self.current_heading)),
            int(self.current_position[1] + heading_length * math.sin(self.current_heading))
        )
        cv2.line(result_frame, current_position_int, heading_end, (255, 0, 0), 2)
        
        # Draw alternative paths
        for path, cost in paths:
            if path == selected_path:
                continue  # Skip the selected path for now
                
            color = (0, 255, 255)  # Yellow for alternative paths
            thickness = 1
            
            for j in range(len(path) - 1):
                p1 = (int(path[j][0]), int(path[j][1]))
                p2 = (int(path[j+1][0]), int(path[j+1][1]))
                cv2.line(result_frame, p1, p2, color, thickness)
        
        # Draw the selected path
        if selected_path:
            color = (0, 0, 255)  # Red for best path
            thickness = 2
            
            for j in range(len(selected_path) - 1):
                p1 = (int(selected_path[j][0]), int(selected_path[j][1]))
                p2 = (int(selected_path[j+1][0]), int(selected_path[j+1][1]))
                cv2.line(result_frame, p1, p2, color, thickness)
            
            # Display the next waypoint
            if len(selected_path) > 1:
                next_waypoint = (int(selected_path[1][0]), int(selected_path[1][1]))
                cv2.circle(result_frame, next_waypoint, 5, (0, 255, 0), -1)
        
        # Display average planning time
        if self.planning_times:
            avg_planning_time = sum(self.planning_times) / len(self.planning_times)
            planning_fps = 1.0 / avg_planning_time if avg_planning_time > 0 else 0
            cv2.putText(result_frame, f"Path Planning: {planning_fps:.1f} Hz", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return result_frame

    def execute_path_step(self, path):
        """
        Execute one step along the selected path.
        This function simulates the vehicle moving along the path.
        """
        if len(path) < 2:
            return
        
        # Move to the next waypoint
        next_waypoint = path[1]
        self.current_position = next_waypoint
        
        # Update heading based on the direction to the next waypoint
        dx = next_waypoint[0] - self.current_position[0]
        dy = next_waypoint[1] - self.current_position[1]
        self.current_heading = math.atan2(dy, dx)
        
        # Simulate movement by updating the current position
        self.current_position = next_waypoint

def main():
    # Initialize the real-time pothole avoidance system
    avoidance = RealTimePotholeAvoidance()
    
    # Load the video
    cap = cv2.VideoCapture("output.mp4")
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Define the codec and create VideoWriter object
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter("output_with_path.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    # Load the pothole detector model
    model = YOLO("runs\\detect\\train2\\weights\\best.pt")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame to detect potholes and plan paths
        result_frame = avoidance.process_frame(frame, model)
        
        # Write the frame to the output video
        out.write(result_frame)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()