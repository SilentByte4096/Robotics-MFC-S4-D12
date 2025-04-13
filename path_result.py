import time
import numpy as np

def evaluate_path_planning(planned_path, actual_path, pothole_positions, grid_size):
    """
    Evaluate the correctness of path planning.
    
    Parameters:
    - planned_path: List of coordinates of the planned path (e.g., [(x1, y1), (x2, y2), ...])
    - actual_path: List of coordinates of the actual path taken
    - pothole_positions: List of pothole coordinates
    - grid_size: Tuple of (width, height) of the grid (for context)
    
    Returns:
    - Success Rate (1 if goal reached without hitting potholes, else 0)
    - Path Length (in grid units)
    - Computation Time (in seconds)
    """
    start_time = time.time()
    
    # Check if the actual path reached the goal
    goal_reached = actual_path[-1] == planned_path[-1]
    
    # Check if any pothole was hit
    pothole_hit = any(pos in pothole_positions for pos in actual_path)
    
    # Success if goal reached without hitting potholes
    success = 1 if goal_reached and not pothole_hit else 0
    
    # Path length (Euclidean distance)
    path_length = sum(np.linalg.norm(np.array(planned_path[i]) - np.array(planned_path[i-1])) 
                      for i in range(1, len(planned_path)))
    
    computation_time = time.time() - start_time
    
    return success, path_length, computation_time

# Example usage (replace with your actual data)
planned_path = [(0, 0), (1, 1), (2, 2)]  # Example planned path
actual_path = [(0, 0), (1, 1), (2, 2)]   # Example actual path
pothole_positions = [(1, 2)]              # Example pothole positions
grid_size = (10, 10)                      # Example grid size
success, path_length, computation_time = evaluate_path_planning(planned_path, actual_path, pothole_positions, grid_size)

# Print results
print("Path Planning Metrics:")
print(f"Success: {success}")
print(f"Path Length: {path_length:.2f} units")
print(f"Computation Time: {computation_time:.4f} seconds")