import cv2
import numpy as np
import networkx as nx
from heapq import heappop, heappush

def detect_potholes(frame):
    """Detects potholes in the given frame using color segmentation."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    potholes = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 500]
    return potholes

def construct_graph(frame, potholes):
    """Constructs a graph representing the road avoiding potholes."""
    G = nx.grid_2d_graph(frame.shape[0] // 10, frame.shape[1] // 10)
    
    for x, y, w, h in potholes:
        for i in range(y // 10, (y + h) // 10):
            for j in range(x // 10, (x + w) // 10):
                if (i, j) in G:
                    G.remove_node((i, j))  
    return G

def heuristic(a, b):
    """Heuristic function for A* (Euclidean distance)."""
    return np.linalg.norm(np.array(a) - np.array(b))

def a_star(graph, start, goal):
    """A* pathfinding algorithm."""
    queue = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    while queue:
        _, current = heappop(queue)
        if current == goal:
            break
        
        for neighbor in graph.neighbors(current):
            new_cost = cost_so_far[current] + 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(goal, neighbor)
                heappush(queue, (priority, neighbor))
                came_from[neighbor] = current
    
    path = []
    while current:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path

def main():
    cap = cv2.VideoCapture("20250307-0413-18.0910669.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        potholes = detect_potholes(frame)
        graph = construct_graph(frame, potholes)
        
        start, goal = (5, 5), (frame.shape[0] // 10 - 5, frame.shape[1] // 10 - 5)
        path = a_star(graph, start, goal)
        
        for x, y, w, h in potholes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Draw potholes
        
        for y, x in path:
            cv2.circle(frame, (x * 10, y * 10), 3, (0, 0, 255), -1)  # Draw best path
        
        out.write(frame)  # Save frame to video
        cv2.imshow("Real-Time Path Planning", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
