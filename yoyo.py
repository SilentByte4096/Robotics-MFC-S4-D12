import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import math

def non_max_suppression(boxes, scores, threshold=0.2):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    indices = np.argsort(scores)[::-1]
    keep = []
    while len(indices) > 0:
        i = indices[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[indices[1:]])
        yy1 = np.maximum(y1[i], y1[indices[1:]])
        xx2 = np.minimum(x2[i], x2[indices[1:]])
        yy2 = np.minimum(y2[i], y2[indices[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        intersection = w * h
        union = areas[i] + areas[indices[1:]] - intersection
        iou = intersection / union
        indices = indices[1:][iou < threshold]
    return keep

def preprocess_image_with_edges(frame):
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    smoothed = cv2.bilateralFilter(gray, 7, 50, 50)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(smoothed)
    edges = cv2.Canny(enhanced, 5, 20)
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    return frame, edges, enhanced

def detect_potential_regions_of_interest(enhanced_frame, edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    potential_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 30:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if 0.05 < circularity < 1.5:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h if h > 0 else 0
                    if 0.1 < aspect_ratio < 5.0:
                        roi_edges = edges[y:y+h, x:x+w]
                        edge_density = np.sum(roi_edges > 0) / (w * h) if w * h > 0 else 0
                        if edge_density > 0.03:
                            potential_regions.append((x, y, x+w, y+h))
    if potential_regions:
        scores = [1.0] * len(potential_regions)
        keep_indices = non_max_suppression(potential_regions, scores, 0.15)
        potential_regions = [potential_regions[i] for i in keep_indices]
    return potential_regions

def apply_texture_analysis(enhanced_frame, box):
    x1, y1, x2, y2 = box
    roi = enhanced_frame[y1:y2, x1:x2]
    if roi.size == 0:
        return 0
    texture_score = 0
    try:
        std_dev = np.std(roi)
        gradient_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_mean = np.mean(gradient_magnitude)
        gradient_var = np.var(gradient_magnitude)
        hist = cv2.calcHist([roi], [0], None, [32], [0, 256])
        hist_normalized = hist / hist.sum() if hist.sum() > 0 else hist
        hist_normalized = hist_normalized[hist_normalized > 0]
        entropy = -np.sum(hist_normalized * np.log2(hist_normalized)) if hist_normalized.size > 0 else 0
        lbp_score = 0
        if roi.shape[0] > 2 and roi.shape[1] > 2:
            for i in range(1, roi.shape[0]-1):
                for j in range(1, roi.shape[1]-1):
                    center = roi[i, j]
                    neighbors = [roi[i-1, j-1], roi[i-1, j], roi[i-1, j+1],
                                roi[i, j-1], roi[i, j+1],
                                roi[i+1, j-1], roi[i+1, j], roi[i+1, j+1]]
                    lbp_score += sum(1 for n in neighbors if n > center)
            lbp_score /= ((roi.shape[0]-2) * (roi.shape[1]-2) * 8)
        edge_kernel = np.ones((2, 2), np.uint8)
        edges = cv2.Canny(roi.astype(np.uint8), 5, 25)
        edges = cv2.dilate(edges, edge_kernel, iterations=1)
        edge_density = np.sum(edges > 0) / edges.size if edges.size > 0 else 0
        texture_score = (std_dev * 0.1 + gradient_mean * 0.2 + gradient_var * 0.05 + 
                        entropy * 8 + lbp_score * 15 + edge_density * 20)
        if std_dev < 3 or gradient_mean < 1:
            texture_score *= 0.9
        if edge_density > 0.05:
            texture_score *= 1.4
    except Exception as e:
        print(f"Error in texture analysis: {e}")
        texture_score = 0
    return texture_score

def estimate_pothole_dimensions(box_coords, image_shape, focal_length=300, real_width_reference=0.2, camera_height=0.3):
    x1, y1, x2, y2 = box_coords
    pixel_width = max(1, x2 - x1)
    pixel_height = max(1, y2 - y1)
    image_height = image_shape[0]
    y_center = (y1 + y2) / 2

    base_distance = (real_width_reference * focal_length) / pixel_width
    horizon_factor = (image_height - y_center) / image_height
    if horizon_factor > 0.7:
        distance_meters = base_distance * (1.0 / (horizon_factor * 2.0))
    elif horizon_factor > 0.3:
        distance_meters = base_distance * (1.0 / horizon_factor)
    else:
        distance_meters = base_distance * (1.0 + (1.0 - horizon_factor) * 1.0)
    angle_rad = math.radians(45 - (horizon_factor * 25))
    height_adjusted_distance = camera_height / math.tan(max(angle_rad, math.radians(20)))
    distance_meters = min(distance_meters, height_adjusted_distance * 1.5)

    width_m = (real_width_reference * distance_meters * pixel_width) / (focal_length * 1.5)
    height_m = width_m * (pixel_height / pixel_width)
    perspective_factor = 1.0 + (y_center / image_height) * 0.1
    width_m *= perspective_factor
    height_m *= perspective_factor

    distance_meters = max(0.1, min(distance_meters, 5.0))
    width_m = max(0.05, min(width_m, 0.5))
    height_m = max(0.05, min(height_m, 0.5))

    return round(width_m * 100, 1), round(height_m * 100, 1), round(distance_meters, 2)

def detect_pothole_in_image(model, image_path, confidence_threshold=0.25, focal_length=300, real_width_reference=0.2, camera_height=0.3, nms_threshold=0.2):
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not read image at {image_path}")
        return
    image_height, image_width = original_image.shape[:2]
    original_image, edges, enhanced = preprocess_image_with_edges(original_image)
    potential_regions = detect_potential_regions_of_interest(enhanced, edges)
    results = model(image_path, conf=0.02)
    rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    all_boxes = []
    all_scores = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf.cpu().numpy()[0]
            y_center = (y1 + y2) / 2
            distance_factor = 1.0 + (y_center / image_height) * 1.2
            adjusted_conf = min(conf * distance_factor, 1.0)
            if adjusted_conf >= confidence_threshold:
                all_boxes.append((x1, y1, x2, y2))
                all_scores.append(adjusted_conf)
    if all_boxes:
        keep_indices = non_max_suppression(all_boxes, all_scores, nms_threshold)
        filtered_boxes = [all_boxes[i] for i in keep_indices]
        filtered_scores = [all_scores[i] for i in keep_indices]
    else:
        filtered_boxes = []
        filtered_scores = []

    pothole_detected = False
    for i, (box, score) in enumerate(zip(filtered_boxes, filtered_scores)):
        x1, y1, x2, y2 = box
        pothole_detected = True
        print(f"Pothole {i+1} Detected with confidence {score:.2f}!")
        width_cm, height_cm, distance_m = estimate_pothole_dimensions(
            (x1, y1, x2, y2), (image_height, image_width), focal_length, real_width_reference, camera_height)

        if distance_m < 0.5:
            box_color = (255, 0, 0)  # Red for close
            warning = "CLOSE"
        else:
            box_color = (255, 165, 0)  # Orange for farther
            warning = "FAR"

        cv2.rectangle(rgb_image, (x1, y1), (x2, y2), box_color, 2)
        text_id = f'ID: {i}'
        text_conf = f'Conf: {score:.2f}'
        text_dist = f'Dist: {distance_m}m - {warning}'
        text_dim = f'Size: {width_cm}cm x {height_cm}cm'
        cv2.putText(rgb_image, text_id, (x1, y1-40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, box_color, 1)
        cv2.putText(rgb_image, text_conf, (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, box_color, 1)
        cv2.putText(rgb_image, text_dist, (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, box_color, 1)
        cv2.putText(rgb_image, text_dim, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, box_color, 1)

    if not pothole_detected and not potential_regions:
        print("No Pothole Detected.")
        cv2.putText(rgb_image, 'No Pothole', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    plt.figure(figsize=(10, 8))
    plt.imshow(rgb_image)
    plt.title('Pothole Detection Result')
    plt.axis('off')
    plt.show()
    output_path = 'pothole_detection_result.jpg'
    cv2.imwrite(output_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    print(f"Annotated image saved as {output_path}")

def detect_potholes_in_video(model, video_path, output_path='pothole_detection_output.mp4', 
                            confidence_threshold=0.25, focal_length=300, real_width_reference=0.2, 
                            camera_height=0.3, nms_threshold=0.2, process_every_n_frames=1):
    print("Step 1: Opening video file...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    print("Step 2: Retrieving video properties...")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    try:
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    except Exception as e:
        print(f"Error creating VideoWriter: {e}")
        print("Trying alternative codec...")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        print(f"Error: Could not open output video file at {output_path}")
        cap.release()
        return

    frame_count = 0

    while True:
        print(f"Step 6: Reading frame {frame_count + 1}...")
        ret, frame = cap.read()
        if not ret:
            print("Step 6: End of video reached.")
            break

        frame_count += 1
        print(f"Processing frame {frame_count}")

        # Process every frame (no persistent tracking across frames)
        print("Step 7.1: Preprocessing frame with edge detection...")
        original_frame, edges, enhanced = preprocess_image_with_edges(frame)
        print("Step 7.2: Detecting potential pothole regions...")
        potential_regions = detect_potential_regions_of_interest(enhanced, edges)

        print("Step 7.3: Running YOLO detection...")
        results = model(frame, conf=0.02)
        print("Step 7.4: Extracting detections...")
        all_boxes = []
        all_scores = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf.cpu().numpy()[0]
                y_center = (y1 + y2) / 2
                distance_factor = 1.0 + (y_center / frame_height) * 1.2
                adjusted_conf = min(conf * distance_factor, 1.0)
                if adjusted_conf >= confidence_threshold:
                    all_boxes.append((x1, y1, x2, y2))
                    all_scores.append(adjusted_conf)

        print("Step 7.5: Applying non-maximum suppression...")
        if all_boxes:
            keep_indices = non_max_suppression(all_boxes, all_scores, nms_threshold)
            filtered_boxes = [all_boxes[i] for i in keep_indices]
            filtered_scores = [all_scores[i] for i in keep_indices]
        else:
            filtered_boxes = []
            filtered_scores = []

        print("Step 8: Drawing detections...")
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()
        pothole_detected = False
        for i, (box, score) in enumerate(zip(filtered_boxes, filtered_scores)):
            x1, y1, x2, y2 = box
            pothole_detected = True
            print(f"Frame {frame_count}: Pothole {i+1} Detected with confidence {score:.2f}!")
            width_cm, height_cm, distance_m = estimate_pothole_dimensions(
                (x1, y1, x2, y2), (frame_height, frame_width), focal_length, real_width_reference, camera_height)

            if distance_m < 0.5:
                box_color = (255, 0, 0)  # Red for close
                warning = "CLOSE"
            else:
                box_color = (255, 165, 0)  # Orange for farther
                warning = "FAR"

            cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), box_color, 2)
            text_id = f'ID: {i}'
            text_conf = f'Conf: {score:.2f}'
            text_dist = f'Dist: {distance_m}m - {warning}'
            text_dim = f'Size: {width_cm}cm x {height_cm}cm'
            cv2.putText(rgb_frame, text_id, (x1, y1-40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, box_color, 1)
            cv2.putText(rgb_frame, text_conf, (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, box_color, 1)
            cv2.putText(rgb_frame, text_dist, (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, box_color, 1)
            cv2.putText(rgb_frame, text_dim, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, box_color, 1)

        if not pothole_detected and not potential_regions:
            print(f"Frame {frame_count}: No Pothole Detected.")
            cv2.putText(rgb_frame, 'No Pothole', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        print("Step 9: Converting to BGR and writing to output video...")
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)

        print("Step 10: Displaying frame... (Disabled due to GUI error)")

    print("Step 11: Releasing resources...")
    cap.release()
    out.release()
    print(f"Video processing complete. Output saved as {output_path}")

if __name__ == "__main__":
    model_path = r"C:\Users\nehar\Downloads\really\My_attempt\needed\best.pt"  # Replace with your model path
    image_path = r"C:\Users\nehar\Downloads\really\My_attempt\needed\pothole.jpg"  # Replace with your image path
    video_path = r"C:\Users\nehar\Downloads\really\My_attempt\needed\demo.mp4"  # Replace with your video path

    model = YOLO(model_path)
    
    # Example usage for image
    detect_pothole_in_image(model, image_path)
    
    # Example usage for video
    detect_potholes_in_video(model, video_path)