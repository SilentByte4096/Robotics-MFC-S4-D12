import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_segmentation_metrics(true_labels, pred_labels, num_classes):
    """
    Calculate IoU and Pixel Accuracy for segmentation.
    
    Parameters:
    - true_labels: Ground truth labels (numpy array)
    - pred_labels: Predicted labels (numpy array)
    - num_classes: Number of classes in segmentation
    
    Returns:
    - IoU per class and mean IoU
    - Pixel Accuracy
    """
    # Flatten the arrays
    true_labels = true_labels.flatten()
    pred_labels = pred_labels.flatten()
    
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=range(num_classes))
    
    # Calculate IoU for each class
    iou_per_class = []
    for i in range(num_classes):
        intersection = cm[i, i]
        union = np.sum(cm[i, :]) + np.sum(cm[:, i]) - intersection
        iou = intersection / union if union > 0 else 0
        iou_per_class.append(iou)
    
    # Mean IoU
    mean_iou = np.mean(iou_per_class)
    
    # Pixel Accuracy
    correct_pixels = np.sum(np.diag(cm))
    total_pixels = np.sum(cm)
    pixel_accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0
    
    return iou_per_class, mean_iou, pixel_accuracy

# Example usage (replace with your actual data)
true_labels = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 0]])  # Example ground truth
pred_labels = np.array([[0, 0, 1], [1, 0, 0], [1, 1, 0]])  # Example predictions
num_classes = 2  # Example: road (0) and non-road (1)
iou_per_class, mean_iou, pixel_accuracy = calculate_segmentation_metrics(true_labels, pred_labels, num_classes)

# Print results
print("Segmentation Model Metrics:")
print(f"IoU (Class 0 - Road): {iou_per_class[0]:.4f}")
print(f"IoU (Class 1 - Non-Road): {iou_per_class[1]:.4f}")
print(f"Mean IoU: {mean_iou:.4f}")
print(f"Pixel Accuracy: {pixel_accuracy:.4f}")