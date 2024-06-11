import torch
import cv2
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
# Load the YOLOv7 model

# Load the YOLOv7 model
weights_path = 'C:\\Users\\hafiz\\Desktop\\newInterface\\YoloWeight\\yolov7W6.pt'
device = select_device('')
model = attempt_load(weights_path, map_location=device)  # load FP32 model
model.eval()  # set model to evaluation mode

def detect_cam(frame, confidence_threshold):
    # Resize frame to 320x320
    img_size = 320
    frame_resized = cv2.resize(frame, (img_size, img_size))
    
    # Convert frame to tensor
    img = frame_resized[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x320x320
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0  # Normalize to [0, 1]
    if img.ndimension() == 3:
        img = img.unsqueeze(0)  # Add batch dimension

    # Perform object detection with resized frame
    with torch.no_grad():
        pred = model(img)[0]

    # Apply NMS (Non-Maximum Suppression)
    pred = non_max_suppression(pred, confidence_threshold, iou_thres= 0.5)

    detected_classes = []
    predictions = []
    # Draw bounding boxes and labels on the original frame
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                confidence = float(conf)
                class_id = int(cls)
                label = f'{model.names[class_id]}: {confidence:.2f}'

                detected_classes.append(model.names[class_id])
                bbox = [x1, y1, x2 - x1, y2 - y1]  # Convert to (x, y, width, height) format
                predictions.append([class_id, x1, y1, x2 - x1, y2 - y1])

                # Draw rectangle and label on the original frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    class_list = model.names
    status = classify_completion(predictions, class_list)

    return frame, detected_classes, status

def classify_completion(predictions, class_list):
    person_bbox = None
    required_items = {'apron', 'hairnet', 'footwear', 'person'}
    detected_items = set()
    iou_threshold = 0.6  # Adjust as needed
    iou_values = []

    # Store detected items before person detection
    stored_items = {}

    # Iterate through predictions to classify labels and extract bounding boxes
    for prediction in predictions:
        label_index = int(prediction[0])  # Extract the class label index
        bbox = [float(coord) for coord in prediction[1:]]  # Extract bounding box coordinates

        # Map class index to class label
        class_label = class_list[label_index]

        # Check if the label is 'person'
        if class_label == 'person':
            person_bbox = bbox
            detected_items.add(class_label)

            # Calculate IOU with previously detected items
            for item_label, stored_bbox in stored_items.items():
                iou = calculate_iou(person_bbox, stored_bbox)
                iou_values.append(f"IOU between person and {item_label}: {iou:.2f}")
                if item_label == 'hairnet':
                    iou += 0.25
                if iou >= iou_threshold:
                    detected_items.add(item_label)

        # Check if the label is in the required items
        if class_label in required_items:
            # If person_bbox is not detected yet, store the item for later calculation
            if person_bbox is None:
                stored_items[class_label] = bbox
            else:
                # Calculate IOU with person bbox
                iou = calculate_iou(person_bbox, bbox)
                iou_values.append(f"IOU between person and {class_label}: {iou:.2f}")
                if class_label == 'hairnet':
                    iou += 0.25
                if iou >= iou_threshold:
                    detected_items.add(class_label)

    if 'person' not in detected_items:
        return "background"
    elif detected_items == required_items:
        return "complete"
    elif detected_items == {'hairnet', 'apron'}:
        return "partial complete"
    else:
        return "not complete"

# Function to calculate IOU between two bounding boxes
def calculate_iou(box1, box2):
    # Convert bounding box coordinates to (x1, y1, x2, y2) format
    box1_coords = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
    box2_coords = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]

    # Calculate intersection coordinates
    x_left = max(box1_coords[0], box2_coords[0])
    y_top = max(box1_coords[1], box2_coords[1])
    x_right = min(box1_coords[2], box2_coords[2])
    y_bottom = min(box1_coords[3], box2_coords[3])

    # Calculate intersection area
    intersection_area = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)

    # Calculate areas of both bounding boxes
    box1_area = (box1_coords[2] - box1_coords[0] + 1) * (box1_coords[3] - box1_coords[1] + 1)
    box2_area = (box2_coords[2] - box2_coords[0] + 1) * (box2_coords[3] - box2_coords[1] + 1)

    # Calculate union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate IOU
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou



