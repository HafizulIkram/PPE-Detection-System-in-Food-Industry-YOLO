import os

def get_latest_predicted_subfolder_path():
    folder_path = 'runs/detect'
    if not os.path.exists(folder_path):
        return None
    
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    if not subfolders:
        return None
    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    return os.path.join(folder_path, latest_subfolder)

def read_predictions(file_path):
    with open(file_path, 'r') as file:
        predictions = [line.strip().split() for line in file]
    return predictions

def map_class_indices_to_labels(predictions, class_list):
    class_labels = []
    class_occurrences = {}
    
    for prediction in predictions:
        if prediction:
            prediction_string = ' '.join(prediction)
            tokens = prediction_string.strip().split()
            first_token = tokens[0]

            if first_token.isdigit():
                class_index = int(first_token)
                if 0 <= class_index < len(class_list):
                    class_label = class_list[class_index]
                    class_labels.append(class_label)
                    class_occurrences[class_label] = class_occurrences.get(class_label, 0) + 1
                else:
                    class_label = 'Unknown'
                    class_labels.append(class_label)
                    class_occurrences[class_label] = class_occurrences.get(class_label, 0) + 1
            else:
                print(f"Invalid class index: {first_token}. Skipping...")
        else:
            print("Empty line. Skipping...")

    formatted_labels = [f"{count} {label}" for label, count in class_occurrences.items()]
    print("format", formatted_labels)
    return formatted_labels

def classify_completion(predictions, class_list):
    person_bbox = None
    required_items = {'apron', 'footwear', 'hairnet'}
    detected_items = set()
    iou_threshold = 0.5

    stored_items = {}

    for prediction in predictions:
        label_index = int(prediction[0])
        bbox = [float(coord) for coord in prediction[1:]]
        class_label = class_list[label_index]

        if class_label == 'person':
            person_bbox = bbox

            for item_label, stored_bbox in stored_items.items():
                print("label", item_label)
                iou = calculate_iou(person_bbox, stored_bbox)
                if item_label == 'hairnet':
                    iou += 0.25
                if iou >= iou_threshold:
                    detected_items.add(item_label)

        if class_label in required_items:
            if person_bbox is None:
                stored_items[class_label] = bbox
            else:
                iou = calculate_iou(person_bbox, bbox)
                if class_label == 'hairnet':
                    iou += 0.25
                if iou >= iou_threshold:
                    detected_items.add(class_label)

    if detected_items == required_items:
        return "complete"
    elif detected_items == {'hairnet', 'apron'}:
        return "partial complete"
    else:
        return "not complete"

def calculate_iou(box1, box2):
    box1_coords = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
    box2_coords = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]

    x_left = max(box1_coords[0], box2_coords[0])
    y_top = max(box1_coords[1], box2_coords[1])
    x_right = min(box1_coords[2], box2_coords[2])
    y_bottom = min(box1_coords[3], box2_coords[3])

    intersection_area = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)
    box1_area = (box1_coords[2] - box1_coords[0] + 1) * (box1_coords[3] - box1_coords[1] + 1)
    box2_area = (box2_coords[2] - box2_coords[0] + 1) * (box2_coords[3] - box2_coords[1] + 1)

    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou
