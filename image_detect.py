import os
from subprocess import Popen
from flask import render_template

def predict_img(request):
    class_labels = []  # Initialize an empty list to store predicted classes
    if 'file' in request.files:
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)

        file_extension = f.filename.rsplit('.', 1)[1].lower()    
        if file_extension == 'jpg':
            process = Popen(["python", "detect.py", '--source', filepath, "--weights","YoloWeight\yolov7W6.pt", "--save-txt"], shell=True)
            process.wait()
        elif file_extension == 'mp4':
            process = Popen(["python", "detect.py", '--source', filepath, "--weights","YoloWeight\yolov7W6.pt", "--save-txt", "True"], shell=True)
            process.communicate()
            process.wait()
        
        # Paths for both uploaded image and predicted image
        uploaded_image_path = "/uploads/" + f.filename

        # Get the latest predicted subfolder path
        latest_subfolder_path = get_latest_predicted_subfolder_path()

        predicted_image_path = os.path.join(latest_subfolder_path, os.path.basename(f.filename))

        # Split the filename into its base name and extension
        base_name, extension = os.path.splitext(f.filename)

        # Change the extension to ".txt"
        new_filename = base_name + ".txt"
        # Example file path to YOLO predictions
        file_path = os.path.join(latest_subfolder_path, "labels", new_filename)

        # Print the predicted classes for verification
        print("filepath ", file_path)
        class_list = ['apron', 'hairnet', 'footwear', 'no_apron', 'no_hairnet', 'no_footwear', 'person']  # Example class labels
        
        # Read YOLO predictions from the file
        predictions = read_predictions(file_path)

        # Print the predicted classes for verification
        print(predictions)

        # Map class indices to class labels
        class_labels = map_class_indices_to_labels(predictions, class_list)

        completion_status = classify_completion(predictions, class_list)
        print("status", completion_status)  # Output: complete

        # Render the template with uploaded image path, predicted image path, and latest subfolder path
        return render_template('index.html', uploaded_image_path=uploaded_image_path, predicted_image_path= predicted_image_path, class_labels = class_labels, completion_status = completion_status)

def get_latest_predicted_subfolder_path():
    folder_path = 'runs\detect'
    if not os.path.exists(folder_path):
        return None
    
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    if not subfolders:
        return None
    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    return os.path.join(folder_path, latest_subfolder)

def map_class_indices_to_labels(predictions, class_list):
    class_labels = []
    class_occurrences = {}  # Dictionary to store occurrences of each class
    
    for prediction in predictions:
        if prediction:  # Check if the prediction list is not empty
            # Join the elements of the prediction sublist with spaces
            prediction_string = ' '.join(prediction)
            
            # Split the prediction line by whitespace
            tokens = prediction_string.strip().split()
            
            # Get the first token
            first_token = tokens[0]

            # Check if the first token is a valid number
            if first_token.isdigit():
                class_index = int(first_token)

                # Check if the class index is within the range of the class list
                if 0 <= class_index < len(class_list):
                    class_label = class_list[class_index]
                    class_labels.append(class_label)
                    
                    # Update the occurrences of the class in the dictionary
                    class_occurrences[class_label] = class_occurrences.get(class_label, 0) + 1
                else:
                    class_label = 'Unknown'
                    class_labels.append(class_label)
                    
                                        # Update the occurrences of the unknown class
                    class_occurrences[class_label] = class_occurrences.get(class_label, 0) + 1
            else:
                print(f"Invalid class index: {first_token}. Skipping...")
        else:
            print("Empty line. Skipping...")

    # Create a list to store formatted class labels with occurrences
    formatted_labels = []
    for label, count in class_occurrences.items():
        formatted_labels.append(f"{count} {label}")
    
    print("format", formatted_labels)
    return formatted_labels

# Read the predictions from the YOLO format text file
def read_predictions(file_path):
    with open(file_path, 'r') as file:
        predictions = [line.strip().split() for line in file]
    return predictions

def classify_completion(predictions, class_list):
    person_bbox = None
    required_items = {'apron', 'footwear', 'hairnet'}
    detected_items = set()
    iou_threshold = 0.5  # Adjust as needed

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

            # Calculate IOU with previously detected items
            for item_label, stored_bbox in stored_items.items():
                print("label", item_label)
                iou = calculate_iou(person_bbox, stored_bbox)
                print("iou before", iou)
                if item_label == 'hairnet':
                    iou = iou + 0.25
                if iou >= iou_threshold:
                    detected_items.add(item_label)
                    print("detect b", detected_items)

        # Check if the label is in the required items
        if class_label in required_items:
            # If person_bbox is not detected yet, store the item for later calculation
            if person_bbox is None:
                stored_items[class_label] = bbox
                print("stored_itmem", stored_items)
            else:
                # Calculate IOU with person bbox
                iou = calculate_iou(person_bbox, bbox)
                if class_label == 'hairnet':
                    iou = iou + 0.25
                if iou >= iou_threshold:
                    print("iou after", iou)
                    detected_items.add(class_label)
                    print("detect after", detected_items)

    if detected_items == required_items:
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
