from flask import Flask
from ultralytics import YOLO
import cv2
import math



app = Flask(__name__)

# Start webcam
cap = cv2.VideoCapture(0)

# Model
model = YOLO('C:\\Users\\hafiz\\Desktop\\newInterface\\best.pt')

# Object classes
classNames = ["apron", "hairnet", "footwear", "no_apron", 'no_hairnet', "no_footwear", "person"]

# Colors for bounding boxes
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 128, 128)]

# Records of detections
detection_records = []

def generate_frames():
    global detection_records, cap

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Resize frame to 320x320
        resized_frame = cv2.resize(frame, (320, 320))

        results = model(resized_frame, stream=True)

        # Record of detections for this instance
        instance_record = []

        # Coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # Class name
                cls = int(box.cls[0])
                class_name = classNames[cls]

                # Confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100

                # Add detection to instance record
                instance_record.append((class_name, confidence))

                # Draw bounding box
                if confidence > 0.6:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    org = (x1, y1 - 10)  # Adjusted to display text above the bounding box
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2
                    text = f"{class_name}: {confidence}"  # Display class name and confidence score
                    cv2.putText(resized_frame, text, org, font, fontScale, color, thickness)


        # Append instance record to detection records
        detection_records.append(instance_record)

        # Yield the frame for video streaming
        ret, buffer = cv2.imencode('.jpg', resized_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
def release_camera():
    global cap
    if cap is not None:
        cap.release()

