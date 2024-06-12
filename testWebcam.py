import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov7lBest.pt")

# Adjust confidence threshold if necessary
model.conf = 0.65  # Set the confidence threshold to a reasonable value

# Open the webcam
cap = cv2.VideoCapture(1)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the videoqqqqq
    success, frame = cap.read()
    
    if success:
        # Preprocess the frame if necessary (resize, normalize, etc.)
        frame_resized = cv2.resize(frame, (640, 640))  # Resize to model's expected input size
        
        # Run YOLOv8 inference on the frame
        results = model(frame_resized, imgsz=320, conf=0.65, iou = 0.75)
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
