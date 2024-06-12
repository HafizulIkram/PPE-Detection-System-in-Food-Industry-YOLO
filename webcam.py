import cv2
import torch
from webDetect import initialize_model, process_frame
from PIL import Image
import io
import numpy as np

# Initialize the model once
class Opt:
    def __init__(self):
<<<<<<< HEAD
        self.weights = 'yolov7W6.pt'
=======
        self.weights = 'YoloWeight\yolov7W6.pt'
>>>>>>> fa949f9 (newYOLO)
        self.img_size = 320
        self.device = ''
        self.no_trace = False
        self.conf_thres = 0.6
        self.iou_thres = 0.6

opt = Opt()

model, modelc, device, half, imgsz = initialize_model(opt)
cap = None

def initialize_webcam():
    global cap
    cap = cv2.VideoCapture(0)

def release_webcam():
    global cap
    if cap is not None:
        cap.release()
        cap = None

def generate_frames():
    initialize_webcam()
    if not cap.isOpened():
        raise RuntimeError("Could not start webcam.")

    while True:
        if cap is None or not cap.isOpened():
            break
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, detected_classes = process_frame(model, modelc, device, half, imgsz, frame, opt)
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            continue

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def get_detected_classes():
    initialize_webcam()
    if not cap.isOpened():
        return []

    ret, frame = cap.read()
    if not ret:
        return []

    _, classes = process_frame(model, modelc, device, half, imgsz, frame, opt)
    return classes
