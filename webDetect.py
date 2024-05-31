import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

def initialize_model(opt):
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(opt.img_size, s=stride)  # check img_size

    if not opt.no_trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
    else:
        modelc = None

    return model, modelc, device, half, imgsz

def process_frame(model, modelc, device, half, imgsz, frame, opt):
    class_names = ['apron', 'hairnet', 'footwear', 'no_apron', 'no_hairnet', 'no_footwear', 'person']
    colors = {
        'apron': (0, 255, 0),  # Green
        'hairnet': (255, 0, 0),  # Blue
        'footwear': (0, 0, 255),  # Red
        'no_apron': (255, 255, 0),  # Cyan
        'no_hairnet': (0, 255, 255),  # Yellow
        'no_footwear': (255, 0, 255),  # Magenta
        'person': (255, 102, 0)  # Orange
    }

    img = cv2.resize(frame, (imgsz, imgsz))
    img = img[..., ::-1].copy()  # BGR to RGB and make a copy to ensure positive strides
    img = img.transpose(2, 0, 1)  # Change to [3, imgsz, imgsz]
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    with torch.no_grad():
        pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres= 0.6, iou_thres= 0.6, classes=None, agnostic=False, multi_label=False)

    detected_classes = []

    # Process detection
    for det in pred:  # detections per image
        im0 = frame.copy()
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                cls = int(cls)
                label = f'{class_names[cls]} {conf:.2f}'
                detected_classes.append(class_names[cls])
                color = colors[class_names[cls]]
                plot_one_box(xyxy, im0, color=color, line_thickness=1)
                # Draw label with black color
                cv2.putText(im0, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


    return im0, detected_classes
