import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time

# Check CUDA availability
if cv2.cuda.getCudaEnabledDeviceCount() == 0:
    raise RuntimeError("No CUDA device found. Ensure OpenCV is built with CUDA support.")
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch CUDA not available. Ensure PyTorch is installed with CUDA support.")

# Parameters
EMA_ALPHA = 0.5  # Smoothing factor (0-1)
MATCH_METHOD = cv2.TM_SQDIFF  # Faster for Jetson
SEARCH_EXPAND_FACTOR = 1.2  # Smaller ROI for speed
FRAME_SIZE = (480, 360)  # Optimized for Jetson Orin

# Initialize YOLOv8
model = YOLO("yolov8n.pt").to("cuda").half()  # Half-precision for Jetson

# Initialize video capture (webcam or Jetson CSI camera)
cap = cv2.VideoCapture(0)  # Or "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGR ! appsink"
if not cap.isOpened():
    raise RuntimeError("Could not open video source.")

# Persistent GPU objects
gpu_frame = cv2.cuda_GpuMat()
gpu_template = cv2.cuda_GpuMat()
gpu_result = cv2.cuda_GpuMat()
matcher = cv2.cuda.createTemplateMatching(cv2.CV_8U, MATCH_METHOD)
stream = cv2.cuda.Stream()

# Variables
smoothed_box = None
template = None
frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Downscale frame for speed
    frame = cv2.resize(frame, FRAME_SIZE)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # YOLO detection
    results = model(frame, device="cuda", verbose=False, conf=0.25)
    detected_box = None
    if results[0].boxes:
        box = results[0].boxes[0].xywh.cpu().numpy()
        x = int(box[0] - box[2] / 2)
        y = int(box[1] - box[3] / 2)
        w, h = int(box[2]), int(box[3])
        detected_box = [x, y, w, h]

    # Smooth the box
    if detected_box:
        x, y, w, h = detected_box
        if smoothed_box is not None:
            sx, sy, sw, sh = smoothed_box
            x = EMA_ALPHA * x + (1 - EMA_ALPHA) * sx
            y = EMA_ALPHA * y + (1 - EMA_ALPHA) * sy
            w = EMA_ALPHA * w + (1 - EMA_ALPHA) * sw
            h = EMA_ALPHA * h + (1 - EMA_ALPHA) * sh
        smoothed_box = [int(x), int(y), int(w), int(h)]

        # Extract template
        if template is None and frame_count < 10:
            template = gray[smoothed_box[1]:smoothed_box[1]+smoothed_box[3],
                            smoothed_box[0]:smoothed_box[0]+smoothed_box[2]]
            if template.size == 0 or template.shape[0] < 10 or template.shape[1] < 10:
                continue
            gpu_template.upload(template, stream)

    # Template matching
    if template is not None and smoothed_box is not None:
        search_x = max(0, smoothed_box[0] - int((SEARCH_EXPAND_FACTOR - 1) * smoothed_box[2] / 2))
        search_y = max(0, smoothed_box[1] - int((SEARCH_EXPAND_FACTOR - 1) * smoothed_box[3] / 2))
        search_w = min(gray.shape[1] - search_x, int(smoothed_box[2] * SEARCH_EXPAND_FACTOR))
        search_h = min(gray.shape[0] - search_y, int(smoothed_box[3] * SEARCH_EXPAND_FACTOR))
        search_roi = gray[search_y:search_y+search_h, search_x:search_x+search_w]

        if search_roi.size == 0 or template.shape[0] > search_roi.shape[0] or template.shape[1] > search_roi.shape[1]:
            continue

        gpu_frame.upload(search_roi, stream)
        matcher.match(gpu_frame, gpu_template, gpu_result, stream=stream)
        stream.waitForCompletion()

        result = gpu_result.download()
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        top_left = min_loc if MATCH_METHOD in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] else max_loc
        matched_box = [search_x + top_left[0], search_y + top_left[1], smoothed_box[2], smoothed_box[3]]
        smoothed_box = matched_box

        cv2.rectangle(frame, (smoothed_box[0], smoothed_box[1]),
                      (smoothed_box[0] + smoothed_box[2], smoothed_box[1] + smoothed_box[3]), (0, 255, 0), 2)

    # FPS calculation
    frame_count += 1
    if frame_count % 30 == 0:
        fps = frame_count / (time.time() - start_time)
        print(f"FPS: {fps:.2f}")
        frame_count = 0
        start_time = time.time()

    cv2.imshow("YOLO + CUDA Template Matching", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()