import streamlit as st
import numpy as np
import cv2
import math
import cvzone
from ultralytics import YOLO
from old.sort import Sort
import time

# Initialize YOLO model
model = YOLO("yolov8l.pt")
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Streamlit file uploader
st.title("Traffic Video Processing: Using Signal Time Optimization Technique")
st.write("Upload a video file. Supports up to 1GB.")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Save uploaded video temporarily
    video_path = f"./temp_video.{uploaded_file.name.split('.')[-1]}"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())
    
    st.video(video_path)

    # Open video
    cap = cv2.VideoCapture(video_path)

    # Initialize Sort Tracker
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    # Lane coordinates (adjusted for your logic)
    limits = {
        1: [(530, 400), (1480, 400)],  # Horizontal line for lane 1
        2: [(530, 440), (1480, 440)],  # Horizontal line for lane 2
        3: [(1350, 140), (1350, 380)],  # Vertical line for lane 3
        4: [(1390, 140), (1390, 380)],  # Vertical line for lane 4
    }

    # Total count per lane
    totalCounts = {1: [], 2: [], 3: [], 4: []}

    # Streamlit buttons for play and pause
    pause = st.button("Pause Analysis")
    play = st.button("Continue")
    
    is_paused = False
    stframe = st.empty()  # Empty frame for Streamlit video output
    
    # Start video processing loop
    while True:
        if is_paused:
            time.sleep(0.1)  # Pause the video by sleeping
            continue  # Skip processing this frame
        
        success, img = cap.read()
        if not success:
            break  # End of video

        imgRegion = cv2.bitwise_and(img, img)
        results = model(imgRegion, stream=True)
        detections = np.empty((0, 5))

        # Process detections
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box and Confidence
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))

        resultsTracker = tracker.update(detections)

        # Draw lanes and process each detection
        for lane in limits.values():
            # Shift the y-coordinate of horizontal lines down by 10 pixels
            if lane[0][1] == lane[1][1]:  # Horizontal lines (lane 1, 2)
                lane[0] = (lane[0][0], lane[0][1] + 10)
                lane[1] = (lane[1][0], lane[1][1] + 10)
            
            # Draw the line (shifted or original)
            cv2.line(img, lane[0], lane[1], (250, 182, 122), 2)

        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2

            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(111, 237, 235))
            cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(25, y1)), scale=1, thickness=1, colorR=(56, 245, 213), colorT=(25, 26, 25), offset=10)
            cv2.circle(img, (cx, cy), 5, (22, 192, 240), cv2.FILLED)

            # Lane crossing logic for counting
            for lane_num, lane in limits.items():
                if lane_num == 1 or lane_num == 4:  # For Lanes 1 and 4
                    if lane[0][0] < cx < lane[1][0] and lane[0][1] - 15 < cy < lane[0][1] + 15:
                        if id not in totalCounts[lane_num]:
                            totalCounts[lane_num].append(id)
                            cv2.line(img, lane[0], lane[1], (12, 202, 245), 3)
                elif lane_num == 2 or lane_num == 3:  # For Lanes 2 and 3
                    if lane[0][1] < cy < lane[1][1] and lane[0][0] - 15 < cx < lane[0][0] + 15:
                        if id not in totalCounts[lane_num]:
                            totalCounts[lane_num].append(id)
                            cv2.line(img, lane[0], lane[1], (0, 255, 0), 5)


        # Display lane counts
        for i in range(1, 5):
            cvzone.putTextRect(img, f'{i} Lane: {len(totalCounts[i])}', (25, 75 + 70 * i), 2, thickness=2, colorR=(147, 245, 186), colorT=(15, 15, 15))

        # Show the processed video frame
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        stframe.image(img_rgb)
