import cv2
import numpy as np
import torch

# 웹캠 신호 받기

# YOLO 가중치 파일과 CFG 파일 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

# YOLO NETWORK 재구성
classes = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

cap = cv2.VideoCapture(0)
while True:
    # Read the next frame from the video stream
    ret, frame = cap.read()

    # Convert the frame from BGR to RGB
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use the YOLOv5 model to detect objects in the frame
    results = model(frame)

    # Process the results to draw bounding boxes on the frame
    for detection in results.xyxy[0]:
        label = classes[int(detection[5])]
        if label != 'person':
            continue
        confidence = detection[4]
        x1, y1, x2, y2 = map(int, detection[:4])
        xn, yn = (x1+x2)/2, (y1+y2)/2
        dx, dy = abs(x1-x2), abs(y1-y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)        
        cv2.putText(frame, f"[{xn:.2f},{yn:.2f}]: {dx:.2f},{dy:.2f}", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
    # Display the frame with the detected objects
    cv2.imshow('YOLOv5', frame)
    

    # Check if the user pressed the 'q' key to quit the program
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture stream and close all windows
cap.release()
cv2.destroyAllWindows()