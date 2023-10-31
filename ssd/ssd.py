import cv2
import numpy as np
import matplotlib.pyplot as plt

import YB_Pcb_Car
import time
car = YB_Pcb_Car.YB_Pcb_Car()

# YOLO 가중치 파일과 CFG 파일 로드
prototxt_path = 'MobileNetSSD_deploy.prototxt.txt'
model_path = 'MobileNetSSD_deploy.caffemodel'
 
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", 
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

cap = cv2.VideoCapture(0)
LABEL_COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
ret, frame = cap.read()

(h, w) = frame.shape[:2]
# 세로 480   아래 480 위 0
# 가로 640   왼 0 오른 640

def move(sx,sy,ex,ey):
    #ww = abs(sx-ex)
    hh = abs(sy-ey)
    mx = int((sx+ex)/2)
    #my = int((sy+ey)/2)
################################### speed 조절해주세용
    # hh 값에 따라 속도 조절  
    if hh > 360:
        speed = 150
    elif hh > 240:
        speed = 100
    elif hh > 120:
        speed = 50
    else:
        speed = 00
    # x 좌표에 따라 차량 기동
    # x ~ [0,640]
################################## 채워주세용
    if mx < 160:
        #move left 1
        #  car.Car_Left(0, speed)
        return
    elif mx < 320:
        #move left 2
        return
    elif mx < 480:
        #move right 1
        return
    else:
        #move right 2
        return
################################### 숫자 채워놓고 말씀해주세용 저도 구경할래용
    
    

while True:
    # Read the next frame from the video stream
    ret, frame = cap.read()

    #print(np.shape(frame))
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (240, 320), 127.5)
    net.setInput(blob)
    detections = net.forward()
    conf = 0.5

    # Convert the frame from BGR to RGB
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    vis = frame.copy()
    # Process the results to draw bounding boxes on the frame
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        is_person = detections[0, 0, i, 1]
        # if is_person != 15:
        #     continue
        if confidence > conf:
            idx = int(detections[0, 0, i, 1])
            if idx == 15:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                mx, my = int((startX+endX)/2), int((startY+endY)/2)
                print("[INFO] {} : [ {:.2f} % ]".format(CLASSES[idx], confidence * 100))
                
                cv2.rectangle(vis, (startX, startY), (endX, endY), LABEL_COLORS[idx], 1)
                #y = startY - 10 if startY - 10 > 10 else startY + 10
                #cv2.putText(vis, "{} : {:.2f}%".format(CLASSES[idx], confidence * 100), (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, LABEL_COLORS[idx], 2)
                cv2.putText(vis, "{} : {}%".format(mx, my), (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.7, LABEL_COLORS[idx], 2)

    cv2.imshow('Object Detection', vis)
    #### move
    move(startX, startY, endX, endY)

    # Check if the user pressed the 'q' key to quit the program
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture stream and close all windows
cap.release()
cv2.destroyAllWindows()



