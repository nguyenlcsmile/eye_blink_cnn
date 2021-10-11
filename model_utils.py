import cv2 
import numpy as np 
import tensorflow as tf 
import dlib 
import os 
import matplotlib.pyplot as plt 

print('[INFO] Loanding face detector model...')
path_model = 'face_detector/deploy.prototxt'
path_weight = 'face_detector/res10_300x300_ssd_iter_140000.caffemodel'

prototxtPath = os.path.join(path_model)
weightsPath = os.path.join(path_weight)
net = cv2.dnn.readNet(prototxtPath, weightsPath)

def model_detection_faces(image):
    ''' 
        Sử dụng pretrain model của res10_300x300 đã được train sẵn dùng để phát hiện các face trong image, video
        Input: images, videos,...
        Output: tọa độ bounding box chứa các face
    '''

    #Convert image to blob (taij vì model face detector này được training bằng pytorch cho nên để implement 
    # chúng ta nên chuyển image sang đúng form của pytorch (batch_size, chanle, weight, height))
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

    #Face detetions
    # print("[INFO] Computing face detections...")
    net.setInput(blob)
    detections = net.forward()

    faces = []
    #Height, width of image
    (h, w) = image.shape[:2]
    #Lặp qua các detections
    for i in range(0, detections.shape[2]):
        #Lấy ra độ tin cậy (xác suất,...) tương ứng của mỗi detection
        confidence = detections[0, 0, i, 2]
        #Lọc ra các detections đảm bảo độ tin cậy > ngưỡng tin cậy
        if confidence > 0.3:
            #Tính toán (x,y) bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            #Đảm bảo bounding box nằm trong kích thước frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            #Convert bounding box to rectangle dlib
            faces.append(dlib.rectangle(left=startX, top=startY, right=endX, bottom=endY))

    return faces

def model_dlib(path_model_dlib):
    ''' 
        Sử dụng để lấy tọa độ các điểm(đối tượng trên face: mắt, mũi, miệng,...) bằng weight 
        đã được training 
    '''
    return dlib.shape_predictor(path_model_dlib)