import cv2
import time
import numpy as np
import os
import mediapipe as mp

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
offset = 20
cap = cv2.VideoCapture(0)

while True:
    sucess,img = cap.read()
    h, w, ic = img.shape
    
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            lm_x = []
            lm_y = []
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for id,lm in enumerate(hand_landmark.landmark): 
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            
            mpDraw.draw_landmarks(img,hand_landmark,mpHands.HAND_CONNECTIONS)
                                    
            cv2.putText(img,'Hand',(x_min-25,y_min-25),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),1)
            cv2.rectangle(img, (x_min-20, y_min-20), (x_max+20, y_max+20), (121, 9, 129), 2)
            
            # top left
            cv2.line(img,(x_min-20,y_min+30),(x_min-20,y_min-20),(255,0,255),5)
            cv2.line(img,(x_min-20,y_min-20),(x_min+30,y_min-20),(255,0,255),5)

            # top right
            cv2.line(img,(x_max-30,y_min-20),(x_max+20,y_min-20),(255,0,255),5)
            cv2.line(img,(x_max+20,y_min-20),(x_max+20,y_min+30),(255,0,255),5)
            
            #bottom left
            cv2.line(img,(x_min-20,y_max+20),(x_min-20,y_max-30),(255,0,255),5)
            cv2.line(img,(x_min-20,y_max+20),(x_min+30,y_max+20),(255,0,255),5)
            
            #bottom right
            cv2.line(img,(x_max+20,y_max+20),(x_max+20,y_max-30),(255,0,255),5)
            cv2.line(img,(x_max+20,y_max+20),(x_max-30,y_max+20),(255,0,255),5)
            
            # print()
            # print(x_max-x_min)
            if ((y_min>20) & (x_min>20)):
                imgCrop = img[y_min-20:y_max+20,x_min-20:x_max+20]
                cv2.imshow('Image2',imgCrop)
                
            print(y_min,x_min)
            # imgCropShape = imgCrop.shape
            # img[0:imgCropShape[0],0:imgCropShape[1]] = imgCrop
            # print(imgCrop.shape)
            # print(f'xmin: {x_min}, xmax: {x_max}, ymin: {y_min}, ymax: {y_max}, width: {x_max+x_min}, heigth: {y_max+y_min}')
            
        
    cv2.imshow("Image",img)
    cv2.waitKey(1)