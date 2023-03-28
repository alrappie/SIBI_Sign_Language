import cv2
import time
import numpy as np
import os
import mediapipe as mp
import tensorflow as tf
from keras.preprocessing import image

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
model = tf.keras.models.load_model('my_model3.h5')


FOLDER = './myData/D'
counter = 0
dict_list = {0:'A',1:'B',2:'C',3:'D',4:'E'}

while True:
    sucess,img = cap.read()
    h, w, ic = img.shape
    
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
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
                                    
            
            ## bounding box
            
            #Rectangle
            cv2.rectangle(img, (x_min-20, y_min-20), (x_max+20, y_max+20), (121, 9, 129), 2)
            
            #top left
            cv2.line(img,(x_min-20,y_min+30),(x_min-20,y_min-20),(255,0,255),5)
            cv2.line(img,(x_min-20,y_min-20),(x_min+30,y_min-20),(255,0,255),5)

            #top right
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
                
                img_crop = img[y_min-20:y_max+20,x_min-20:x_max+20]
                
                cv2.imshow('Image2',img_crop)
                    
                ## do preprocessing
                img_crop = cv2.resize(img_crop,(200,200),cv2.INTER_AREA)
                img_crop = image.img_to_array(img_crop)
                img_crop = np.expand_dims(img_crop, axis=0)
                img_crop /= 255.
                
                # Predict
                pred = np.argmax(model.predict(img_crop))
                
                cv2.putText(img,f'{dict_list[pred]}',(x_min-25,y_min-25),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),1)
                
                
                
    cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    if key == ord('s'):
        counter +=1
        print(img_crop.shape)
        cv2.imwrite(f'{FOLDER}/Image_{counter}.jpg',img_crop)
        print(counter)