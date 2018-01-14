# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 12:31:41 2017

@author: Sagar Lakhmani
"""

from keras.models import load_model
import numpy as np
import cv2
import os
import time
from pyardrone import ARDrone
drone = ARDrone()
path = os.getcwd()
model = load_model(path+'\Models\LRCN_GRU.h5')
img_rows,img_cols,img_depth=32,32,150
i = 1; 
#drone.video_ready.wait()
while i:
    X_test = []
    frames = []
    cap = cv2.VideoCapture(0)
    fps = cap.get(5)
    
    print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    for k in range(150):
        ret, frame = cap.read()
        cv2.imshow("Image", frame)
#        frame = drone.frame
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#        cv2.imshow("Image", gray)
        frames.append(gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    inp=np.array(frames)
#    print (inp.shape)
    ipt=np.rollaxis(np.rollaxis(inp,2,0),2,0)
#    print (ipt.shape)
    
    X_test.append(ipt)
#    X_test = X_test.astype('float32')

    X_test -= np.mean(X_test)

    X_test /=np.max(X_test)

    
    X_new = np.resize(X_test,(1,1,32,32,150))
    
    y_test = model.predict(X_new)
    
    if np.max(y_test) == y_test[0,0]:
        print('Action Predicted: Boxing')
        startTime = time.time()
        endTime = time.time()
        print('Drone moving forward')
        while(endTime - startTime < 2):
            drone.move(forward=0.1)
            endTime = time.time()
        drone.move(forward=0)
    if np.max(y_test) == y_test[0,1]:
        print('Action Predicted: Clapping')
#        send takeoff command
        print('Drone taking off')
        drone.takeoff()
    if np.max(y_test) == y_test[0,2]:
        print('Action Predicted: Waiving')
        print('Drone landing')
        #send land command
        drone.land()
    if np.max(y_test) == y_test[0,3]:
        print('Action Predicted: Jogging')
    if np.max(y_test) == y_test[0,4]:
        print('Action Predicted: Running')
    if np.max(y_test) == y_test[0,5]:
        print('Action Predicted: Walking')
        
    a = input('Do u want to continue(y/n)?:') 
    if a=='y':
        i = 1
    else:
        i = 0
        
#drone.land()