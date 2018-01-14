# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 19:51:28 2017

@author: Sagar Lakhmani
"""
import cv2
from matplotlib import pyplot as plt
while True:
    cap1 = cv2.VideoCapture(0)
#    cap2 = cv2.VideoCapture(1)
    ret, frame = cap1.read()
#    ret2, frame2 = cap2.read()
    frame1=cv2.resize(frame[:][:][1],(640,480))
    frame2=cv2.resize(frame[:][:][2],(640,480))
    frame3=cv2.resize(frame[:][:][3],(640,480))
#    cv2.imshow("Image1", frame1)
#    cv2.imshow("Image2", frame2)
    cv2.imshow("Image3", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    stereo = cv2.StereoBM_create(numDisparities=32, blockSize=5)
    disparity = stereo.compute(frame1,frame2)
    plt.imshow(disparity,'gray')
    plt.show()
    cap1.release()
#    cap2.release()
cv2.destroyAllWindows()
    
