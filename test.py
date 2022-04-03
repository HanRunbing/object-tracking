# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 15:25:28 2021

@author: hanr0003
"""
import cv2
video_capture = cv2.VideoCapture("./test_data/image_02/data/%04d.png")
while(video_capture.isOpened() == True):
    ret, frame = video_capture.read()

    cv2.imshow('frame', frame)
    c = cv2.waitKey(50) & 0xFF
    if c==27: # ESC
        break

video_capture.release()
cv2.destroyAllWindows()