import cv2
import importlib_resources
import mediapipe as mp
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
segmentor = SelfiSegmentation()

while True:
    success, img = cap.read()
    img_out = segmentor.removeBG(img, (0,0,0), threshold=0.4)
   
    #cv2.imshow('Image', img)
    cv2.imshow('Image out', img_out)
    cv2.waitKey(1)
    #cv2.destroyAllWindows(0)
