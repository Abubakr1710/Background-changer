import cv2
from cv2 import threshold
import importlib_resources
import mediapipe as mp
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
segmentor = SelfiSegmentation()
fps_reader = cvzone.FPS()
imgbn = cv2.imread('img1.png')
imgbn = cv2.resize(imgbn, (640, 480))

while True:
    success, img = cap.read()
    img_out = segmentor.removeBG(img, imgbn, threshold=0.65)
    
    img_stacked = cvzone.stackImages([img, img_out], 2,1)
    _,  img_stacked = fps_reader.update(img_stacked,color=(255,0,0))
    #cv2.imshow('Image', img)
    cv2.imshow('Images', img_stacked)
    cv2.waitKey(1)
    #cv2.destroyAllWindows(0)
