#!/usr/bin/python

import argparse
import cv2 as cv
from utils import yolov4 as yolo
from utils import GrabCut

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.25, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.45, help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

img, boxes, idxs = yolo.runYOLOBoundingBoxes(args)
print("boxes' length: ", len(boxes))
print("idxs' shape: ", idxs.shape)
images = GrabCut.runGrabCut(img, boxes, idxs)

# show the output image
#cv.namedWindow("Image", cv.WINDOW_NORMAL)
#cv.resizeWindow("image", 1920, 1080)
for i in range(len(images)):
    #cv.imshow("Image", image)
    cv.imwrite("grabcut{}.jpg".format(i), images[i])
cv.waitKey(0)