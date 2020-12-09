import numpy as np
import cv2 as cv
from utils import yolov4 as yolo

def runGrabCut(_image, boxes, indices):
    imgs = []

    # ensure at least one detection exists
    if len(indices) > 0:
        # loop over the indices we are keeping
        for i in indices.flatten():
            image = _image.copy()
            mask = np.zeros(_image.shape[:2], np.uint8)
            bgdModel = np.zeros((1, 65), np.float64)
            fgbModel = np.zeros((1, 65), np.float64)
            # extract the bounding box coordinates
            rect = (boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])
            print(rect)

            # apply GrabCut
            cv.grabCut(image, mask, rect, bgdModel, fgbModel, 5, cv.GC_INIT_WITH_RECT)

            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            image = image * mask2[:, :, np.newaxis]

            imgs.append(image)

    return imgs

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    ap.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
    ap.add_argument("-c", "--confidence", type=float, default=0.35, help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.45, help="threshold when applying non-maxima suppression")
    args = vars(ap.parse_args())

    img, boxes, idxs = yolo.runYOLOBoundingBoxes(args)

    images = runGrabCut(img, boxes, idxs)

    # show the output images
    #cv.namedWindow("Image", cv.WINDOW_NORMAL)
    #cv.resizeWindow("image", 1920, 1080)
    for i in range(len(images)):
        cv.imshow("Image{}".format(i), images[i])
        cv.imwrite("grabcut{}.jpg".format(i), images[i])
    cv.waitKey(0)