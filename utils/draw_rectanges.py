import cv2 as cv


def draw_rectangles(image,boxes,idxs):

    if len(idxs)!=0:
        image=image

        for i in idxs.flatten():
            image= cv.rectangle(image,(boxes[i][0],boxes[i][1]),(boxes[i][2],boxes[i][3]),(255,0,0),1)

    return image
