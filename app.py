#!/usr/bin/python

import os
import stat
import time
import cv2 as cv
import numpy as np
import pandas as pd
import streamlit as st

from utils import yolov4 as yolo
from utils import GrabCut
from utils.draw_rectangles import draw_rectangles


def run():
    yolopath = "./fishv4"
    confidence = 0.30
    threshold = 0.40

    st.title("Fishv4[tiny] Demo 2020")
    st.text("Repo from: https://fishv4.herokuapp.com/")
    st.text("More info: https://github.com/DZPeru/fishv4")
    fileStatsObj=os.stat('app.py')
    modificationTime = time.ctime ( fileStatsObj [ stat.ST_MTIME ] )
    last_update="Last Modified Time : "+ modificationTime 
    st.text(last_update)

    uploaded_img = st.file_uploader("Elige una imagen compatible", type=[
                                    'png', 'jpg', 'bmp', 'jpeg'])
    if uploaded_img is not None:

        file_details = {"FileName": uploaded_img.name,
                        "FileType": uploaded_img.type,
                        "FileSize": uploaded_img.size}
        st.write(file_details)

        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        image = cv.imdecode(file_bytes, 1)

        dimensions=image.shape

        st.text(f"This is your uploaded image:\n-height:{dimensions[0]}\n-width:{dimensions[1]}\n-channels:{dimensions[2]}")
        st.image(image, caption='Uploaded Image', channels="BGR", use_column_width=True)

        boxes, idxs = yolo.runYOLOBoundingBoxes_streamlit(image, yolopath, confidence, threshold)
        st.write(pd.DataFrame.from_dict({'confidence' : [confidence],
                                        'threshold' : [threshold],
                                        'Encontrados (Boxes)': [len(boxes)],
                                        'Válidos (idxs)': [len(idxs)],}))
        st.write(boxes)
        result_images = GrabCut.runGrabCut(image, boxes, idxs)

        st.write("Here appears the rectangles that the algorithm recognize:")
        img_mod=draw_rectangles(image,boxes,idxs)
        st.image(img_mod, channels="BGR", use_column_width=True)

        st.write("")
        st.write("finish grabcut")
        st.write(f"There are {len(result_images)} segmented fish image. Each listed as below:")
        for i in range(len(result_images)):
            #cv.imwrite(f'grabcut{i}.jpg', result_images[i])
            st.image(result_images[i], channels="BGR", use_column_width=True)

if __name__ == '__main__':
    run()
