import cv2 as cv
import streamlit as st

def draw_rectangles(image,boxes,idxs):

    if len(idxs)>0:
        st.write(f'Indexes of boxes {idxs}')
        for i in idxs.flatten():
            st.write(boxes[i])
            
            x=boxes[i][0]
            y=boxes[i][1]
            width=boxes[i][2]
            height=boxes[i][3]
            if x<0: x=0
            if y<0: y=0
            if x+width>image.shape[0]: 
                x_max=image.shape[0]
            else:
                x_max=x+width
                
            if y+height>image.shape[1]: 
                y_max=image.shape[1]
            else:
                y_max=y+height

            image= cv.rectangle(image,(x,y),(x_max,y_max),(255,0,0),2)
            

    return image
