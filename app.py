# import basic libraries
import numpy as np
import streamlit as st
from deepface import DeepFace as dfc
from PIL import Image
import os
import tensorflow as tf
import tensorflow_hub as hub
import cv2
from matplotlib import pyplot as plt
import tempfile
import time

# Optional if you are using a GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 4)

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 6, (0,255,0), -1)

# Function to loop through each person detected and render
def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def main():

    st.sidebar.title("Real-time Multi-Person Pose Detection")
    activiteis = ["Video",  "Webcam", "About"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    st.sidebar.text(
        """ Subject: BDAS436177_FA2022    
Instructor: Nguyen Xuan Sam     
Member: 
Nguyen Thi My Linh  
Vo Thi Ngoc Tham    
Nguyen Phuoc Tuan""")
    # cap = cv2.VideoCapture(0)
    stframe = st.empty()
    if choice == "Video":
        f = st.file_uploader('Upload File')
        if f is not None:
            tmpfile = tempfile.NamedTemporaryFile(delete=False)
            tmpfile.write(f.read())

            cap = cv2.VideoCapture(tmpfile.name)
            while cap.isOpened():
                ret, frame = cap.read()
                imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize image
                img = frame.copy()
                img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 384,640)
                input_img = tf.cast(img, dtype=tf.int32)
                
                # Detection section
                results = movenet(input_img)
                keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
                
                # Render keypoints 
                loop_through_people(frame, keypoints_with_scores, EDGES, 0.1)
                
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                stframe.image(frame)

                # btn = st.download_button(
                # label="Download image",
                # data = frame,
                # file_name="output.mp4",
                # mime="image/mp4"
                # )

                if cv2.waitKey(1) & 0xFF==ord('q'):
                    break
                

    elif choice == "Webcam":
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize image
            img = frame.copy()
            img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 384,640)
            input_img = tf.cast(img, dtype=tf.int32)
            
            # Detection section
            results = movenet(input_img)
            keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
            
            # Render keypoints 
            loop_through_people(frame, keypoints_with_scores, EDGES, 0.1)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            stframe.image(frame)
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
    elif choice == "About":
        st.subheader("About this app")
        html_temp_about1= """<div style="background-color:#6D7B8D;padding:10px">
                                    <p >
                                    MoveNet là một nhanh cực và mô hình chính xác mà phát hiện 17 keypoint của một cơ thể. Mô hình này được cung cấp trên TF Hub với hai biến thể, được gọi là Lightning và Thunder</p>
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)

    else:
        pass

if __name__ == '__main__':
    main()


