import cv2
import numpy as np
import streamlit as st
from streamlit_lottie import st_lottie
import json

def app():

    st.title("Total Grain :violet[Count Detection]")
    st.write("\n")
    st.write("\n")

    with open("anima1.json") as source:
        animation = json.load(source)
    st_lottie(animation, width = 400)
    
    photo = st.camera_input("Upload an image containing grains to count them.")

    if photo is not None:
        file_bytes = np.asarray(bytearray(photo.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image_resized = cv2.resize(image, (600, 600))
        gray_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Threshold Parameters
        block_size = st.slider("Adaptive Threshold Block Size", 3, 21, 11, 2)
        c_value = st.slider("Adaptive Threshold Constant", -10, 10, 2, 1)
        thresh_image = cv2.adaptiveThreshold(
            blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c_value
        )

        # Morphological Parameters
        kernel_size = st.slider("Kernel Size for Morphology", 1, 15, 5, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        morphed_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel)

        # Contour Filtering
        contours, _ = cv2.findContours(morphed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        min_area = st.slider("Minimum Area Threshold", 10, 300, 50, 10)
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        # Count and Draw Contours
        total_grains = len(filtered_contours)
        image_copy = image_resized.copy()
        for cnt in filtered_contours:
            cv2.drawContours(image_copy, [cnt], -1, (0, 255, 0), 2)

        # Display Results
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image_copy, f"Total Count: {total_grains}", (20, 50), font, 1, (255, 0, 0), 2)
        _, img_encoded = cv2.imencode('.png', image_copy)
        st.image(img_encoded.tobytes(), caption="Processed Image with Total Count", use_column_width=True)

        st.subheader("Thresholded Image")
        _, thresh_encoded = cv2.imencode('.png', thresh_image)
        st.image(thresh_encoded.tobytes(), use_column_width=True)

        st.subheader("Morphed Image")
        _, morphed_encoded = cv2.imencode('.png', morphed_image)
        st.image(morphed_encoded.tobytes(), use_column_width=True)

        st.write(f"## Total number of grains detected: {total_grains}")

app()
