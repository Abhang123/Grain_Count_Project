import cv2
import numpy as np
import streamlit as st
from streamlit_lottie import st_lottie
import json

def app():

    st.title("Computer :violet[Vision App]")
    st.write("\n")
    st.write("\n")

    with open("anima1.json") as source:
        animation = json.load(source)
    st_lottie(animation, width = 300)

    # Create a container with custom size
    with st.container():
        st.markdown("<style>div.stButton > button {width: 100%; height: 100% !important;}</style>", unsafe_allow_html=True)
        photo = st.camera_input("Upload an image containing grains to count them.")

        if photo is not None:
            st.image(photo)

    if photo is not None:
        file_bytes = np.asarray(bytearray(photo.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image_resized = cv2.resize(image, (600, 600))

        # Convert the image to HSV
        hsv_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)

        # Define HSV ranges for red color
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        brown_lower = np.array([10, 100, 20])
        brown_upper = np.array([20, 255, 200])

        black_lower = np.array([0, 0, 0])  
        black_upper = np.array([180, 255, 50])  


        # Create masks for red regions
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        mask3 = cv2.inRange(hsv_image,brown_lower,brown_upper)
        mask4 = cv2.inRange(hsv_image,black_lower,black_upper)

        # Combine masks
        red_mask = cv2.bitwise_or(mask1, mask2)
        red_mask = cv2.bitwise_or(red_mask,mask3)
        red_mask = cv2.bitwise_or(red_mask,mask4)

        # Apply the mask to the original image
        filtered_image = cv2.bitwise_and(image_resized, image_resized, mask=red_mask)

        gray_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Apply a morphological operation (Top-Hat) to remove shadows
        kernel = np.ones((15, 15), np.uint8)  # You can adjust the kernel size
        top_hat = cv2.morphologyEx(blurred_image, cv2.MORPH_TOPHAT, kernel)

        # Threshold Parameters
        block_size = 3 # st.slider("Adaptive Threshold Block Size", 3, 21, 3, 2)
        c_value = 2 # st.slider("Adaptive Threshold Constant", -10, 10, 2, 1)
        thresh_image = cv2.adaptiveThreshold(
            top_hat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c_value
        )

        # Morphological Parameters
        kernel_size = 5 # st.slider("Kernel Size for Morphology", 1, 15, 5, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        morphed_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel)

        # Contour Filtering
        contours, _ = cv2.findContours(morphed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 50 # st.slider("Minimum Area Threshold", 10, 300, 30, 10)
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