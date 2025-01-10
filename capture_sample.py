import cv2
import numpy as np
import streamlit as st
from streamlit_lottie import st_lottie
import json

def app():

    st.title("Computer :violet[Vision App]")
    st.write("\n")
    st.write("\n")

    with open("animation5.json") as source:
        animation = json.load(source)
    st_lottie(animation, width = 300)

    st.write("\n")
    st.write("### Take 3 photos of daal.")
    st.write("\n")

    # Create a container with custom size
    with st.container():
        st.markdown("<style>div.stButton > button {width: 100%; height: 100% !important;}</style>", unsafe_allow_html=True)
        photo1 = st.camera_input("Take first photo of daal")
        st.write("\n")
        st.write("\n")
        if photo1:
            st.write("Image 1 captured successfully!")
            st.write("\n")
            photo2 = st.camera_input("Take second photo of daal.")
            st.write("\n")
            st.write("\n")
            if photo2:
                st.write("Image 2 captured successfully!")
                st.write("\n")
                photo3 = st.camera_input("Take third photo of daal.")
                if photo3:
                    st.write("Image 3 captured successfully!")
                st.write("\n")
                st.write("\n")
        else:
            st.write("Please take first photo of daal.")

    if (photo1 and photo2 and photo3) is not None:

        # Define HSV ranges for red color
        # Note: Red is at both ends of the HSV spectrum, so we define two ranges
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        brown_lower = np.array([15,200,150])
        brown_upper = np.array([20,255,150])

        black_lower = np.array([0, 0, 0])  
        black_upper = np.array([180, 255, 50])  

        file_bytes1 = np.asarray(bytearray(photo1.read()), dtype=np.uint8)
        image1 = cv2.imdecode(file_bytes1, 1)
        image_resized1 = cv2.resize(image1, (600, 600))

        # Convert the image to HSV
        hsv_image1 = cv2.cvtColor(image_resized1, cv2.COLOR_BGR2HSV)

        file_bytes2 = np.asarray(bytearray(photo2.read()), dtype=np.uint8)
        image2 = cv2.imdecode(file_bytes2, 1)
        image_resized2 = cv2.resize(image2, (600, 600))

        # Convert the image to HSV
        hsv_image2 = cv2.cvtColor(image_resized2, cv2.COLOR_BGR2HSV)   

        file_bytes3 = np.asarray(bytearray(photo3.read()), dtype=np.uint8)
        image3 = cv2.imdecode(file_bytes3, 1)
        image_resized3 = cv2.resize(image3, (600, 600))

        hsv_image3 = cv2.cvtColor(image_resized2, cv2.COLOR_BGR2HSV)   

        # Convert the image to HSV

        # Create masks for red regions
        mask1 = cv2.inRange(hsv_image1, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_image1, lower_red2, upper_red2)
        mask3 = cv2.inRange(hsv_image1,brown_lower,brown_upper)
        mask4 = cv2.inRange(hsv_image1,black_lower,black_upper)

        red_mask1 = cv2.bitwise_or(mask1, mask2)
        red_mask1 = cv2.bitwise_or(red_mask1,mask3)
        red_mask1 = cv2.bitwise_or(red_mask1,mask4)

        mask12 = cv2.inRange(hsv_image2, lower_red1, upper_red1)
        mask22 = cv2.inRange(hsv_image2, lower_red2, upper_red2)
        mask32 = cv2.inRange(hsv_image2,brown_lower,brown_upper)
        mask42 = cv2.inRange(hsv_image2,black_lower,black_upper)

        red_mask2 = cv2.bitwise_or(mask12, mask22)
        red_mask2 = cv2.bitwise_or(red_mask2,mask32)
        red_mask2 = cv2.bitwise_or(red_mask2,mask42)

        mask13 = cv2.inRange(hsv_image3, lower_red1, upper_red1)
        mask23 = cv2.inRange(hsv_image3, lower_red2, upper_red2)
        mask33 = cv2.inRange(hsv_image3,brown_lower,brown_upper)
        mask43 = cv2.inRange(hsv_image3,black_lower,black_upper)

        red_mask3 = cv2.bitwise_or(mask13, mask23)
        red_mask3 = cv2.bitwise_or(red_mask3,mask33)
        red_mask3 = cv2.bitwise_or(red_mask3,mask43)

        # Apply the mask to the original image
        filtered_image1 = cv2.bitwise_and(image_resized1, image_resized1, mask=red_mask1)
        filtered_image2 = cv2.bitwise_and(image_resized2, image_resized2, mask=red_mask2)
        filtered_image3 = cv2.bitwise_and(image_resized3, image_resized3, mask=red_mask3)
        
        gray_image1 = cv2.cvtColor(filtered_image1, cv2.COLOR_BGR2GRAY)
        blurred_image1 = cv2.GaussianBlur(gray_image1, (5, 5), 0)

        gray_image2 = cv2.cvtColor(filtered_image2, cv2.COLOR_BGR2GRAY)
        blurred_image2 = cv2.GaussianBlur(gray_image2, (5, 5), 0)

        gray_image3 = cv2.cvtColor(filtered_image3, cv2.COLOR_BGR2GRAY)
        blurred_image3 = cv2.GaussianBlur(gray_image3, (5, 5), 0)                

        # Threshold Parameters
        block_size = 3 # st.slider("Adaptive Threshold Block Size", 3, 21, 3, 2)
        c_value = 2 # st.slider("Adaptive Threshold Constant", -10, 10, 2, 1)
        thresh_image1 = cv2.adaptiveThreshold(
            blurred_image1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c_value
        )

        thresh_image2 = cv2.adaptiveThreshold(
            blurred_image2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c_value
        )

        thresh_image3 = cv2.adaptiveThreshold(
            blurred_image3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c_value
        )                

        # Morphological Parameters
        kernel_size = 5 # st.slider("Kernel Size for Morphology", 1, 15, 5, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        morphed_image1 = cv2.morphologyEx(thresh_image1, cv2.MORPH_CLOSE, kernel)
        morphed_image2 = cv2.morphologyEx(thresh_image2, cv2.MORPH_CLOSE, kernel)
        morphed_image3 = cv2.morphologyEx(thresh_image3, cv2.MORPH_CLOSE, kernel)


        # Contour Filtering

        min_area = 30 # st.slider("Minimum Area Threshold", 10, 300, 30, 10)
        contours1, _ = cv2.findContours(morphed_image1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours1 = [cnt for cnt in contours1 if cv2.contourArea(cnt) > min_area]

        contours2, _ = cv2.findContours(morphed_image2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours2 = [cnt for cnt in contours2 if cv2.contourArea(cnt) > min_area]

        contours3, _ = cv2.findContours(morphed_image3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours3 = [cnt for cnt in contours3 if cv2.contourArea(cnt) > min_area]        

        # Count and Draw Contours
        total_grains = (len(filtered_contours1) + len(filtered_contours2) + len(filtered_contours3)) / 3
        image_copy = image_resized1.copy()
        for cnt in filtered_contours1:
            cv2.drawContours(image_copy, [cnt], -1, (0, 255, 0), 2)

        # Display Results
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(image_copy, f"Total Count: {total_grains}", (20, 50), font, 1, (255, 0, 0), 2)
        _, img_encoded = cv2.imencode('.png', image_copy)
        st.image(img_encoded.tobytes(), caption="Processed Image with Total Count", use_column_width=True)

        st.subheader("Thresholded Image")
        _, thresh_encoded = cv2.imencode('.png', thresh_image2)
        st.image(thresh_encoded.tobytes(), use_column_width=True)

        st.subheader("Morphed Image")
        _, morphed_encoded = cv2.imencode('.png', morphed_image3)
        st.image(morphed_encoded.tobytes(), use_column_width=True)

        st.write(f"## Total Count: {int(total_grains)}")

app()