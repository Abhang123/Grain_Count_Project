# import cv2
# import numpy as np
# import streamlit as st
# from streamlit_lottie import st_lottie
# import json

# def app():

#     st.title("Computer :violet[Vision App]")
#     st.write("\n")
#     st.write("\n")

#     with open("anima1.json") as source:
#         animation = json.load(source)
#     st_lottie(animation, width = 300)

#     # Create a container with custom size
#     with st.container():
#         st.markdown("<style>div.stButton > button {width: 100%; height: 100% !important;}</style>", unsafe_allow_html=True)
#         photo = st.camera_input("Take a photo containing grains to count them.")
#         st.write("\n")
#         st.write("\n")
#         photo1 = st.file_uploader("Upload an image containing grains to count them.")

#         if photo1 is not None:
#             st.image(photo)

#     if photo1 is not None:
#         file_bytes = np.asarray(bytearray(photo1.read()), dtype=np.uint8)
#         image = cv2.imdecode(file_bytes, 1)
#         image_resized = cv2.resize(image, (600, 600))

#         # Convert the image to HSV
#         hsv_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)

#         # Define HSV ranges for red color

#         brown_lower = np.array([10, 100, 20])
#         brown_upper = np.array([20, 255, 200])  

#         dark_brown_lower = np.array([10, 100, 50])
#         dark_brown_upper =  np.array([30, 255, 255])

#         # lower_black = np.array([0, 0, 0])
#         # upper_black = np.array([180, 255, 50])

#         # Create masks for red regions

#         brown_mask = cv2.inRange(hsv_image,brown_lower,brown_upper)
#         dark_brown_mask = cv2.inRange(hsv_image,dark_brown_lower,dark_brown_upper)
#         # black_mask = cv2.inRange(hsv_image,lower_black,upper_black)

#         # Combine masks

#         mask = cv2.bitwise_or(brown_mask, dark_brown_mask)
#         # mask = cv2.bitwise_or(mask,black_mask)
        
#         # Apply the mask to the original image
#         filtered_image = cv2.bitwise_and(image_resized, image_resized, mask=mask)

#         gray_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
#         blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

#         # Apply a morphological operation (Top-Hat) to remove shadows
#         kernel = np.ones((15, 15), np.uint8)  # You can adjust the kernel size
#         top_hat = cv2.morphologyEx(blurred_image, cv2.MORPH_TOPHAT, kernel)

#         # Threshold Parameters
#         block_size = 3 # st.slider("Adaptive Threshold Block Size", 3, 21, 3, 2)
#         c_value = 2 # st.slider("Adaptive Threshold Constant", -10, 10, 2, 1)
#         thresh_image = cv2.adaptiveThreshold(
#             top_hat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c_value
#         )

#         # Morphological Parameters
#         kernel_size = 5 # st.slider("Kernel Size for Morphology", 1, 15, 5, 2)
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
#         morphed_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel)

#         # Contour Filtering
#         contours, _ = cv2.findContours(morphed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         min_area = 50 # st.slider("Minimum Area Threshold", 10, 300, 30, 10)
#         filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

#         # Count and Draw Contours
#         total_grains = len(filtered_contours)
#         image_copy = image_resized.copy()
#         for cnt in filtered_contours:
#             cv2.drawContours(image_copy, [cnt], -1, (0, 255, 0), 2)

#         # Display Results
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         cv2.putText(image_copy, f"Total Count: {total_grains}", (20, 50), font, 1, (255, 0, 0), 2)
#         _, img_encoded = cv2.imencode('.png', image_copy)
#         st.image(img_encoded.tobytes(), caption="Processed Image with Total Count")

#         st.subheader("Thresholded Image")
#         _, thresh_encoded = cv2.imencode('.png', thresh_image)
#         st.image(thresh_encoded.tobytes())

#         st.subheader("Morphed Image")
#         _, morphed_encoded = cv2.imencode('.png', morphed_image)
#         st.image(morphed_encoded.tobytes())

#         st.write(f"## Total number of grains detected: {total_grains}")

# app()












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

        # **1. Noise Reduction**
        # Apply Gaussian Blur to reduce noise
        blurred_image = cv2.GaussianBlur(image_resized, (5, 5), 0) 

        # **2. Color Space Conversion**
        # Convert to HSV color space
        hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

        # Define HSV ranges for grain colors (adjust as needed)
        brown_lower = np.array([10, 100, 20])
        brown_upper = np.array([20, 255, 200]) 
        dark_brown_lower = np.array([10, 100, 50])
        dark_brown_upper = np.array([30, 255, 255])
        # black_lower = np.array([0, 0, 0])
        # black_upper = np.array([180, 255, 50])

        # Create masks for grain colors
        brown_mask = cv2.inRange(hsv_image, brown_lower, brown_upper)
        dark_brown_mask = cv2.inRange(hsv_image, dark_brown_lower, dark_brown_upper)
        # black_mask = cv2.inRange(hsv_image, black_lower, upper_black)
        mask = cv2.bitwise_or(brown_mask, dark_brown_mask)
        # mask = cv2.bitwise_or(mask, black_mask)

        # Apply the mask to the original image
        filtered_image = cv2.bitwise_and(image_resized, image_resized, mask=mask)

        # Convert to grayscale
        gray_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)

        # **3. Adaptive Thresholding**
        block_size = 11  # Adjust as needed
        c_value = 2 
        thresh_image = cv2.adaptiveThreshold(
            gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c_value
        )

        # Morphological Operations
        kernel = np.ones((5, 5), np.uint8) 
        morphed_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel) 

        # **4. Contour Filtering**
        contours, _ = cv2.findContours(morphed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 100  # Adjust as needed
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
        st.image(img_encoded.tobytes(), caption="Processed Image with Total Count")

        # Display intermediate results (optional)
        st.subheader("Thresholded Image")
        _, thresh_encoded = cv2.imencode('.png', thresh_image)
        st.image(thresh_encoded.tobytes())

        st.subheader("Morphed Image")
        _, morphed_encoded = cv2.imencode('.png', morphed_image)
        st.image(morphed_encoded.tobytes())

        st.write(f"## Total number of grains detected: {total_grains}")

app()