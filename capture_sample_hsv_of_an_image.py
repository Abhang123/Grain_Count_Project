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
#         st.write("### You can upload or capture live photo below - ")
#         photo = st.file_uploader("Upload an image or capture it using your camera.")

#     if photo is not None:
#         file_bytes = np.asarray(bytearray(photo.read()), dtype=np.uint8)
#         image = cv2.imdecode(file_bytes, 1)
#         image_resized = cv2.resize(image, (600, 600))

#         # **1. Noise Reduction**
#         # Apply Gaussian Blur to reduce noise
#         blurred_image = cv2.GaussianBlur(image_resized, (5, 5), 0) 

#         # **2. Color Space Conversion**
#         # Convert to HSV color space
#         hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

#         # Define HSV ranges for grain colors (adjust as needed)
#         brown_lower = np.array([10, 100, 20])
#         brown_upper = np.array([20, 255, 200]) 
#         dark_brown_lower = np.array([10, 100, 50])
#         dark_brown_upper = np.array([30, 255, 255])
#         # black_lower = np.array([0, 0, 0])
#         # black_upper = np.array([180, 255, 50])

#         yellow_lower = np.array([25,100,255])
#         yellow_upper = np.array([35,255,255])

#         # Create masks for grain colors
#         brown_mask = cv2.inRange(hsv_image, brown_lower, brown_upper)
#         dark_brown_mask = cv2.inRange(hsv_image, dark_brown_lower, dark_brown_upper)
#         # black_mask = cv2.inRange(hsv_image, black_lower, upper_black)
#         yellow_mask = cv2.inRange(hsv_image, yellow_lower,yellow_upper)
#         mask = cv2.bitwise_or(brown_mask, dark_brown_mask)
#         # mask = cv2.bitwise_or(mask, yellow_mask)

#         # Apply the mask to the original image
#         filtered_image = cv2.bitwise_and(image_resized, image_resized, mask=mask)

#         # Convert to grayscale
#         gray_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)

#         # **3. Adaptive Thresholding**
#         block_size = 3  # Adjust as needed
#         c_value = 2 
#         thresh_image = cv2.adaptiveThreshold(
#             gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c_value
#         )

#         # Morphological Operations
#         kernel = np.ones((5, 5), np.uint8) 
#         morphed_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel) 
        
#         # **4. Contour Filtering**
#         contours, _ = cv2.findContours(morphed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         min_area = 50  # Adjust as needed
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

#         # # Display intermediate results (optional)
#         # st.subheader("Thresholded Image")
#         # _, thresh_encoded = cv2.imencode('.png', thresh_image)
#         # st.image(thresh_encoded.tobytes())

#         # st.subheader("Morphed Image")
#         # _, morphed_encoded = cv2.imencode('.png', morphed_image)
#         # st.image(morphed_encoded.tobytes())

#         st.write(f"## Total number of grains detected: {total_grains}")

#         # Visualize masks (optional)
#         # Visualize masks (optional)
#         st.subheader("Brown Mask")
#         st.image(brown_mask)
#         st.subheader("Dark Brown Mask")
#         st.image(dark_brown_mask)
#         st.subheader("Yellow Mask")
#         st.image(yellow_mask)
        
# app()









# -------------------------------------------------------------------------------------------- #






import cv2
import numpy as np
import streamlit as st

def app():

    st.title("Total Grain Count Detection")
    photo = st.file_uploader("Upload an image of daal.")

    if photo is not None:

        file_bytes = np.asarray(bytearray(photo.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image_resized = cv2.resize(image, (600, 600))
        gray_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Threshold Parameters
        block_size = 3
        c_value = 2
        thresh_image = cv2.adaptiveThreshold(
            blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c_value
        )

        # Morphological Parameters
        kernel_size = 5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        morphed_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel)

        # Contour Filtering
        contours, _ = cv2.findContours(morphed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 80
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

        # st.subheader("Thresholded Image")
        # _, thresh_encoded = cv2.imencode('.png', thresh_image)
        # st.image(thresh_encoded.tobytes())

        # st.subheader("Morphed Image")
        # _, morphed_encoded = cv2.imencode('.png', morphed_image)
        # st.image(morphed_encoded.tobytes())

        st.write(f"## Total number of grains detected: {total_grains}")

app()
