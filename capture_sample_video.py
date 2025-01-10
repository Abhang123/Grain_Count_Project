import streamlit as st
import numpy as np
import tempfile
import cv2
import json
from streamlit_lottie import st_lottie


def process_frame(frame):
    # Resize the frame
    frame_resized = cv2.resize(frame, (600, 600))

    # Convert the frame to HSV
    hsv_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)

    # Define HSV ranges for red color
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    brown_lower = np.array([15, 200, 150])
    brown_upper = np.array([20, 255, 150])

    black_lower = np.array([0, 0, 0])
    black_upper = np.array([180, 255, 50])

    # Create masks for specific regions
    mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    mask3 = cv2.inRange(hsv_frame, brown_lower, brown_upper)
    mask4 = cv2.inRange(hsv_frame, black_lower, black_upper)

    # Combine masks
    red_mask = cv2.bitwise_or(mask1, mask2)
    red_mask = cv2.bitwise_or(red_mask, mask3)
    red_mask = cv2.bitwise_or(red_mask, mask4)

    # Apply the mask to the original frame
    filtered_frame = cv2.bitwise_and(frame_resized, frame_resized, mask=red_mask)

    gray_frame = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # Apply morphological operation (Top-Hat)
    kernel = np.ones((15, 15), np.uint8)
    top_hat = cv2.morphologyEx(blurred_frame, cv2.MORPH_TOPHAT, kernel)

    # Apply adaptive threshold
    thresh_frame = cv2.adaptiveThreshold(
        top_hat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 2
    )

    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morphed_frame = cv2.morphologyEx(thresh_frame, cv2.MORPH_CLOSE, kernel)

    # Find and filter contours
    contours, _ = cv2.findContours(morphed_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 50
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # Count and draw contours
    total_grains = len(filtered_contours)
    frame_copy = frame_resized.copy()
    for cnt in filtered_contours:
        cv2.drawContours(frame_copy, [cnt], -1, (0, 255, 0), 2)

    # Add total count to the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame_copy, f"Total Count: {total_grains}", (20, 50), font, 1, (255, 0, 0), 2)

    return frame_copy, total_grains

def app():
    st.title("Computer Vision App - Video Processing")

    with open("anima1.json") as source:
        animation = json.load(source)
    st_lottie(animation, width=300)

    video_file = st.file_uploader("Upload a video containing grains to count them.", type=["mp4", "avi", "mov"])

    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        frame_count = 0
        total_frames_processed = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Process every nth frame for efficiency
            if frame_count % 5 == 0:
                processed_frame, grain_count = process_frame(frame)
                stframe.image(processed_frame, caption=f"Processed Frame with {grain_count} grains")

                total_frames_processed += 1

        cap.release()
        st.write(f"Total frames processed: {total_frames_processed}")
app()