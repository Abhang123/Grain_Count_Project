from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np 
import cv2
import streamlit as st 
from PIL import Image
import json
from streamlit_lottie import st_lottie

model = YOLO(r"E:\AI\Computer Vision\Sample_Project\Count_Analysis\Arhar_Daal\runs\detect\grain_count_train3\weights\best.pt")

st.title("Welcome to :violet[Computer Vision App]")

st.write("\n")
st.write("\n")

with open("anima1.json") as source:
    animation = json.load(source)
st_lottie(animation, width = 300)

st.write("\n")
st.write("\n")

image1 = st.camera_input("Upload your first image of red grams.")

if image1 is not None:

    image2 = st.camera_input("Upload your second image of red grams.")

    if image2 is not None:

        image3 = st.camera_input("Upload your third image of red grams.")

        if image3 is not None:
            
            i1 = Image.open(image1)
            i2 = Image.open(image2)
            i3 = Image.open(image3)

            result1 = model.predict(source = i1, save = False, conf = 0.05)
            result2 = model.predict(source = i2, save = False, conf = 0.05)
            result3 = model.predict(source = i3, save = False, conf = 0.05)

        
            count1 = len(result1[0])
            count2 = len(result2[0])
            count3 = len(result3[0])

            print(f"Total Grain Count of i1-> {len(result1[0])}")

            print("\n")

            print(f"Total Grain Count i2-> {len(result2[0])}")

            print("\n")

            print(f"Total Grain Count i3-> {len(result3[0])}")

            print("\n")

            avg_count = (count1 + count2 + count3) / 3

            annotated_image1 = result1[0].plot()
            annotated_image2 = result2[0].plot()
            annotated_image3 = result3[0].plot()

            st.write("\n")
            st.write("\n")

            if (count1 > count2 > count3):
                st.image(annotated_image1,caption="Grain Count")

            elif (count2 > count1 > count3):
                st.image(annotated_image1,caption="Grain Count")

            else:
                st.image(annotated_image1,caption="Grain Count")

            
            st.write(f"### Total Grain Count -> {int(avg_count)}")
else:

    st.write("### Please cpature the first image of red grams.")
