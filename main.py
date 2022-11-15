import cv2
import os
from utils.utils import save_image, delete_img
import numpy as np
import streamlit as st
from PIL import Image, ImageChops
from matplotlib import pyplot as plt

st.title('Childs\' Drawing Evaluation')

st.sidebar.write("Please input your images of the drawings...")

img1_file = st.sidebar.file_uploader("Source drawing", type=[
    'png', 'jpeg', 'jpg'], key=1)

img2_file = st.sidebar.file_uploader("Childs' drawnig", type=[
    'png', 'jpeg', 'jpg'], key=2)

submit = st.sidebar.button("Submit")

if img1_file and img2_file and submit is not None:
    img1_path = save_image(img1_file)
    img2_path = save_image(img2_file)
    img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)
    # --- take the absolute difference of the images ---
    img1 = cv2.resize(img1, (500, 500))
    img2 = cv2.resize(img2, (500, 500))
    res = cv2.absdiff(img1, img2)

    # --- convert the result to integer type ---
    res = res.astype(np.uint8)
    images = [img1_path, img2_path, res]
    captions = ["Source image", "Childs\' drawing", "Differences"]
    st.image(images, use_column_width=True, caption=captions)

    # st.image(res, caption='Differences between two images')
    # --- find percentage difference based on number of pixels that are not zero ---
    percentage = (np.count_nonzero(res) * 100) / res.size
    st.write("Score: ", (100 - percentage), " %")
    level = ''
    if percentage >= 70.0:
        level += 'Low'
    elif percentage <70.0 and percentage >30.0:
        level += 'Medium'
    elif percentage <= 30.0:
        level += 'High'
    st.write(level)
    delete_img()
