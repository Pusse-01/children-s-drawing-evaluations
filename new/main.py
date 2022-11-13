import cv2
import os
import image_similarity_measures
from sys import argv
from image_similarity_measures.quality_metrics import rmse, ssim, sre
import streamlit as st
from utils import save_image, delete_img

st.title('Childs\' Drawing Evaluation')

st.sidebar.write("Please input your images of the drawings...")



img1_file = st.sidebar.file_uploader("Source drawing", type=[
    'png', 'jpeg', 'jpg'], key=1)

img2_file = st.sidebar.file_uploader("Childs' drawnig", type=[
    'png', 'jpeg', 'jpg'], key=2)

submit = st.sidebar.button("Submit")

ssim_measures = {}
rmse_measures = {}
sre_measures = {}

def calc_closest_val(dict, checkMax):
	result = {}
	if (checkMax):
		closest = max(dict.values())
		
	else:
		closest = min(dict.values())
	for key, value in dict.items():
		print("The difference between ", key ," and the original image is : \n", value)
		if (value == closest):
			result[key] = closest
			
	print("The closest value: ", closest)	    
	print("######################################################################")
	return result
    


if img1_file and img2_file and submit is not None:
    img1_path = save_image(img1_file)
    img2_path = save_image(img2_file)
    test_img = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    data_img = cv2.imread(img2_path, cv2.IMREAD_COLOR)
    scale_percent = 100 # percent of original img size
    width = int(test_img.shape[1] * scale_percent / 100)
    height = int(test_img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_img = cv2.resize(data_img, dim, interpolation = cv2.INTER_AREA)
    ssim_measures['Drawing']= ssim(test_img, resized_img)
    rmse_measures['Drawing']= rmse(test_img, resized_img)
    sre_measures['Drawing']= sre(test_img, resized_img)
    ssim = calc_closest_val(ssim_measures, True)
    rmse = calc_closest_val(rmse_measures, False)
    sre = calc_closest_val(sre_measures, True)

    st.write("The most similar according to SSIM: " , ssim)
    st.write("The most similar according to RMSE: " , rmse)
    st.write("The most similar according to SRE: " , sre)

    delete_img() 
