import cv2
import os
import glob
import shutil
import numpy as np
from PIL import Image, ImageChops
from matplotlib import pyplot as plt

# def save_image(image_file):
#     file_path = os.path.join("data", image_file)
#     with open(file_path, "wb") as f:
#         f.write(image_file.getbuffer())
#     return file_path


# def delete_img():
#     folder = '././data'
#     for filename in os.listdir(folder):
#         file_path = os.path.join(folder, filename)
#         try:
#             if os.path.isfile(file_path) or os.path.islink(file_path):
#                 os.unlink(file_path)
#             elif os.path.isdir(file_path):
#                 shutil.rmtree(file_path)
#         except Exception as e:
#             print('Failed to delete %s. Reason: %s' % (file_path, e))

def predict(img1_file, img2_file):
    # img1_path = save_image(img1_file)
    # img2_path = save_image(img2_file)
    img1 = cv2.imread(img1_file, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img2_file, cv2.IMREAD_COLOR)
    # --- take the absolute difference of the images ---
    img1 = cv2.resize(img1, (500, 500))
    img2 = cv2.resize(img2, (500, 500))
    res = cv2.absdiff(img1, img2)

    # --- convert the result to integer type ---
    res = res.astype(np.uint8)
    # st.image(res, caption='Differences between two images')
    # --- find percentage difference based on number of pixels that are not zero ---
    percentage = (np.count_nonzero(res) * 100) / res.size
    result = "Difference: "+ str(percentage)+ " %"
    # delete_img()
    return result

print(predict('1.png', '2.jpg'))
