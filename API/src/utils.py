import os
import shutil
import cv2
import numpy as np
from PIL import Image as im
import json

#single message
class Model():
    class_name = os.path.basename(__file__)

    def __init__(self) -> None:
        self.dir()
    


        #temp locations
    def dir(self):
        temp="temp/"
        os.makedirs(temp,exist_ok=True)
        source="temp/src_img"
        os.makedirs(source,exist_ok=True)
        drawing="temp/draw_img"
        os.makedirs(drawing,exist_ok=True)


    def save_image(self,image_file,path):
        # file_details = {"FileName": image_file.name, "FileType": image_file.type}
        file_location = f"{path}/{image_file.filename}"
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(image_file.file, file_object)  


    
    def predict(self,source_file,drawing_file):
        self.save_image(source_file,"temp/src_img")
        self.save_image(drawing_file,"temp/draw_img")

        img1_path=f"temp/src_img/{source_file.filename}"
        img2_path=f"temp/draw_img/{drawing_file.filename}"

        img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
        img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)
        # --- take the absolute difference of the images ---
        img1 = cv2.resize(img1, (500, 500))
        img2 = cv2.resize(img2, (500, 500))
        res = cv2.absdiff(img1, img2)

        # --- convert the result to integer type ---
        res = res.astype(np.uint8)


        images = [img1_path, img2_path, res]
        
        # --- find percentage difference based on number of pixels that are not zero ---
        percentage = (np.count_nonzero(res) * 100) / res.size
        # st.write("Difference: ", percentage, " %")

        res = im.fromarray(res)
        res.save('result.png')

        shutil.rmtree("temp")
        return percentage



### sample request

# import requests

# headers = {
#     'accept': 'application/json',
#     # requests won't add a boundary if this header is set when you pass files=
#     # 'Content-Type': 'multipart/form-data',
# }

# files = {
#     'source_file': open('fire (1).png;type=image/png', 'rb'),
#     'drawing_file': open('fire.png;type=image/png', 'rb'),
# }

# response = requests.post('http://127.0.0.1:8000/check-difference', headers=headers, files=files)
