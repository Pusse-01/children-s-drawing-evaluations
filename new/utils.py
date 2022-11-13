import os
import shutil
import glob


def save_image(image_file):
    file_details = {"FileName": image_file.name, "FileType": image_file.type}
    file_path = os.path.join("data", image_file.name)
    with open(file_path, "wb") as f:
        f.write(image_file.getbuffer())
    return file_path


def delete_img():
    folder = '././data'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))