import cv2
import numpy as np
import tensorflow as tf

model_path = './model.h5'

test_img = './data/3338.png'

class_names = ['cloud',
 'sun',
 'pants',
 'umbrella',
 'table',
 'ladder',
 'eyeglasses',
 'clock',
 'scissors',
 'cup']

def predict(image_path, model_path):
    model = tf.keras.models.load_model(model_path)
    IMG_SIZE = 28
    img_array = cv2.imread(image_path, 0)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE), 1)
    new_arr = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    predicted = model.predict(new_arr)
    y_pred = np.argmax(predicted, axis = 1)
    return(class_names[y_pred[0]])

print(predict(test_img, model_path))