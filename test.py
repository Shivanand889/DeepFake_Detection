import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model , Sequential
import os
from PIL import Image
import random
import os
import pickle

def extract_faces(image_path, output_folder , name):
    # Load pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    image = cv2.imread(image_path, 1)
    gray_images = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # converting to grayscale
    face_images = face_cascade.detectMultiScale(gray_images, 1.3, 5)

    try:
        for (x, y, w, h) in face_images:
            region_of_interest = image[y:y+h, x:x+w]
        resized = cv2.resize(region_of_interest, (128, 128))
        cv2.imwrite(f"{output_folder}/{name}.jpg", resized)    
    except:
        print("No Faces Detected.")

model = load_model("m_icept.h5")

extract_faces("src/public/x.jpg","src/public" ,"x")
data = float(model(x)[0][0])