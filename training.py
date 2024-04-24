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

IMGWIDTH = 224

def InceptionLayer(x, a, b, c, d):
    x1 = Conv2D(a, (1, 1), padding='same', activation='relu')(x)
    
    x2 = Conv2D(b, (1, 1), padding='same', activation='relu')(x)
    x2 = Conv2D(b, (3, 3), padding='same', activation='relu')(x2)
    
    x3 = Conv2D(c, (1, 1), padding='same', activation='relu')(x)
    x3 = Conv2D(c, (3, 3), dilation_rate=2, strides=1, padding='same', activation='relu')(x3)
    
    x4 = Conv2D(d, (1, 1), padding='same', activation='relu')(x)
    x4 = Conv2D(d, (3, 3), dilation_rate=3, strides=1, padding='same', activation='relu')(x4)

    y = Concatenate(axis=-1)([x1, x2, x3, x4])
    
    return y

def MesoInception4(learning_rate=0.001):
    x = Input(shape=(IMGWIDTH, IMGWIDTH, 3))
    
    x1 = InceptionLayer(x, 1, 4, 4, 2)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
    
    x2 = InceptionLayer(x1, 2, 4, 4, 2)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)        
    
    x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
    x3 = BatchNormalization()(x3)
    x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
    
    x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
    x4 = BatchNormalization()(x4)
    x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)
    
    y = Flatten()(x4)
    y = Dropout(0.5)(y)
    y = Dense(16)(y)
    y = LeakyReLU(alpha=0.1)(y)
    y = Dropout(0.5)(y)
    y = Dense(1, activation='sigmoid')(y)
    
    model = Model(inputs=x, outputs=y)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

model = MesoInception4()




image_dimensions = {'height':256, 'width':256, 'channels':3}
x = []
y = []
folder_path = "./Train/Fake"

c = 0 
for file in os.listdir(folder_path):
    if file.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        image_path = os.path.join(folder_path, file)
        img = Image.open(image_path)
        img = img.resize((224,224))
        x.append(np.array(img))
        y.append(0)
        print(c)
        c=c+1
folder_path = "./Train/Real"


for file in os.listdir(folder_path):
    if file.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        image_path = os.path.join(folder_path, file)
        img = Image.open(image_path)
        img = img.resize((224,224))
        x.append(np.array(img))
        y.append(1)
        print(c)
        c=c+1
       
combined = list(zip(x, y))
random.shuffle(combined)
x, y = zip(*combined)
x = np.array(x)
y= np.array(y)
x_train = x[:60000]
# x_test = x[75000:]
y_train = y[:60000]
# y_test = y[75000:]
model.fit(x_train,y_train , epochs=20)
model.save("m_icept.h5")