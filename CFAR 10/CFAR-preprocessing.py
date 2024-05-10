import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import keras
from keras import Sequential
from keras import layers
import os

def load_images_from_folder(folder):
    images = []
    for file_name in os.listdir(folder):
        img_path = os.path.join(folder, file_name)
        with Image.open(img_path) as img:
            img_array = np.array(img)
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(img_array)
            gray_image = pil_image.convert('L')
            img_array = np.array(gray_image)
            images.append(img_array)
    return np.array(images)

# def load_images_from_folder(folder):
#     images = []
#     for file_name in os.listdir(folder):
#         img_path = os.path.join(folder, file_name)
#         with Image.open(img_path) as img:
#             img_array = np.array(img)
#             images.append(img_array)
#     return np.array(images)
def image_normalization(arr):
    normalized_images = []
    for image in arr:
        image = image.astype('float32') / 255
        normalized_images.append(image)
    return normalized_images

data = pd.read_csv("C:/Users/Nadeem/Desktop/ML-Demo/data/CFAR 10 data/trainLabels.csv")

seed = 123
training_data_path = "C:/Users/Nadeem/Desktop/ML-Demo/data/CFAR 10 data/training/train"

le = LabelEncoder()
data["label"] = le.fit_transform(data["label"])

images = load_images_from_folder(training_data_path)
X = images.astype('float32')/255
y = data["label"]
print(y.dtype)

model = Sequential()
model.add(layers.Conv2D(6, kernel_size=(5,5), activation='relu', padding='same', strides=(1, 1), input_shape=(32,32,1)))
model.add(layers.MaxPooling2D(pool_size=(1,1), padding='valid', strides=(1,1)))
model.add(layers.Conv2D(16, kernel_size=(5,5), activation='relu', padding='valid', strides=(2,2)))
model.add(layers.AveragePooling2D(pool_size=(2,2), padding='valid', strides=(1,1)))
model.add(layers.Conv2D(6, kernel_size=(5,5), activation='relu', padding='same', strides=(1, 1)))
model.add(layers.MaxPooling2D(pool_size=(2,2), padding='valid', strides=(1,1)))
model.add(layers.Flatten())
model.add(layers.Dense(120, activation='relu'))
model.add(layers.Dense(84, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

#Model Summary
model.summary()
#
#Compiling and fitting model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer= keras.optimizers.Adam(learning_rate=1e-3),
    metrics=['accuracy']
)

model.fit(X, y, epochs=10, batch_size=32)
