import pandas as pd
import numpy as np
import csv
from PIL import Image
from sklearn.model_selection import train_test_split
import keras
from keras import Sequential
from keras import layers
import tensorflow as tf
import os

raw_data = pd.read_csv("C:/Users/Nadeem/Desktop/ML-Demo/data/MNist hand sign data/sign_mnist_train/sign_mnist_train.csv")
column_name = raw_data["label"].unique()

training_data = "C:/Users/Nadeem/Desktop/ML-Demo/data/MNist hand sign data/sign_mnist_train/sign_mnist_train.csv"
validation_data = "C:/Users/Nadeem/Desktop/ML-Demo/data/MNist hand sign data/sign_mnist_test/sign_mnist_test.csv"

def parse_data_from_input(filename):

  with open(filename) as file:

    csv_reader = csv.reader(file, delimiter=',')
    #
    next(csv_reader)

    labels_list = []
    images_list = []

    for row in csv_reader:
        # The first item is the label
        labels_list.append(int(row[0]))
        # The remaining items are the pixel values, reshaped from 784 to 28x28
        image = np.array(row[1:], dtype=np.float64).reshape(28, 28)
        images_list.append(image)

    # Convert lists to numpy arrays with type float64
    labels = np.array(labels_list, dtype=np.float64)
    images = np.array(images_list, dtype=np.float64)


    return images, labels

training_images, training_labels = parse_data_from_input(training_data)
validation_images, validation_labels = parse_data_from_input(validation_data)
print(training_images.shape)
print(training_labels.shape)
print(validation_images.shape)
print(validation_labels.shape)

input_shape = training_images.shape

model = Sequential()
initial_lr = 0.0001

model = keras.Sequential([
    layers.InputLayer(input_shape),

    # Convulutional Block
    layers.Conv2D(filters=8, kernel_size=3, strides=2, padding='valid', activation='relu'),
    layers.Conv2D(filters=16, kernel_size=3, strides=2, padding='same', activation='relu'),
    layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
    layers.Flatten(),

    # Fully Connected Block
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(len(column_name), activation='softmax')
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=initial_lr),
                     loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(
    training_images, training_labels,
    epochs=10,
    batch_size=32
)