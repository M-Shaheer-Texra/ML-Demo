import os
import random
import keras
import numpy as np
from PIL import Image
import tensorflow as tf
from keras import layers
from keras import Sequential
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

train_set = keras.utils.image_dataset_from_directory(
    directory="C:/Users/Nadeem/Desktop/ML-Demo/data/mnistdata/trainingSet/trainingSet",
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(28, 28),
)

test_set = keras.utils.image_dataset_from_directory(
    directory="C:/Users/Nadeem/Desktop/ML-Demo/data/mnistdata/testSet",
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(28, 28),
)

print(test_set.class_names)

# def load_training_images_from_folder(folder):
#     images = []
#     labels = []
#     for class_folder in os.listdir(folder):
#         class_folder_path = os.path.join(folder, class_folder)
#         if os.path.isdir(class_folder_path):
#             label = int(class_folder)
#             for file_name in os.listdir(class_folder_path):
#                 img_path = os.path.join(class_folder_path, file_name)
#                 with Image.open(img_path) as img:
#                     img_array = np.array(img)
#                     images.append(img_array)
#                     labels.append(label)
#     return np.array(images), np.array(labels)
#
# def load_testing_images_from_folder(folder):
#     images = []
#     labels = []
#     for file_name in os.listdir(folder):
#         img_path = os.path.join(folder, file_name)
#         with Image.open(img_path) as img:
#             img_array = np.array(img)
#             images.append(img_array)
#             labels.append(random.randint(0, 9))
#     return np.array(images), np.array(labels)
#
# training_folder = "C:/Users/Nadeem/Desktop/ML-Demo/data/mnistdata/trainingSet/trainingSet"
# test_folder = "C:/Users/Nadeem/Desktop/ML-Demo/data/mnistdata/testSet/testSet"
#
# train_images, train_labels = load_training_images_from_folder(training_folder)
# test_images, test_labels = load_testing_images_from_folder(test_folder)
#

#Designing a random CNN:
model = Sequential()
model.add(layers.Conv2D(6, kernel_size=(5,5), activation='relu', padding='same', strides=(1, 1), input_shape=(28, 28 ,1)))
model.add(layers.MaxPooling2D(pool_size=(1,1), padding='valid', strides=(1,1)))
model.add(layers.Conv2D(16, kernel_size=(5,5), activation='relu', padding='valid', strides=(2,2)))
model.add(layers.AveragePooling2D(pool_size=(2,2), padding='valid', strides=(1,1)))
model.add(layers.Conv2D(6, kernel_size=(5,5), activation='relu', padding='same', strides=(1, 1), input_shape=(28, 28 ,1)))
model.add(layers.MaxPooling2D(pool_size=(2,2), padding='valid', strides=(1,1)))
model.add(layers.Flatten())
model.add(layers.Dense(120, activation='relu'))
model.add(layers.Dense(84, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

#Model Summary
model.summary()

#Compiling and fitting model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer= keras.optimizers.Adam(learning_rate=1e-3),
    metrics=['accuracy']
)
model.fit(train_set, epochs=15, batch_size=32)

#Pridiction
y_hat = model.predict(test_set)
y_hat = tf.nn.softmax(y_hat).numpy()
y_hat = np.argmax(y_hat, axis=1)
print(model.evaluate(test_set, verbose=2))
#
#
# #Evaluation
# precision, recall, fscore, support = precision_recall_fscore_support(test_labels, y_hat, average='micro')
# print("Precision:", precision)
# print("Recall:", recall)
# # Calculate TPR (Recall) for each class
# TPR = recall
# print("True Positive Rate OR sensitivity (Recall):", TPR)
# FPR = 1 - precision
# print("False Positive Rate OR specificity:", FPR)
