import os
import random
import keras
import numpy as np
from PIL import Image
import tensorflow as tf
from keras import layers
from keras import Sequential
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split

def load_training_images_from_folder(folder):
    images = []
    labels = []
    for class_folder in os.listdir(folder):
        class_folder_path = os.path.join(folder, class_folder)
        if os.path.isdir(class_folder_path):
            label = int(class_folder)
            for file_name in os.listdir(class_folder_path):
                img_path = os.path.join(class_folder_path, file_name)
                with Image.open(img_path) as img:
                    img_array = np.array(img)
                    images.append(img_array)
                    labels.append(label)
    return np.array(images), np.array(labels)

def load_testing_images_from_folder(folder):
    images = []
    labels = []
    for file_name in os.listdir(folder):
        img_path = os.path.join(folder, file_name)
        with Image.open(img_path) as img:
            img_array = np.array(img)
            images.append(img_array)
            labels.append(random.randint(0, 9))
    return np.array(images), np.array(labels)

training_folder = "C:/Users/Nadeem/Desktop/ML-Demo/data/mnistdata/trainingSet/trainingSet"
test_folder = "C:/Users/Nadeem/Desktop/ML-Demo/data/mnistdata/testSamplecls"

images, labels = load_training_images_from_folder(training_folder)
images = images.astype('float32') / 255.0
print(images.shape, labels.shape)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=2)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# test_images = test_images.astype('float32') / 255.0#

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
#
#Compiling and fitting model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer= keras.optimizers.Adam(learning_rate=1e-3),
    metrics=['accuracy']
)
model.fit(X_train, y_train, epochs=15, batch_size=32)
#
#Pridiction
y_hat = model.predict(X_test)
y_hat = tf.nn.softmax(y_hat).numpy()
y_hat = np.argmax(y_hat, axis=1)
(print(accuracy_score(y_hat, y_test)))
#
#
#Evaluation
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_hat, average='micro')
print("Precision:", precision)
print("Recall:", recall)
# Calculate TPR (Recall) for each class
TPR = recall
print("True Positive Rate OR sensitivity (Recall):", TPR)
FPR = 1 - precision
print("False Positive Rate OR specificity:", FPR)
