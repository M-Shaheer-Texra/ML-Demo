from PIL import Image
import keras
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from keras import layers
from keras import Sequential
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# def rbg2gray(images):
#     grey_images = []
#     for image in images:
#         img_gray = ImageOps.grayscale(image)
#         grey_images.append(img_gray)
#     return grey_images

data_set_path = "C:/Users/Nadeem/Desktop/ML-Demo/data/mnistdata/trainingSet/trainingSet"

train_data, test_data = keras.utils.image_dataset_from_directory(
    directory=data_set_path,
    labels="inferred",
    label_mode="categorical",
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="both",
    image_size=(28, 28),
    batch_size=32,
)

x, y = zip(*train_data)
X_train = np.concatenate(x)
y_train = np.concatenate(y)
x, y = zip(*test_data)
X_test = np.concatenate(x)
y_test = np.concatenate(y)

y_train = y_train.argmax(axis=1)
y_test = y_test.argmax(axis=1)

#normalize these images
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

class_names = np.array(train_data.class_names)
print(class_names)

my_input_shape = (28, 28, 3)
num_classes = len(class_names)

initial_lr = 0.0001

model = keras.Sequential([
    layers.InputLayer(my_input_shape),

    # Convulutional Block
    layers.Conv2D(filters=8, kernel_size=3, strides=2, padding='valid', activation='relu'),
    layers.Conv2D(filters=16, kernel_size=3, strides=2, padding='same', activation='relu'),
    layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
    layers.Flatten(),

    # Fully Connected Block
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=initial_lr),
                     loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(
    train_data,
    epochs=10,
    batch_size=32
)
#Pridiction
y_hat = model.predict(X_test)
y_hat = tf.nn.softmax(y_hat).numpy()
y_hat = np.argmax(y_hat, axis=1)
print(accuracy_score(y_hat, y_test))

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
