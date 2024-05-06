#Importing dependencies
import keras
import numpy as np
import tensorflow as tf
from keras import layers
from keras import Sequential
from keras import datasets
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

#Loading and splitting data
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

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
model.fit(x_train, y_train, epochs=15, batch_size=32)

#Pridiction
y_hat = model.predict(x_test)
y_hat = tf.nn.softmax(y_hat).numpy()
y_hat = np.argmax(y_hat, axis=1)
print(model.evaluate(x_test, y_test))

#Evaluation
conf_matrix = confusion_matrix(y_test, y_hat)
print("Confusion Matrix:")
print(conf_matrix)
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_hat, average='micro')
print("Precision:", precision)
print("Recall:", recall)
# Calculate TPR (Recall) for each class
TPR = recall
print("True Positive Rate OR sensitivity (Recall):", TPR)
FPR = 1 - precision
print("False Positive Rate OR specificity:", FPR)
