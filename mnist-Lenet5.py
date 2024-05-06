#importing dependencies
import keras
import numpy as np
import tensorflow as tf
from keras import datasets
from keras import Sequential
from keras import layers
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

#Loading and splitting data
data = datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = data
#Designing model
model = Sequential()
model.add(layers.Conv2D(6, kernel_size=(5, 5), activation='tanh', padding='valid', strides=(1, 1), input_shape=(28, 28, 1)))
model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
model.add(layers.Conv2D(16, kernel_size=(5, 5), activation='tanh', padding='valid', strides=(1, 1)))
model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
model.add(layers.Flatten())
#Fully Connected Layers
model.add(layers.Dense(120, activation='tanh'))
model.add(layers.Dense(84, activation='tanh'))
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
print("True Positive Rate (Recall) for each class:")
print(TPR)
FPR = 1 - precision
print("False Positive Rate for each class :")
print(FPR)
