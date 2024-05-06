#importing dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import Sequential
import tensorflow as tf

#Defining a custom class for creating dense layers
class Dense:
    def __init__(self, units, activation=None):
        self.units = units
        self.activation = activation


#loading Data and splitting data into train and test
data = pd.read_csv('data/churn.csv')
X = pd.get_dummies(data.drop(['Churn', 'Customer ID'], axis=1))
y = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Defining custom Layers
dense_layer_1 = tf.keras.layers.Dense(units=32, activation='relu', input_dim=len(X_train.columns))
dense_layer_2 = tf.keras.layers.Dense(units=64, activation='relu')
dense_layer_3 = tf.keras.layers.Dense(units=1, activation='sigmoid')

#Defining model
model = Sequential()
model.add(dense_layer_1)
model.add(dense_layer_2)
model.add(dense_layer_3)

#Checking model summary
model.summary()

#Compiling and fitting model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=32)

#Predicting
y_hat = model.predict(X_test)
y_hat = [0 if x < 0.5 else 1 for x in y_hat]
print(y_hat)

