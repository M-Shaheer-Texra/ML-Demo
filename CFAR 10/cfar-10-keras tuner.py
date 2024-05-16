import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import Sequential



(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
plt.imshow(X_train[0], interpolation='nearest')
plt.show()
def train_test_generator(X_train,y_train,X_test,y_test):
    train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    train_generator = train_data_gen.flow(
        x=X_train,
        y=y_train,
        batch_size=32
    )
    test_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
    )
    test_generator = test_data_gen.flow(
        x=X_test,
        y=y_test,
        batch_size=32
    )
    return train_generator, test_generator


training_data, validation_data = train_test_generator(X_train,y_train,X_test,y_test)

def build_hyper_model(hp):
  model = keras.Sequential()
  num_layers = hp.Int('num_layers', min_value=1, max_value=10, step=1)
  for i in range(num_layers):
    model.add(keras.layers.Conv2D(
       filters=hp.Int("conv_1_filter" + str(i), min_value=32, max_value=128, step=16),
            kernel_size=hp.Choice("conv_1_kernel", values=[3, 5]),
            activation="relu",
            input_shape=(32, 32, 3)
    ))
    model.add(keras.layers.Dense(
     units = hp.Int("dense_units", min_value=32, max_value=128, step=16),
      activation= hp.Choice('dense_activation', values=['relu', 'tanh'])
    ))
    model.add(keras.layers.Dense(10, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice("learning_rate", values=[1e-2, 1e-3])),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])

  return model
#
# tuner = RandomSearch(build_hyper_model,
#                     objective='val_accuracy',
#                      max_trials = 4)
#
# tuner.search(training_data,epochs=3,validation_data=(validation_data.x,validation_data.y))
#
# best_hps = tuner.get_best_hyperparameters()[0]
# model_tuned = build_hyper_model(best_hps)
# model_tuned.summary()