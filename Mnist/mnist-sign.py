import csv
# from keras_tuner import RandomSearch
import keras
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras import layers
# from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_data = "C:/Users/Nadeem/Desktop/ML-Demo/data/MNist hand sign data/sign_mnist_train/sign_mnist_train.csv"
test_data = "C:/Users/Nadeem/Desktop/ML-Demo/data/MNist hand sign data/sign_mnist_test/sign_mnist_test.csv"

def parse_data_from_csv(filename):
  with open(filename) as file:
    csv_reader = csv.reader(file, delimiter=',')
    next(csv_reader)
    labels_list = []
    images_list = []
    for row in csv_reader:
      labels_list.append(row[0])
      image = np.array(row[1:]).reshape((28, 28))
      images_list.append(image)
    labels = np.array(labels_list).astype('float')
    images = np.array(images_list).astype('float')
    return images, labels


train_images, train_labels = parse_data_from_csv(train_data)
test_images, test_labels = parse_data_from_csv(test_data)

def train_test_generators(train_images, train_labels, test_images, test_labels):
  train_images = np.expand_dims(train_images, axis=3)
  test_images = np.expand_dims(test_images, axis=3)

  train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
  train_generator = train_datagen.flow(x=train_images,
                                       y=train_labels,
                                       batch_size=32)

  test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      rescale=1./255
      )
  test_generator = test_datagen.flow(x=test_images,
                                     y=test_labels,
                                     batch_size=32)

  return train_generator, test_generator


train_generator, test_generator = train_test_generators(train_images, train_labels, test_images, test_labels)

model = Sequential([
          layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
          layers.MaxPooling2D(2, 2),
          layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
          layers.MaxPooling2D(2, 2),
          layers.Conv2D(64, (3,3), activation='relu'),
          layers.MaxPooling2D(2,2),
          layers.Flatten(),
          layers.Dense(128, activation='relu'),
          layers.Dense(128, activation='relu'),
          layers.Dense(25, activation='softmax')
      ])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
model.fit(
    train_generator,
    epochs=20,
    validation_data=train_generator)
model.save("model.keras")
load_model = keras.saving.load_model("model.keras")
load_model.evaluate(test_generator)

def rgb_to_grayscale(rgb_image):
    grayscale_image = np.dot(rgb_image[...,:3], [0.2989, 0.5870, 0.1140])
    return grayscale_image


def display_prediction(img_path):
  img = keras.preprocessing.image.load_img(img_path, target_size=(28, 28))
  img_array = keras.preprocessing.image.img_to_array(img)
  grayscale_image = rgb_to_grayscale(img_array)
  grayscale_image = grayscale_image / 255
  x = grayscale_image.reshape((1, 28, 28, 1))
  prediction = load_model.predict(x)
  print(prediction.argmax(axis=1))


img_path = "C:/Users/Nadeem/Desktop/ML-Demo/images/a-hand-gesture-of-the-raise-hand-or-a-sign-language-of-b-collection-of-the-sign-language-using-hand-gestures-free-photo.jpg"
display_prediction(img_path)

# performing a simple hyperparameter tuning using kerad tuner(hyperparameters are not learned by the machine)
def build_model(hp):
  model = Sequential([
    keras.layers.Conv2D(
        filters = hp.Int("conv_1_filter", min_value=32, max_value=128, step=16),
        kernel_size = hp.Choice("conv_1_kernel", values = [3,5]),
        activation = "relu",
        input_shape = (28,28,1)
    ),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(
        filters = hp.Int("conv_2_filter", min_value=32, max_value=64, step=16),
        kernel_size = hp.Choice("conv_2_kernel", values = [3,5]),
        activation = "relu"
    ),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(
        units = hp.Int("dense_1_units", min_value=32, max_value=128, step=16),
        activation = "relu"
    ),
    keras.layers.Dense(25, activation = "softmax")
  ])


  model.compile(
      optimizer = keras.optimizers.Adam(hp.Choice("learning_rate", values = [1e-2, 1e-3])),
      loss = "sparse_categorical_crossentropy",
      metrics = ["accuracy"]
  )
  return model
#
# tuner = RandomSearch(build_model,
#                     objective='val_accuracy',
#                     max_trials = 5)
# tuner.search(train_generator,epochs=3,validation_data=(test_generator))
# models = tuner.get_best_models(num_models=2)
# best_model = models[0]
# best_model.summary()
#
# best_hps = tuner.get_best_hyperparameters()[0]
# model_tuned = build_model(best_hps)
# model_tuned.fit(train_generator, epochs=20, validation_data=test_generator)