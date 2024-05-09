import pandas as pd
import numpy as np
from PIL import Image
import keras
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
def image_normalization(arr):
    normalized_images = []
    for image in arr:
        image = image.astype('float32') / 255
        normalized_images.append(image)
    return normalized_images

data = pd.read_csv("C:/Users/Nadeem/Desktop/ML-Demo/data/CFAR 10 data/trainLabels.csv")
print(data["label"].value_counts())
print(data["label"].unique())

seed = 123
training_data_path = "C:/Users/Nadeem/Desktop/ML-Demo/data/CFAR 10 data/training/train"


images = load_images_from_folder(training_data_path)

normalized_images = image_normalization(images)
data["normalized_images"] = normalized_images
print(data.head(1))


