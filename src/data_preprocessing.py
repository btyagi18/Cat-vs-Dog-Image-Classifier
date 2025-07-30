import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

IMG_SIZE = 100 
CATEGORIES = ['Cat', 'Dog']

def load_data(data_dir):
    data = []
    for category in CATEGORIES:
        path = os.path.join(data_dir, category)
        class_num = CATEGORIES.index(category)

        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                data.append((img, class_num))
            except Exception as e:
                continue
    return data

def preprocess_data(data):
    np.random.shuffle(data)
    X = []
    y = []
    for features, label in data:
        X.append(features)
        y.append(label)
    X = np.array(X) / 255.0
    y = np.array(y)
    return train_test_split(X, y, test_size=0.2, random_state=42)

