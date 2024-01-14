import numpy as np
import pandas as pd
import cv2
from google.colab.patches import cv2_imshow

def make_kmeans(n_neighbor=2, img_name='ski.jpg'):

    # wczytanie zdjęcia
    img = cv2.imread(img_name)
    cv2_imshow(img)

    # przygotowanie zdjęcia
    img_data = img.reshape((-1, 3))
    img_data = np.float32(img_data)

    # kmeans
    _, label, center = cv2.kmeans(
        data=img_data, 
        K=2, 
        bestLabels=None,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 
        attempts=10, 
        flags=cv2.KMEANS_RANDOM_CENTERS)

    # przygotowanie do wyświetlenia
    center = np.uint8(center)
    res = center[label.flatten()]
    res = res.reshape((img.shape))
    cv2_imshow(res)

make_kmeans()

