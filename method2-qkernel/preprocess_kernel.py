import cv2
import numpy as np
from sklearn.decomposition import PCA
import joblib

# Global PCA object
pca = PCA(n_components=4)

def fit_pca(image_paths):
    data = []

    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (32, 32))
        data.append(img.flatten())

    data = np.array(data) / 255.0

    pca.fit(data)

    # Save PCA for later use
    joblib.dump(pca, "models/pca.pkl")


def load_pca():
    global pca
    pca = joblib.load("models/pca.pkl")


def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (32, 32))
    img = img.flatten() / 255.0

    features = pca.transform([img])[0]

    # Normalize to [0, π]
    max_val = np.max(np.abs(features))
    if max_val != 0:
        features = features / max_val * np.pi

    return features