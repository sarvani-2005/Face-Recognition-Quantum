import cv2
import numpy as np

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to 4x4
    img = cv2.resize(img, (4, 4))

    # Normalize to [0, π]
    img = img / 255.0 * np.pi

    # Divide into 4 blocks (2x2 blocks)
    block1 = np.mean(img[0:2, 0:2])
    block2 = np.mean(img[0:2, 2:4])
    block3 = np.mean(img[2:4, 0:2])
    block4 = np.mean(img[2:4, 2:4])

    return np.array([block1, block2, block3, block4])