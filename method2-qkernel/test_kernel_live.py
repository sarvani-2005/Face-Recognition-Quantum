import cv2
import numpy as np
import joblib

from preprocess_kernel import preprocess_image, load_pca
from quantum_kernel import kernel

# Load PCA
load_pca()

clf, X_train, label_map = joblib.load("models/kernel_model.pkl")


def compute_test_kernel(x, X_train):
    return np.array([kernel(x, x_train) for x_train in X_train]).reshape(1, -1)


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

print("Press Q to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (32, 32))

        temp_path = "temp.jpg"
        cv2.imwrite(temp_path, face)

        features = preprocess_image(temp_path)

        K_test = compute_test_kernel(features, X_train)

        probs = clf.predict_proba(K_test)[0]
        confidence = max(probs) * 100
        pred = np.argmax(probs)

        label = label_map[pred]
        text = f"{label} ({confidence:.1f}%)"

        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0,255,0), 2)

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow("Quantum Kernel Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()