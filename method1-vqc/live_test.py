import cv2
import numpy as np
import pennylane as qml
import joblib

# ===============================
# SETTINGS
# ===============================
n_qubits = 4
n_layers = 3

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)

    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

# ===============================
# LOAD MODEL
# ===============================
weights, label_map = joblib.load("models/quantum_model.pkl")
max_val = joblib.load("models/scale.pkl")

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
        face = face.flatten() / 255.0

        face = face[:n_qubits]
        face = face / max_val * np.pi

        pred = circuit(face, weights)
        prob = (pred + 1) / 2

        # Unknown threshold
        if prob > 0.75:
            label = label_map[1]
        elif prob < 0.25:
            label = label_map[0]
        else:
            label = "Unknown"

        confidence = prob * 100

        # Draw box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        text = f"{label} ({confidence:.1f}%)"
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,255,0), 2)

    cv2.imshow("Quantum Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()