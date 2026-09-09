import os
import cv2
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
from pennylane import numpy as pnp
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_curve,
    auc,
    precision_recall_curve
)
import joblib

# ===============================
# SETTINGS
# ===============================
n_qubits = 4
n_layers = 3
epochs = 50
learning_rate = 0.05

dev = qml.device("default.qubit", wires=n_qubits)

# ===============================
# DATASET LOADER
# ===============================
def load_dataset(path):
    X = []
    y = []
    label_map = {}
    label_index = 0

    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)

        if not os.path.isdir(folder_path):
            continue

        label_map[label_index] = folder

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (32, 32))
            img = img.flatten() / 255.0

            X.append(img)
            y.append(label_index)

        label_index += 1

    return np.array(X), np.array(y), label_map


# ===============================
# QUANTUM CIRCUIT
# ===============================
@qml.qnode(dev)
def circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)

    qml.templates.StronglyEntanglingLayers(
        weights,
        wires=range(n_qubits)
    )

    return qml.expval(qml.PauliZ(0))


# ===============================
# TRAINING FUNCTION
# ===============================
def train():
    print("Loading dataset...")

    X, y, label_map = load_dataset("dataset")

    # Binary classification only
    if len(label_map) != 2:
        raise ValueError("Dataset must contain exactly 2 classes.")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    # Dimensionality reduction
    X_train = X_train[:, :n_qubits]
    X_test = X_test[:, :n_qubits]

    # Normalize
    max_val = np.max(np.abs(X_train))
    X_train = X_train / max_val * np.pi
    X_test = X_test / max_val * np.pi

    joblib.dump(max_val, "models/scale.pkl")

    # Initialize weights
    weights = pnp.random.randn(
        n_layers,
        n_qubits,
        3,
        requires_grad=True
    )

    opt = qml.AdamOptimizer(learning_rate)

    loss_values = []

    def cost(weights):
        loss = 0

        for i in range(len(X_train)):
            pred = circuit(X_train[i], weights)
            pred = (pred + 1) / 2
            loss += (pred - y_train[i]) ** 2

        return loss / len(X_train)

    # ===============================
    # Training Loop
    # ===============================
    for epoch in range(epochs):
        weights = opt.step(cost, weights)

        current_loss = cost(weights)
        loss_values.append(current_loss)

        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {current_loss:.4f}")

    # ===============================
    # Evaluation
    # ===============================
    y_pred = []
    y_scores = []

    for x in X_test:
        pred = circuit(x, weights)
        prob = (pred + 1) / 2

        y_scores.append(prob)
        y_pred.append(1 if prob > 0.5 else 0)

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print("\n===== CONFUSION MATRIX =====")
    print(cm)

    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\nAccuracy:", acc)

    # Save model
    joblib.dump((weights, label_map), "models/quantum_model.pkl")

    print("\nModel Saved Successfully!")

    # =========================================
    # RESEARCH PAPER VISUALIZATIONS
    # =========================================

    # 1 Confusion Matrix Plot
    plt.figure(figsize=(6,5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()

    for i in range(len(cm)):
        for j in range(len(cm[0])):
            plt.text(j, i, cm[i][j], ha='center', va='center')

    plt.title("Method 1 - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("method1_confusion_matrix.png")
    plt.show()

    # 2 Training Loss Curve
    plt.figure(figsize=(6,5))
    plt.plot(loss_values)
    plt.title("Method 1 - Training Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("method1_loss_curve.png")
    plt.show()

    # 3 ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    print("AUC:", roc_auc)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], '--')

    plt.title("Method 1 - ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    plt.savefig("method1_roc_curve.png")
    plt.show()

    # 4 Precision Recall Curve
    precision, recall, _ = precision_recall_curve(
        y_test,
        y_scores
    )

    plt.figure(figsize=(6,5))
    plt.plot(recall, precision)

    plt.title("Method 1 - Precision Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.savefig("method1_precision_recall.png")
    plt.show()

    # 5 Class Distribution
    unique, counts = np.unique(y, return_counts=True)

    plt.figure(figsize=(6,5))
    plt.bar(unique, counts)

    plt.title("Method 1 - Class Distribution")
    plt.xlabel("Classes")
    plt.ylabel("Count")

    plt.savefig("method1_class_distribution.png")
    plt.show()


if __name__ == "__main__":
    train()