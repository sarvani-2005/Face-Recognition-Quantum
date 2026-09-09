import os
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)

from preprocess_kernel import preprocess_image, fit_pca
from quantum_kernel import kernel


def load_dataset(path):
    image_paths = []
    labels = []
    label_map = {}
    idx = 0

    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)

        if not os.path.isdir(folder_path):
            continue

        label_map[idx] = folder

        for img in os.listdir(folder_path):
            image_paths.append(os.path.join(folder_path, img))
            labels.append(idx)

        idx += 1

    return image_paths, np.array(labels), label_map


def compute_kernel_matrix(X1, X2):
    K = np.zeros((len(X1), len(X2)))

    for i in range(len(X1)):
        for j in range(len(X2)):
            K[i][j] = kernel(X1[i], X2[j])

    return K


def train():
    print("Loading dataset...")
    image_paths, y, label_map = load_dataset("dataset")

    print("Fitting PCA...")
    fit_pca(image_paths)

    print("Extracting features...")
    X = np.array([preprocess_image(p) for p in image_paths])

    print("Class distribution:", np.bincount(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=42
    )

    print("Computing kernel matrix...")
    K_train = compute_kernel_matrix(X_train, X_train)
    K_test = compute_kernel_matrix(X_test, X_train)

    clf = SVC(
        kernel='precomputed',
        C=10,
        probability=True
    )

    clf.fit(K_train, y_train)

    y_pred = clf.predict(K_test)
    y_scores = clf.predict_proba(K_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n===== RESULTS =====")
    print("Accuracy:", acc)

    print("\nConfusion Matrix:\n", cm)

    print(
        "\nClassification Report:\n",
        classification_report(y_test, y_pred)
    )

    joblib.dump(
        (clf, X_train, label_map),
        "models/kernel_model.pkl"
    )

    joblib.dump(
        (y_test, y_pred),
        "models/kernel_results.pkl"
    )

    print("\nModel saved successfully!")

    # ====================================
    # RESEARCH PAPER VISUALIZATIONS
    # ====================================

    # 1 Confusion Matrix
    plt.figure(figsize=(6,5))
    plt.imshow(cm, cmap="Greens")
    plt.colorbar()

    for i in range(len(cm)):
        for j in range(len(cm[0])):
            plt.text(j, i, cm[i][j], ha='center', va='center')

    plt.title("Method 2 - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("method2_confusion_matrix.png")
    plt.show()

    # 2 ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    print("AUC:", roc_auc)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1], [0,1], '--')

    plt.title("Method 2 - ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    plt.savefig("method2_roc_curve.png")
    plt.show()

    # 3 Precision Recall Curve
    precision, recall, _ = precision_recall_curve(
        y_test,
        y_scores
    )

    plt.figure(figsize=(6,5))
    plt.plot(recall, precision)

    plt.title("Method 2 - Precision Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.savefig("method2_precision_recall.png")
    plt.show()

    # 4 Class Distribution
    unique, counts = np.unique(y, return_counts=True)

    plt.figure(figsize=(6,5))
    plt.bar(unique, counts)

    plt.title("Method 2 - Class Distribution")
    plt.xlabel("Classes")
    plt.ylabel("Count")

    plt.savefig("method2_class_distribution.png")
    plt.show()

    # 5 Actual vs Predicted Comparison
    plt.figure(figsize=(8,5))
    plt.plot(y_test[:20], label="Actual")
    plt.plot(y_pred[:20], label="Predicted")

    plt.title("Method 2 - Actual vs Predicted")
    plt.legend()

    plt.savefig("method2_actual_vs_predicted.png")
    plt.show()


if __name__ == "__main__":
    train()