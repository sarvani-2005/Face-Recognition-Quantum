import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

# Load saved results
y_test, y_pred = joblib.load("models/kernel_results.pkl")

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot
plt.figure(figsize=(5,5))
plt.imshow(cm, cmap='Blues')

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# Add numbers inside boxes
for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i][j],
                 ha='center', va='center', fontsize=14)

plt.colorbar()
plt.tight_layout()

plt.savefig("confusion_matrix.png")  # saves image
plt.show()