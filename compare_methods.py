import matplotlib.pyplot as plt
import numpy as np

# Method 1 - VQC
method1 = {
    "Accuracy": 0.70,
    "Precision": 0.71,
    "Recall": 0.70,
    "F1": 0.70,
    "AUC": 0.6689
}

# Method 2 - Quantum Kernel
method2 = {
    "Accuracy": 0.60,
    "Precision": 0.60,
    "Recall": 0.60,
    "F1": 0.60,
    "AUC": 0.6044
}

metrics = list(method1.keys())

m1_values = list(method1.values())
m2_values = list(method2.values())

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10,6))

plt.bar(
    x - width/2,
    m1_values,
    width,
    label="Method 1 - VQC"
)

plt.bar(
    x + width/2,
    m2_values,
    width,
    label="Method 2 - QKernel"
)

plt.xticks(x, metrics)
plt.ylabel("Score")
plt.title("Performance Comparison of Quantum Face Recognition Methods")
plt.legend()

plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.savefig("final_comparison_chart.png", dpi=300)
plt.show()