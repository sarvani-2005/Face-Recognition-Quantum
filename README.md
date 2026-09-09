\# Quantum Face Recognition



A Quantum Machine Learning based face recognition project that implements and compares two quantum approaches for face classification:



\- \*\*Variational Quantum Classifier (VQC)\*\*

\- \*\*Quantum Kernel + Support Vector Machine (SVM)\*\*



\## 🎯 Objective



To explore and compare quantum machine learning approaches for face recognition using image preprocessing, dimensionality reduction, quantum circuits, and classical machine learning.



\## 🧠 Approaches



\### 1. Variational Quantum Classifier (VQC)



```text

Face Image → Grayscale → Resize → Feature Extraction

→ Scaling → Quantum Circuit → VQC → Prediction

2\. Quantum Kernel + SVM

Face Image → Grayscale → Resize → Feature Extraction

→ PCA → Quantum Kernel → SVM → Prediction

📊 Results

Method	Accuracy	AUC

VQC	70%	0.6689

Quantum Kernel + SVM	60%	0.6044



The VQC approach achieved better performance on the selected dataset.



📁 Project Structure

Face-Recognition-Quantum/

│

├── method1-vqc/

│   ├── preprocess.py

│   ├── quantum\_model.py

│   ├── train.py

│   └── live\_test.py

│

├── method2-qkernel/

│   ├── preprocess\_kernel.py

│   ├── quantum\_kernel.py

│   ├── train\_kernel.py

│   ├── test\_kernel\_live.py

│   └── plot\_results.py

│

├── compare\_methods.py

├── requirements.txt

└── README.md

🛠️ Technologies



Python • OpenCV • NumPy • Scikit-learn • PennyLane • PCA • SVM • Matplotlib



▶️ Installation

git clone https://github.com/sarvani-2005/Face-Recognition-Quantum.git

cd Face-Recognition-Quantum

pip install -r requirements.txt

📷 Real-Time Recognition



Both approaches support webcam-based face recognition using their respective trained models.



🚀 Future Improvements

Larger and more diverse datasets

Improved feature extraction

Vision Transformer (ViT) integration

Additional quantum classifiers

Improved real-time performance

👩‍💻 Author



Sarvani Kosaraju

B.Tech – Artificial Intelligence and Machine Learning

