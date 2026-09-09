\# 🔬 Quantum Face Recognition



A \*\*Quantum Machine Learning\*\* project that implements and compares two approaches for face classification:



\* \*\*Variational Quantum Classifier (VQC)\*\*

\* \*\*Quantum Kernel + Support Vector Machine (SVM)\*\*



\## 📌 Overview



Face images are preprocessed and converted into numerical features before being classified using quantum-based models.



```text

Face Image

&#x20;   ↓

Grayscale → Resize → Feature Extraction

&#x20;   ↓

&#x20;┌───────────────┬────────────────────┐

&#x20;↓               ↓                    ↓

&#x20;VQC        Quantum Kernel → SVM

&#x20;↓               ↓

&#x20;└───────────────┴────────────────────┘

&#x20;             Prediction

```



\## 📂 Dataset



A custom dataset containing \*\*2 classes with approximately 40 images per class\*\* was used.



```text

dataset/

├── Person\_1/

└── Person\_2/

```



The dataset is not included in this repository.



\## ⚛️ Methods



\*\*VQC:\*\* Uses a parameterized quantum circuit for face classification.



\*\*Quantum Kernel + SVM:\*\* Uses PCA for dimensionality reduction, followed by a quantum kernel and classical SVM classifier.



\## 📊 Results



| Method               | Accuracy |        AUC |

| -------------------- | -------: | ---------: |

| \*\*VQC\*\*              |  \*\*70%\*\* | \*\*0.6689\*\* |

| Quantum Kernel + SVM |      60% |     0.6044 |



VQC achieved better performance on the selected dataset.



Evaluation includes \*\*confusion matrices, ROC curves, precision-recall curves, and comparison charts\*\*.



> Results are experimental due to the small dataset size.



\## 🛠️ Technologies



\*\*Python · OpenCV · NumPy · Pandas · Scikit-learn · PennyLane · PCA · SVM · Matplotlib · Seaborn · Joblib\*\*



\## ⚙️ Setup



```bash

git clone https://github.com/sarvani-2005/Face-Recognition-Quantum.git

cd Face-Recognition-Quantum

pip install -r requirements.txt

```



Add the dataset using the structure shown above.



\### Run VQC



```bash

cd method1-vqc

python train.py

python live\_test.py

```



\### Run Quantum Kernel + SVM



```bash

cd method2-qkernel

python train\_kernel.py

python test\_kernel\_live.py

```



Both methods include \*\*webcam-based testing\*\*.



\## 🚀 Future Improvements



\* Larger and more diverse datasets

\* Improved feature extraction

\* ViT-based feature extraction

\* Additional quantum classifiers

\* Improved real-time performance



\## 👩‍💻 Author



\*\*Sarvani Kosaraju\*\*

B.Tech — Computer Science \& Engineering



