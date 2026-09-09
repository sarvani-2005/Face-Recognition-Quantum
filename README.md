\# Quantum Face Recognition



A Quantum Machine Learning based face recognition project that implements and compares two different quantum approaches for face classification:



\- \*\*Variational Quantum Classifier (VQC)\*\*

\- \*\*Quantum Kernel with Support Vector Machine (SVM)\*\*



The project explores how quantum machine learning techniques can be applied to face image classification and compares their performance using standard evaluation metrics.



\---



\## 🎯 Objective



The main objectives of this project are:



\- To implement face recognition using quantum machine learning.

\- To explore a Variational Quantum Classifier for classification.

\- To implement a Quantum Kernel combined with an SVM classifier.

\- To preprocess and reduce image features for quantum processing.

\- To evaluate and compare the performance of both approaches.



\---



\## 🧠 Method 1: Variational Quantum Classifier (VQC)



The first approach uses a Variational Quantum Circuit for face classification.



\### Workflow



```text

Face Image

&#x20;   ↓

Grayscale Conversion

&#x20;   ↓

Image Resizing

&#x20;   ↓

Feature Extraction

&#x20;   ↓

Feature Scaling

&#x20;   ↓

Quantum Circuit

&#x20;   ↓

Variational Quantum Classifier

&#x20;   ↓

Prediction

Main files

preprocess.py – image preprocessing and feature preparation

quantum\_model.py – quantum classifier implementation

train.py – model training

live\_test.py – real-time webcam testing

Saved models

quantum\_model.pkl

scale.pkl

⚛️ Method 2: Quantum Kernel + SVM



The second approach uses a quantum kernel to represent similarity between samples in a quantum feature space. The resulting kernel is used with a Support Vector Machine for classification.



Workflow

Face Image

&#x20;   ↓

Grayscale Conversion

&#x20;   ↓

Image Resizing

&#x20;   ↓

Feature Extraction

&#x20;   ↓

PCA Dimensionality Reduction

&#x20;   ↓

Quantum Kernel

&#x20;   ↓

SVM Classifier

&#x20;   ↓

Prediction

Main files

preprocess\_kernel.py – preprocessing and feature preparation

quantum\_kernel.py – quantum kernel implementation

train\_kernel.py – model training

test\_kernel\_live.py – real-time webcam testing

plot\_results.py – result visualization

Saved models

kernel\_model.pkl

kernel\_results.pkl

pca.pkl

📊 Dataset



The project uses a two-class face image dataset, with approximately 40 images per class.



The dataset is not included in this repository.



Expected structure:



dataset/

├── Person\_1/

│   ├── image1.jpg

│   ├── image2.jpg

│   └── ...

│

└── Person\_2/

&#x20;   ├── image1.jpg

&#x20;   ├── image2.jpg

&#x20;   └── ...

📈 Results



The two approaches were evaluated using accuracy, AUC, confusion matrix, ROC curve, and precision-recall analysis.



Method	Accuracy	AUC

VQC	70%	0.6689

Quantum Kernel + SVM	60%	0.6044



Based on the current experiment, the VQC approach achieved better performance than the Quantum Kernel approach on the selected dataset.



The repository contains the generated evaluation graphs and comparison visualizations.



🛠️ Technologies Used

Python

NumPy

Pandas

OpenCV

Scikit-learn

Matplotlib

Seaborn

PennyLane

Joblib

PCA

Support Vector Machine

Variational Quantum Circuits

Quantum Kernels

⚙️ Installation



Clone the repository:



git clone <YOUR\_REPOSITORY\_URL>

cd Face-Recognition-Quantum



Create a virtual environment:



python -m venv qenv



Activate it on Windows:



qenv\\Scripts\\activate



Install dependencies:



pip install -r requirements.txt

▶️ Running the Project

Method 1 – VQC

cd method1-vqc

python train.py

python live\_test.py

Method 2 – Quantum Kernel

cd method2-qkernel

python train\_kernel.py

python test\_kernel\_live.py

📷 Real-Time Face Recognition



Both approaches include webcam-based testing. The captured face is processed using the same preprocessing pipeline used during training and then classified using the trained quantum model.



🚀 Future Improvements

Increase the size and diversity of the dataset.

Explore stronger feature extraction techniques.

Integrate Vision Transformer (ViT) based feature extraction.

Experiment with additional quantum classifiers.

Test the approaches on larger face datasets.

Improve real-time recognition performance.

👩‍💻 Author



Sarvani Kosaraju



B.Tech – Artificial Intelligence and Machine Learning

