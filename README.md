\# Quantum Face Recognition



A Quantum Machine Learning based face recognition project that implements and compares two quantum approaches for face classification:



\* \*\*Variational Quantum Classifier (VQC)\*\*

\* \*\*Quantum Kernel + Support Vector Machine (SVM)\*\*



\## 🎯 Objective



To explore and compare quantum machine learning approaches for face recognition using image preprocessing, dimensionality reduction, quantum circuits, and classical machine learning.



\## 🧠 Approaches



\### 1. Variational Quantum Classifier (VQC)



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

VQC

&#x20;   ↓

Prediction

```



\### 2. Quantum Kernel + SVM



```text

Face Image

&#x20;   ↓

Grayscale Conversion

&#x20;   ↓

Image Resizing

&#x20;   ↓

Feature Extraction

&#x20;   ↓

PCA

&#x20;   ↓

Quantum Kernel

&#x20;   ↓

SVM

&#x20;   ↓

Prediction

```



\## 📊 Dataset



The project uses a \*\*custom face image dataset\*\* containing \*\*2 classes\*\*, with approximately \*\*40 images per class\*\*.



The dataset is \*\*not included in this repository\*\*.



Expected structure:



```text

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

```



The images are converted to grayscale, resized, and transformed into numerical features before being passed to the quantum models.



\## 📈 Results



| Method               | Accuracy |    AUC |

| -------------------- | -------: | -----: |

| VQC                  |      70% | 0.6689 |

| Quantum Kernel + SVM |      60% | 0.6044 |



The \*\*VQC approach achieved better performance\*\* on the selected dataset.



The repository also contains the generated evaluation and comparison visualizations.



\## 📁 Project Structure



```text

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

├── final\_comparison\_chart.png

├── radar\_comparison.png

├── requirements.txt

└── README.md

```



\## 🛠️ Technologies



\* Python

\* OpenCV

\* NumPy

\* Pandas

\* Scikit-learn

\* PennyLane

\* PCA

\* SVM

\* Matplotlib

\* Seaborn

\* Joblib



\## ⚙️ Installation



Clone the repository:



```bash

git clone https://github.com/sarvani-2005/Face-Recognition-Quantum.git

cd Face-Recognition-Quantum

```



Install the required dependencies:



```bash

pip install -r requirements.txt

```



\## ▶️ Running the Project



\### Method 1 – VQC



```bash

cd method1-vqc

python train.py

python live\_test.py

```



\### Method 2 – Quantum Kernel



```bash

cd method2-qkernel

python train\_kernel.py

python test\_kernel\_live.py

```



\## 📷 Real-Time Recognition



Both approaches support webcam-based face recognition using their respective trained quantum models.



\## 🚀 Future Improvements



\* Increase the size and diversity of the dataset

\* Improve feature extraction

\* Integrate Vision Transformer (ViT) based feature extraction

\* Experiment with additional quantum classifiers

\* Test on larger face datasets

\* Improve real-time recognition performance



\## 👩‍💻 Author



\*\*Sarvani Kosaraju\*\*



B.Tech – Artificial Intelligence and Machine Learning



