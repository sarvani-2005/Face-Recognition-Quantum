\# Quantum Face Recognition



A Quantum Machine Learning based face recognition project that implements and compares two quantum approaches for face classification:



\- Variational Quantum Classifier (VQC)

\- Quantum Kernel with Support Vector Machine (SVM)



\## Overview



This project explores the application of Quantum Machine Learning techniques to face image classification and compares the performance of two different quantum approaches.



\## Dataset



A custom face image dataset was used for the experiments.



\- Number of classes: 2

\- Images per class: approximately 40

\- Total images: approximately 80



The dataset is not included in this repository.



The expected dataset structure is:



&#x20;   dataset/

&#x20;   ├── Person\_1/

&#x20;   │   ├── image1.jpg

&#x20;   │   ├── image2.jpg

&#x20;   │   └── ...

&#x20;   └── Person\_2/

&#x20;       ├── image1.jpg

&#x20;       ├── image2.jpg

&#x20;       └── ...



The images are converted to grayscale, resized, and transformed into numerical features before being used by the quantum models.



\## Methods



\### 1. Variational Quantum Classifier (VQC)



The first approach uses a Variational Quantum Circuit for face classification.



Workflow:



&#x20;   Face Image

&#x20;       ↓

&#x20;   Grayscale Conversion

&#x20;       ↓

&#x20;   Image Resizing

&#x20;       ↓

&#x20;   Feature Extraction

&#x20;       ↓

&#x20;   Feature Scaling

&#x20;       ↓

&#x20;   Quantum Circuit

&#x20;       ↓

&#x20;   VQC

&#x20;       ↓

&#x20;   Prediction



\### 2. Quantum Kernel + SVM



The second approach uses a quantum kernel to map image features into a quantum feature space. The resulting kernel is then used with an SVM classifier.



Workflow:



&#x20;   Face Image

&#x20;       ↓

&#x20;   Grayscale Conversion

&#x20;       ↓

&#x20;   Image Resizing

&#x20;       ↓

&#x20;   Feature Extraction

&#x20;       ↓

&#x20;   PCA

&#x20;       ↓

&#x20;   Quantum Kernel

&#x20;       ↓

&#x20;   SVM

&#x20;       ↓

&#x20;   Prediction



\## Results



The two approaches were evaluated using accuracy and AUC, along with confusion matrices, ROC curves, and precision-recall curves.



| Method | Accuracy | AUC |

| --- | ---: | ---: |

| VQC | 70% | 0.6689 |

| Quantum Kernel + SVM | 60% | 0.6044 |



The VQC approach achieved better performance on the selected dataset.



\## Project Structure



&#x20;   Face-Recognition-Quantum/

&#x20;   ├── method1-vqc/

&#x20;   │   ├── preprocess.py

&#x20;   │   ├── quantum\_model.py

&#x20;   │   ├── train.py

&#x20;   │   ├── live\_test.py

&#x20;   │   ├── quantum\_model.pkl

&#x20;   │   └── scale.pkl

&#x20;   │

&#x20;   ├── method2-qkernel/

&#x20;   │   ├── preprocess\_kernel.py

&#x20;   │   ├── quantum\_kernel.py

&#x20;   │   ├── train\_kernel.py

&#x20;   │   ├── test\_kernel\_live.py

&#x20;   │   ├── plot\_results.py

&#x20;   │   ├── kernel\_model.pkl

&#x20;   │   ├── kernel\_results.pkl

&#x20;   │   └── pca.pkl

&#x20;   │

&#x20;   ├── compare\_methods.py

&#x20;   ├── final\_comparison\_chart.png

&#x20;   ├── radar\_comparison.png

&#x20;   ├── requirements.txt

&#x20;   └── README.md



\## Technologies Used



\- Python

\- OpenCV

\- NumPy

\- Pandas

\- Scikit-learn

\- PennyLane

\- PCA

\- Support Vector Machine

\- Matplotlib

\- Seaborn

\- Joblib



\## Installation



Clone the repository:



&#x20;   git clone https://github.com/sarvani-2005/Face-Recognition-Quantum.git

&#x20;   cd Face-Recognition-Quantum



Install the required dependencies:



&#x20;   pip install -r requirements.txt



\## Running the Project



\### VQC



&#x20;   cd method1-vqc

&#x20;   python train.py

&#x20;   python live\_test.py



\### Quantum Kernel + SVM



&#x20;   cd method2-qkernel

&#x20;   python train\_kernel.py

&#x20;   python test\_kernel\_live.py



\## Real-Time Face Recognition



Both approaches include webcam-based testing. The captured face is processed using the corresponding preprocessing pipeline and classified using the trained model.



\## Future Improvements



\- Increase the size and diversity of the dataset

\- Improve feature extraction

\- Integrate Vision Transformer (ViT) based feature extraction

\- Experiment with additional quantum classifiers

\- Evaluate the models on larger face datasets

\- Improve real-time recognition performance



\## Author



Sarvani Kosaraju



B.Tech - Artificial Intelligence and Machine Learning

