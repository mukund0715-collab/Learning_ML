# My Machine Learning Journey 🚀

Welcome to my Machine Learning repository! This workspace is dedicated to tracking my progress, logging my experiments, and documenting the core concepts I learn as I dive deeper into machine learning and data science. 

As I build new models and tackle different datasets, I will continually update this file to reflect the preprocessing techniques, algorithms, and mathematical foundations I've mastered.

---

## 📑 Table of Contents
1. [Project 1: MAGIC Gamma Telescope Data Classification](#project-1-magic-gamma-telescope-data-classification)
2. [Project 2: Coming Soon...](#project-2-coming-soon)

---

## Project 1: MAGIC Gamma Telescope Data Classification

### Overview
This project focuses on the classification of high-energy gamma particles using the MAGIC Gamma Telescope dataset. The goal is to accurately distinguish between gamma rays (signal) and hadronic showers (background noise). This section details my approach to data exploration, robust preprocessing, and the evaluation of multiple classification models, ranging from traditional machine learning to deep learning.

### The Dataset
* **Source:** MAGIC Gamma Telescope dataset (`magic04.data`).
* **Features:** 10 continuous features characterizing the image of the showers (e.g., `fLength`, `fWidth`, `fSize`, `fConc`, `fAsym`, `fAlpha`).
* **Target:** The `class` variable indicating whether the event is a gamma ray (`g`) or a hadron (`h`).

### Workflow & Deep-Dive Preprocessing
1. **Data Formatting:** Loaded the `.data` file using `pandas` and explicitly assigned structural column names.
2. **Target Encoding:** Converted the categorical target `class` into binary format (Gamma rays = `1`, Hadrons = `0`).
3. **Data Splitting:** Divided the dataset into Training, Validation, and Testing sets using a `random_state` for strict reproducibility.
4. **Feature Scaling:** Implemented a custom `scale_data` function using `StandardScaler` to ensure distance-based algorithms process all features fairly without bias toward larger numerical scales.
5. **Handling Imbalanced Data:** Applied `RandomOverSampler` strictly to the training data to perfectly balance the classes without causing data leakage into the validation or test sets.

---

### Model Implementations & Results

#### 1. K-Nearest Neighbors (KNN)
* **Overall Accuracy:** **82%**
* **Insights:** Strong performance in detecting actual signals (90% recall for Gamma Rays), but occasionally struggles to distinguish complex background noise (68% recall for Hadrons).

#### 2. Logistic Regression
* **Overall Accuracy:** **79%**
* **Insights:** More balanced precision and recall between the two classes compared to KNN. It correctly identified 72% of the Hadron background noise, outperforming KNN in that specific metric.

#### 3. Naive Bayes
* **Overall Accuracy:** **72%** (Baseline probabilistic approach using Gaussian Naive Bayes)

#### 4. Support Vector Machine (SVM)
* **Overall Accuracy:** **86%** * **Insights:** SVM significantly outperformed the traditional linear models. It achieved an excellent 90% recall for the Gamma Rays while simultaneously jumping to an 80% recall for the tricky Hadron background noise, proving its effectiveness in finding complex decision boundaries.

#### 5. Deep Learning / Neural Network (TensorFlow)
Moved beyond traditional algorithms to build and train a custom Feedforward Neural Network using `tensorflow` and `keras`.
* **Hyperparameter Tuning:** Instead of guessing the best architecture, I implemented a robust grid search to train and evaluate multiple model configurations systematically. I tested combinations of:
  * **Nodes per layer:** 16, 32, 64
  * **Dropout Probability:** 0, 0.2 (to prevent overfitting)
  * **Learning Rate:** 0.01, 0.005, 0.001
  * **Batch Size:** 32, 64, 128
* **Model Selection:** The models were evaluated on the *Validation Set* across 100 epochs. I tracked the validation loss (`val_loss`) and programmatically saved the model that achieved the lowest loss to ensure maximum generalization.
* **Overall Accuracy:** **~87%** (86.96%) *(Best Performing Model!)*
* **Insights:** The neural network, optimized through rigorous hyperparameter tuning, became the strongest performing model in the project, proving the power of deep learning when paired with well-scaled, balanced data.

---

### 🧠 Core Mathematical & Architectural Concepts Mastered
A major focus of this project was understanding the underlying mechanics rather than just using library functions. 

* **Neural Network Architecture:** Understanding the impact of layers, nodes, batch sizes, and learning rates on model convergence.
* **Dropout Regularization:** Using dropout layers to randomly deactivate neurons during training, forcing the network to learn robust features and preventing overfitting.
* **Hyperplanes & Margins (SVM):** Understanding how SVM calculates the decision boundary (hyperplane) that maximizes the distance (margin) between the closest data points.
* **Bayes' Theorem:** The foundation of the Naive Bayes classifier: `P(A|B) = [P(B|A) * P(A)] / P(B)`
* **Euclidean Distance:** The core of the K-Nearest Neighbors algorithm, calculating the straight-line distance between two points in multidimensional space.
* **Z-Score Normalization:** The formula behind the `StandardScaler`: `z = (x - μ) / σ`, ensuring our data features have a mean of 0 and standard deviation of 1.

---

## Project 2: Coming Soon...

*(Details for the next project will be added here.)*
