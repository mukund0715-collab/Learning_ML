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
This project focuses on the classification of high-energy gamma particles using the MAGIC Gamma Telescope dataset. The goal is to accurately distinguish between gamma rays (signal) and hadronic showers (background noise). This section details my approach to data exploration, robust preprocessing, and the evaluation of multiple classification models built from the ground up.

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
KNN is a distance-based algorithm that classifies a new data point based on the majority class of its 'k' closest neighbors in the training data.
* **Configuration:** `n_neighbors=1`
* **Overall Accuracy:** **82%**
* **Insights:** Strong performance in detecting actual signals (90% recall for Gamma Rays), but occasionally struggles to distinguish complex background noise (68% recall for Hadrons).

#### 2. Logistic Regression
A fundamental linear classification algorithm that estimates the probability of an event occurring using a logistic (sigmoid) function.
* **Overall Accuracy:** **79%**
* **Insights:** More balanced precision and recall between the two classes compared to KNN. It correctly identified 72% of the Hadron background noise, outperforming KNN in that specific metric.

#### 3. Naive Bayes
A probabilistic classifier based on applying Bayes' theorem with strong (naive) independence assumptions between the features. Given that our dataset features are continuous, Gaussian Naive Bayes was utilized.
* **Overall Accuracy:** **[XX%]**
* **Performance Breakdown:**
  * **Gamma Rays (Signal):** Precision: `[XX]`, Recall: `[XX]`
  * **Hadrons (Background):** Precision: `[XX]`, Recall: `[XX]`
* **Insights:** *(Add a brief sentence here about how it compared to the others. For example: "While the independence assumption is 'naive' for physical dimensions, the model still provided a strong probabilistic baseline...")*

---

### 🧠 Core Mathematical Concepts Mastered
A major focus of this project was understanding the mathematical "core" of the algorithms rather than just using library functions. 

* **Bayes' Theorem:** The foundation of the Naive Bayes classifier. It calculates the probability of a hypothesis (class) given prior knowledge. 
  * Formula: `P(A|B) = [P(B|A) * P(A)] / P(B)`
  * *Application:* Calculating the probability that an event is a Gamma Ray given its specific feature measurements.
* **Euclidean Distance:** The core of the K-Nearest Neighbors algorithm, calculating the straight-line distance between two points in multidimensional space to find the closest matches.
* **The Sigmoid Function:** The mathematical curve used in Logistic Regression to map any real-valued number into a value between 0 and 1, representing a probability.
* **Z-Score Normalization:** The formula behind the `StandardScaler`: `z = (x - μ) / σ`, ensuring our data features have a mean of 0 and standard deviation of 1.

---

## Project 2: Coming Soon...

*(Details for the next project will be added here.)*
