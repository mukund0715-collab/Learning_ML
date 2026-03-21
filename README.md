# Learning_ML
Have a naq about ML but moving to complete basic and learning from start
## My Machine Learning Journey 🚀

Welcome to my Machine Learning repository! This workspace is dedicated to tracking my progress, logging my experiments, and documenting the core concepts I learn as I dive deeper into machine learning and data science. 

As I build new models and tackle different datasets, I will continually update this file to reflect the preprocessing techniques, algorithms, and insights I've gained.

---

## 📑 Table of Contents
1. [Project 1: MAGIC Gamma Telescope Data Classification](#project-1-magic-gamma-telescope-data-classification)
2. [Project 2: Coming Soon...](#project-2-coming-soon)
3. [Project 3: Coming Soon...](#project-3-coming-soon)

---

## Project 1: MAGIC Gamma Telescope Data Classification

### Overview
This project focuses on the classification of high-energy gamma particles using the MAGIC Gamma Telescope dataset. The goal is to accurately distinguish between gamma rays (signal) and hadronic showers (background noise). This section details my approach to data exploration and the robust preprocessing techniques required before feeding data into machine learning models.

### The Dataset
* **Source:** MAGIC Gamma Telescope dataset (`magic04.data`).
* **Features:** 10 continuous features characterizing the image of the showers (e.g., `fLength`, `fWidth`, `fSize`, `fConc`, `fAsym`, `fAlpha`).
* **Target:** The `class` variable indicating whether the event is a gamma ray (`g`) or a hadron (`h`).

### Workflow & Deep-Dive Preprocessing

#### 1. Data Loading and Formatting
* Imported necessary libraries (`numpy`, `pandas`, `matplotlib`, `sklearn`, `imblearn`).
* Read the raw `.data` file using `pandas`. I explicitly defined and assigned meaningful column names to give the raw data a readable, tabular structure for analysis.

#### 2. Target Encoding
Machine learning algorithms perform mathematical operations and require numerical inputs. 
* **What I did:** Converted the categorical target variable `class` into a binary format. Gamma rays ('g') were mapped to `1`, and hadrons ('h') to `0`.

#### 3. Data Splitting (`train_test_split`)
The dataset was divided into three distinct subsets to ensure the model could be trained and evaluated fairly:
* **Training Set:** The data the model actually learns from.
* **Validation Set:** Used to tweak and tune the model's settings (hyperparameters) without exposing it to the final test data.
* **Testing Set:** Kept completely isolated until the very end to test how the model performs on truly unseen real-world data.
* **The Role of `random_state`:** When splitting the data, I used a `random_state` parameter. This acts as a seed for the random number generator, ensuring *reproducibility*. It guarantees that every time the script is run, the data is shuffled and split in the exact same way, allowing me to accurately measure if changes to my code actually improved the model.

#### 4. Feature Scaling (`StandardScaler`)
* **What I did:** Implemented a custom `scale_data` function using `sklearn`'s `StandardScaler`.
* **Why it's crucial:** The features in this dataset have vastly different scales (e.g., `fLength` values are much larger than `fConc` values). Distance-based ML algorithms (like K-Nearest Neighbors) are highly sensitive to these differences. `StandardScaler` standardizes the features by removing the mean and scaling them to unit variance (mean = 0, standard deviation = 1). This levels the playing field, ensuring no single feature dominates the model's decision-making simply because its raw numbers are larger.

#### 5. Handling Imbalanced Data (`RandomOverSampler`)
* **What I did:** Applied `RandomOverSampler` from the `imblearn` library.
* **Why it's crucial:** In many real-world datasets, one class heavily outnumbers the other. If a model trains on heavily imbalanced data, it might just learn to guess the majority class every time. The oversampler fixes this by duplicating random examples from the minority class until both classes are equally represented.
* **Important Note:** I intentionally applied this oversampling *only* to the training set. Applying it to the validation or test sets would cause "data leakage" and ruin the integrity of the final evaluation.

#### 6. Final Data Preparation
The perfectly scaled and resampled features (`x`) and targets (`y`) were horizontally stacked back together using `np.hstack`. This resulted in clean, optimized numpy arrays ready to be fed directly into predictive algorithms.

### Next Steps / Model Implementations
### Model Implementations & Results

#### 1. K-Nearest Neighbors (KNN)
The first algorithm applied to the fully preprocessed dataset was the K-Nearest Neighbors (KNN) classifier. 

**How it works:**
KNN is a simple, distance-based algorithm. When given a new, unseen data point, it looks at the 'k' closest data points (neighbors) in the training set. The new point is then assigned the class that is most common among those neighbors.

**Implementation Details:**
* **Library:** Used `KNeighborsClassifier` from `sklearn.neighbors`.
* **Hyperparameter:** I initialized the model with `n_neighbors=1`. This means the algorithm makes its prediction based *only* on the single closest data point in the training set. 

**Model Evaluation:**
To evaluate the model's performance on the completely unseen testing set, I used `sklearn.metrics.classification_report`. 

**Results (Accuracy: 82%):**
* **Class 1 (Gamma Rays - Signal):**
  * **Precision (0.84):** When the model predicted a particle was a gamma ray, it was correct 84% of the time.
  * **Recall (0.90):** The model successfully identified 90% of all actual gamma rays in the test set.
  * **F1-Score (0.87):** An excellent balance between precision and recall for the signal class.
* **Class 0 (Hadrons - Background):**
  * **Precision (0.78):** When predicting a hadron, it was correct 78% of the time.
  * **Recall (0.68):** It successfully identified 68% of the actual background noise.
  * **F1-Score (0.73):** Slightly lower performance compared to the signal class, indicating the model occasionally struggles to distinguish complex background noise from actual signals.

**Key Takeaway:**
With an overall accuracy of 82% using just `n_neighbors=1`, the baseline KNN model performs strongly, particularly in correctly identifying the critical gamma ray signals (90% recall). Future experiments could involve hyperparameter tuning (testing different values for 'k') to see if the background noise recall can be improved without sacrificing signal accuracy.

---

## Project 2: Coming Soon...

*(Details for the next project will be added here, following a similar structure of dataset overview, preprocessing deep-dives, and model results.)*

---

