# CodeAlpha Machine Learning Internship Projects 🚀

Welcome to my repository for the **1-Month Machine Learning Internship** at **CodeAlpha**. This repository contains end-to-end Python implementations, model architectures, and performance evaluations for 4 machine learning projects spanning classification, deep learning audio processing, computer vision, and medical diagnostic modeling.

---

## 📌 Internship Overview

* **Company:** CodeAlpha
* **Program:** Machine Learning Internship
* **Duration:** 1 Month
* **Repository Name:** `CodeAlpha_MachineLearning_Tasks`
* **Core Stack:** Python, Pandas, NumPy, Scikit-Learn, TensorFlow/Keras, Matplotlib, Seaborn

---

## 📋 Summary of Internship Tasks

### 1. Task 1: Credit Scoring Model
* **Objective:** Predict an individual's creditworthiness and financial default risk using supervised classification algorithms.
* **Dataset Used:** [German Credit Risk Dataset](https://www.kaggle.com/datasets/uciml/german-credit) (Kaggle / UCI Machine Learning Repository)
* **Approach & Methods:** 
  * Imputed missing financial/account categories.
  * Encoded categorical attributes via One-Hot Encoding and scaled numerical features.
  * Trained **Logistic Regression** and **Random Forest Classifier** models.
  * Applied **GridSearchCV** for optimal hyperparameter tuning.
* **Key Evaluation:** Evaluated using Precision, Recall, F1-Score, and achieved an **ROC-AUC score of ~0.998**.

---

### 2. Task 2: Emotion Recognition from Speech
* **Objective:** Classify human emotional states (e.g., Happy, Sad, Angry, Neutral) from raw speech audio signals.
* **Dataset Used:** [RAVDESS / TESS Audio Datasets](https://www.kaggle.com/datasets/uwrv/ravdess-emotional-speech-audio)
* **Approach & Methods:** 
  * Extracted **MFCCs (Mel-Frequency Cepstral Coefficients)** and audio spectral features using `librosa`.
  * Designed Deep Learning architectures (**1D-CNN / LSTM**) to capture sequential speech audio patterns.
  * Trained and cross-validated multiclass classifiers.
* **Key Evaluation:** Evaluated via Confusion Matrix and Categorical Classification Accuracy.

---

### 3. Task 3: Handwritten Character Recognition
* **Objective:** Recognize and classify handwritten digits/alphabets from image data.
* **Dataset Used:** [MNIST / EMNIST Dataset](https://www.tensorflow.org/datasets/catalog/mnist)
* **Approach & Methods:** 
  * Preprocessed image arrays through normalization and reshaping.
  * Built a **Convolutional Neural Network (CNN)** with Convolutional, MaxPooling, Dropout, and Dense layers.
  * Evaluated feature extraction and spatial representation capabilities.
* **Key Evaluation:** Test Accuracy, Precision, Recall, and Loss Curves over training epochs.

---

### 4. Task 4: Disease Prediction from Medical Data
* **Objective:** Predict patient disease probability based on medical diagnostics and symptoms.
* **Dataset Used:** [Pima Indians Diabetes / UCI Heart Disease Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
* **Approach & Methods:** 
  * Preprocessed clinical data, checked feature correlations, and scaled continuous medical parameters.
  * Trained ensemble models including **Support Vector Machines (SVM)**, **XGBoost**, and **Random Forest**.
* **Key Evaluation:** Evaluated model sensitivity/recall to minimize false negatives in diagnostic risk assessment.

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.x
* **Data Processing & ML:** Pandas, NumPy, Scikit-learn
* **Deep Learning & Audio:** TensorFlow, Keras, Librosa
* **Visualization:** Matplotlib, Seaborn

---

## ✒️ Author
**Alisha**  
Machine Learning Intern @ CodeAlpha
