# Student Performance Prediction Using Machine Learning

## 📌 Project Overview

This project predicts whether a student will **Pass or Fail** based on academic and lifestyle-related features.

The project compares multiple machine learning classification algorithms to understand their performance on the dataset.

## 📊 Features

The model uses the following input features:

- Study Hours
- Attendance
- Sleep Hours
- Previous Score

**Target:** Result

- `0` = Fail
- `1` = Pass

## 🤖 Machine Learning Models

The following classification algorithms are used:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

## 🔄 Project Workflow

1. Create and explore the dataset
2. Check missing values and duplicates
3. Prepare features (`X`) and target (`y`)
4. Split data into training and testing sets
5. Apply feature scaling
6. Train multiple classification models
7. Make predictions
8. Compare model accuracy
9. Predict the result of a new student

## 📈 Results

On the current small sample dataset and train/test split, the models achieved:

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | 100% |
| Decision Tree | 100% |
| Random Forest | 100% |
| KNN | 100% |
| SVM | 100% |

> **Note:** This project uses a small manually created dataset for learning purposes. The reported accuracy should not be interpreted as expected performance on real-world student data.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- VS Code

## ▶️ How to Run

Install the required libraries:

```bash
python -m pip install pandas scikit-learn
```

Run the project:

```bash
python student_performance.py
```

## 🎯 Example Prediction

Example student:

- Study Hours: 7
- Attendance: 80%
- Sleep Hours: 7
- Previous Score: 70

Output:

```text
Prediction: Student will PASS
```

## 📚 What I Learned

Through this project, I practiced:

- Data exploration and cleaning
- Feature selection
- Train/test splitting
- Feature scaling
- Classification algorithms
- Model training and prediction
- Accuracy evaluation
- Comparing multiple ML models

## ⚠️ Limitations

The current dataset contains only a small number of manually created records. Future improvements could include using a larger real-world dataset, additional student features, cross-validation, and more detailed evaluation metrics.
