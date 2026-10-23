import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score


# -----------------------------------
# 1. CREATE DATASET
# -----------------------------------

data = {
    "Study_Hours": [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 2, 3, 4, 5, 6, 7, 8, 9, 4, 7 ],

    "Attendance": [ 45, 50, 55, 60, 65, 70, 75, 80, 85, 95, 48, 58, 62, 68, 72, 78, 82, 90, 64, 76 ],

    "Sleep_Hours": [ 5, 6, 6, 7, 6, 7, 7, 8, 7, 8, 5, 6, 7, 6, 7, 8, 7, 8, 6, 7 ],

    "Previous_Score": [ 30, 35, 40, 45, 50, 60, 65, 72, 80, 90, 32, 42, 48, 55, 62, 68, 75, 85, 52, 66 ],
    "Result": [ 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1 ]
}

df = pd.DataFrame(data)


# -----------------------------------
# 2. DATA EXPLORATION
# -----------------------------------

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# -----------------------------------
# 3. DATA CLEANING
# -----------------------------------

df = df.drop_duplicates()


# -----------------------------------
# 4. CREATE X AND y
# -----------------------------------

X = df[[ "Study_Hours", "Attendance", "Sleep_Hours", "Previous_Score" ]]
y = df["Result"]


# -----------------------------------
# 5. TRAIN TEST SPLIT
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split( X,y,test_size=0.2,random_state=42,stratify=y)


# -----------------------------------
# 6. FEATURE SCALING
# -----------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -----------------------------------
# 7. CREATE MODELS
# -----------------------------------

logistic = LogisticRegression()

decision_tree = DecisionTreeClassifier(random_state=42)

random_forest = RandomForestClassifier(n_estimators=100,random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)

svm = SVC()


# -----------------------------------
# 8. TRAIN MODELS
# -----------------------------------

logistic.fit(X_train_scaled, y_train)

decision_tree.fit(X_train, y_train)

random_forest.fit(X_train, y_train)

knn.fit(X_train_scaled, y_train)

svm.fit(X_train_scaled, y_train)


# -----------------------------------
# 9. MAKE PREDICTIONS
# -----------------------------------

logistic_pred = logistic.predict(X_test_scaled)

tree_pred = decision_tree.predict(X_test)

forest_pred = random_forest.predict(X_test)

knn_pred = knn.predict(X_test_scaled)

svm_pred = svm.predict(X_test_scaled)


# -----------------------------------
# 10. CALCULATE ACCURACY
# -----------------------------------

logistic_accuracy = accuracy_score(y_test,logistic_pred)

tree_accuracy = accuracy_score( y_test,tree_pred)

forest_accuracy = accuracy_score(y_test,forest_pred)

knn_accuracy = accuracy_score(y_test,knn_pred)

svm_accuracy = accuracy_score(y_test,svm_pred)


# -----------------------------------
# 11. DISPLAY RESULTS
# -----------------------------------

print("\n--- MODEL ACCURACY ---")

print("Logistic Regression:",round(logistic_accuracy * 100, 2),"%")

print("Decision Tree:",round(tree_accuracy * 100, 2),"%")

print("Random Forest:",round(forest_accuracy * 100, 2),"%")

print("KNN:",round(knn_accuracy * 100, 2),"%")

print("SVM:",round(svm_accuracy * 100, 2),"%")


# -----------------------------------
# 12. NEW STUDENT PREDICTION
# -----------------------------------

# Order:
# Study_Hours, Attendance,
# Sleep_Hours, Previous_Score

new_student = pd.DataFrame([[7, 80, 7, 70]],columns=["Study_Hours","Attendance","Sleep_Hours","Previous_Score"])

prediction = random_forest.predict(new_student)

print("\n--- NEW STUDENT PREDICTION ---")

if prediction[0] == 1:
    print("Prediction: Student will PASS")
else:
    print("Prediction: Student will FAIL")

