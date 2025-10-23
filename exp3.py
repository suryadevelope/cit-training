# exp3_knn.py
# exp3_knn.py
# -------------------------------------------------------------
# Aim: To demonstrate K-Nearest Neighbors (KNN) algorithm for
#      both Classification (Iris dataset) and Regression (Diabetes dataset)
# -------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_diabetes
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# -------------------- KNN CLASSIFICATION (Iris Dataset) --------------------
# Load the built-in Iris dataset
iris = load_iris()
Xc, yc = iris.data, iris.target   # Xc = features, yc = class labels

# Split dataset into training and testing parts (80%-20%)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=42)

# Create a KNN classifier model with k=5 neighbors
clf = KNeighborsClassifier(n_neighbors=5)

# Train (fit) the classifier on the training data
clf.fit(Xc_train, yc_train)

# Predict the class labels for test data
y_pred = clf.predict(Xc_test)

# Calculate and print accuracy
print("KNN Classification accuracy (Iris):", accuracy_score(yc_test, y_pred))

# -------------------- KNN REGRESSION (Diabetes Dataset) --------------------
# Load the built-in Diabetes dataset
diab = load_diabetes()
Xr, yr = diab.data, diab.target   # Xr = features, yr = continuous target values

# Split dataset into training and testing parts (80%-20%)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)

# Create a KNN regressor model with k=5 neighbors
reg = KNeighborsRegressor(n_neighbors=5)

# Train (fit) the regressor on the training data
reg.fit(Xr_train, yr_train)

# Predict target values for test data
y_reg_pred = reg.predict(Xr_test)

# Calculate and print Mean Squared Error (MSE)
print("KNN Regression MSE (Diabetes):", mean_squared_error(yr_test, y_reg_pred))

# -------------------- REAL-LIFE ANALOGY --------------------
# Classification analogy:
#   To identify a flower, look at the 5 most similar known flowers.
#   If most are "Iris-setosa", you classify it the same.
#
# Regression analogy:
#   To estimate a person’s blood sugar level, find 5 people
#   with similar age, weight, and health data, and take their average.
#
# -------------------------------------------------------------
# Example Output:
# KNN Classification accuracy (Iris): 1.0
# KNN Regression MSE (Diabetes): ~2980
# -------------------------------------------------------------
