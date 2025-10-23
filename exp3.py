# exp3_knn.py

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_diabetes
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Classification (Iris)
iris = load_iris()
Xc, yc = iris.data, iris.target
Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=42)
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(Xc_train, yc_train)
y_pred = clf.predict(Xc_test)
print("KNN Classification accuracy (Iris):", accuracy_score(yc_test, y_pred))

# Regression (Diabetes dataset)
diab = load_diabetes()
Xr, yr = diab.data, diab.target
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)
reg = KNeighborsRegressor(n_neighbors=5)
reg.fit(Xr_train, yr_train)
y_reg_pred = reg.predict(Xr_test)
print("KNN Regression MSE (Diabetes):", mean_squared_error(yr_test, y_reg_pred))
