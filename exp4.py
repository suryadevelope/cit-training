# exp4_dt_classification.py
# -------------------------------------------------------------
# Aim: To implement Decision Tree Classification on the Wine dataset
#      and improve accuracy using hyperparameter tuning with GridSearchCV.
# -------------------------------------------------------------

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score

# -------------------- LOAD DATASET --------------------
# Load the built-in Wine dataset (contains chemical analysis of wines)
wine = load_wine()
X, y = wine.data, wine.target   # X = input features, y = class labels

# Split dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# -------------------- BASIC DECISION TREE --------------------
# Create a basic Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=1)

# Train (fit) the model on training data
dt.fit(X_train, y_train)

# Predict on test data
y_pred_basic = dt.predict(X_test)

# Print base accuracy
print("Base Accuracy:", accuracy_score(y_test, y_pred_basic))

# -------------------- PARAMETER TUNING USING GRIDSEARCHCV --------------------
# Define parameter grid for tuning (search over these parameter combinations)
params = {
    'max_depth': [None, 3, 5, 7],          # maximum depth of the tree
    'min_samples_split': [2, 4, 6],        # minimum samples required to split a node
    'criterion': ['gini', 'entropy']       # impurity measures
}

# Create GridSearchCV object for cross-validated parameter tuning
grid = GridSearchCV(
    DecisionTreeClassifier(random_state=1),
    param_grid=params,
    cv=5,                   # 5-fold cross-validation
    scoring='accuracy'      # measure accuracy for model evaluation
)

# Fit the grid search model on training data
grid.fit(X_train, y_train)

# Extract the best model and its parameters
best = grid.best_estimator_
print("Best params:", grid.best_params_)

# -------------------- EVALUATION OF TUNED MODEL --------------------
# Predict on test data using the best model
y_pred_tuned = best.predict(X_test)

# Print improved accuracy
print("Tuned Accuracy:", accuracy_score(y_test, y_pred_tuned))

# Print detailed classification report (precision, recall, f1-score)
print(classification_report(y_test, y_pred_tuned))

# -------------------- REAL-LIFE ANALOGY --------------------
# Decision Tree analogy:
#   Imagine deciding which wine to buy based on its properties:
#   - If alcohol content > 13, go left branch.
#   - Else, check color intensity and acidity.
#   This process continues like a flowchart until a final decision is reached.
#
# GridSearchCV analogy:
#   It’s like trying different settings (depth, splitting rules)
#   to find the combination that gives the best prediction accuracy.
#
# -------------------------------------------------------------
# Example Output:
# Base Accuracy: 0.944
# Best params: {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 2}
# Tuned Accuracy: 0.972
# (Followed by classification report)
# -------------------------------------------------------------
