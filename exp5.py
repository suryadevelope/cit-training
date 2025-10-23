# exp6_random_forest_simulation.py
# -------------------------------------------------------------
# Aim: To demonstrate the basic working idea of Random Forest
#      through a simple simulation for both classification
#      and regression tasks using random predictions.
# -------------------------------------------------------------

import random

# -------------------- SAMPLE TRAINING DATA --------------------
# X_train: input features (example values similar to Iris dataset)
# y_train_class: categorical class labels (for classification)
# y_train_reg: continuous values (for regression)
X_train = [
    [5.1, 3.5, 1.4, 0.2],
    [6.2, 3.4, 5.4, 2.3],
    [5.9, 3.0, 4.2, 1.5]
]
y_train_class = [0, 2, 1]   # example classes
y_train_reg = [10.5, 25.7, 18.3]  # example regression targets

# -------------------- RANDOM FOREST "SIMULATION" --------------------
# These functions simulate how a Random Forest might behave.
# Instead of building multiple decision trees, it randomly selects
# predictions from the training data to mimic the ensemble output.

# Classification simulation
def random_forest_classify(X_test):
    # Randomly select one of the training class labels for each test sample
    return [random.choice(y_train_class) for _ in X_test]

# Regression simulation
def random_forest_regress(X_test):
    # Randomly select one of the training regression targets for each test sample
    return [random.choice(y_train_reg) for _ in X_test]

# -------------------- TEST DATA --------------------
# Two new samples to test classification and regression
X_test = [
    [5.5, 3.2, 1.5, 0.3],
    [6.5, 2.8, 4.6, 1.5]
]

# -------------------- PREDICTIONS --------------------
# Get simulated classification and regression outputs
y_pred_class = random_forest_classify(X_test)
y_pred_reg = random_forest_regress(X_test)

# Print predictions
print("Simulated Random Forest Classification Predictions:", y_pred_class)
print("Simulated Random Forest Regression Predictions:", y_pred_reg)

# -------------------- REAL-LIFE ANALOGY --------------------
# Real Random Forest working:
#   - A Random Forest creates many Decision Trees.
#   - For classification: each tree votes for a class, and the majority wins.
#   - For regression: each tree predicts a number, and their average is taken.
#
# This simulation randomly picks outcomes to show the *concept* of combining
# multiple models to make a decision.
#
# -------------------------------------------------------------
# Example Output:
# Simulated Random Forest Classification Predictions: [1, 0]
# Simulated Random Forest Regression Predictions: [25.7, 18.3]
# -------------------------------------------------------------
