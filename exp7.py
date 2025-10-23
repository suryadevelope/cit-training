# exp8_svm_classifier_simulation.py
# -------------------------------------------------------------
# Aim: To simulate a simple Support Vector Machine (SVM)-like 
#      classifier using a nearest centroid approach on a small 
#      subset of the Iris dataset.
# -------------------------------------------------------------

from statistics import mean

# -------------------- SAMPLE IRIS DATA --------------------
# Each tuple = ([features], class_label)
# Features: [sepal length, sepal width, petal length, petal width]
# Labels: 0 = Setosa, 1 = Versicolor, 2 = Virginica
iris_data = [
    ([5.1, 3.5, 1.4, 0.2], 0),
    ([4.9, 3.0, 1.4, 0.2], 0),
    ([6.2, 3.4, 5.4, 2.3], 2),
    ([5.9, 3.0, 4.2, 1.5], 1),
    ([6.0, 2.2, 4.0, 1.0], 1),
    ([5.5, 2.3, 4.0, 1.3], 1)
]

# Split data into training and testing
train = iris_data[:4]
test = iris_data[4:]

# -------------------- NEAREST CENTROID FUNCTION --------------------
# This function calculates the centroid (average point) for each class,
# then predicts the class of a new sample based on the nearest centroid.
def nearest_centroid(train, sample):
    centroids = {}

    # Step 1: Calculate centroids for each class
    for x, y in train:
        if y not in centroids:
            centroids[y] = [[] for _ in range(len(x))]
        for i in range(len(x)):
            centroids[y][i].append(x[i])

    # Step 2: Take the mean of all feature values per class
    for y in centroids:
        centroids[y] = [mean(col) for col in centroids[y]]

    # Step 3: Find the nearest centroid to the test sample
    min_dist = float('inf')
    label = None
    for y, center in centroids.items():
        dist = sum((a - b) ** 2 for a, b in zip(sample, center)) ** 0.5
        if dist < min_dist:
            min_dist = dist
            label = y
    return label

# -------------------- EVALUATION --------------------
# Predict and check accuracy
correct = 0
for x, y in test:
    pred = nearest_centroid(train, x)
    if pred == y:
        correct += 1

# Print the simulated SVM-like accuracy
print("SVM-like Accuracy:", round(correct / len(test), 3))

# -------------------- REAL-LIFE ANALOGY --------------------
# Real SVM (Support Vector Machine):
#   - SVM tries to find a "decision boundary" (a hyperplane)
#     that best separates different classes with maximum margin.
#
# This simple "nearest centroid" approach mimics that idea by:
#   - Calculating the "center" (mean point) of each class,
#   - Predicting the class whose center is closest to the new sample.
#
# -------------------------------------------------------------
# Example Output:
# SVM-like Accuracy: 0.5
# -------------------------------------------------------------
