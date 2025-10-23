# exp7_naive_bayes_text_classification.py
# -------------------------------------------------------------
# Aim: To implement a simple Naive Bayes Text Classifier from scratch
#      to classify text data into different categories.
# -------------------------------------------------------------

import math
from collections import defaultdict

# -------------------- TRAINING DATA --------------------
# Each tuple has (text, category)
train_data = [
    ("The rocket launched successfully", "sci.space"),
    ("NASA discovered a new planet", "sci.space"),
    ("Stars and galaxies are fascinating", "sci.space"),
    ("Rendering 3D graphics is fun", "comp.graphics"),
    ("I love creating digital art", "comp.graphics"),
    ("Computer graphics are amazing", "comp.graphics"),
    ("Atheism is the lack of belief in gods", "alt.atheism"),
    ("Many people follow religion strictly", "alt.atheism"),
    ("There is no evidence for gods", "alt.atheism"),
]

# -------------------- TOKENIZATION FUNCTION --------------------
# Converts text into a list of lowercase words
def tokenize(text):
    return text.lower().split()

# -------------------- NAIVE BAYES CLASSIFIER --------------------
class NaiveBayes:
    def __init__(self):
        self.classes = set()                              # Set of all class labels
        self.word_counts = defaultdict(lambda: defaultdict(int))  # Word frequency per class
        self.class_counts = defaultdict(int)               # Number of samples per class
        self.total_words = defaultdict(int)                # Total words per class

    # Train the classifier on given data
    def fit(self, data):
        for text, label in data:
            self.classes.add(label)
            self.class_counts[label] += 1
            for word in tokenize(text):
                self.word_counts[label][word] += 1
                self.total_words[label] += 1

    # Predict the most likely class for a given text
    def predict(self, text):
        words = tokenize(text)
        best_label, max_prob = None, -float("inf")

        # Compute log-probabilities for each class
        for label in self.classes:
            # Start with log prior probability
            log_prob = math.log(self.class_counts[label] / sum(self.class_counts.values()))
            # Add log likelihood for each word
            for word in words:
                word_freq = self.word_counts[label][word]
                log_prob += math.log((word_freq + 1) / (self.total_words[label] + len(self.word_counts[label])))
            # Track the best label (highest log probability)
            if log_prob > max_prob:
                max_prob = log_prob
                best_label = label

        return best_label

# -------------------- TRAINING THE MODEL --------------------
nb = NaiveBayes()
nb.fit(train_data)

# -------------------- TESTING THE MODEL --------------------
test_texts = [
    "Space shuttle launched",
    "Drawing in Photoshop is fun",
    "I don't believe in any god",
]

print("---- Naive Bayes Predictions ----")
for text in test_texts:
    print(f"{text} → {nb.predict(text)}")

# -------------------- REAL-LIFE ANALOGY --------------------
# Naive Bayes analogy:
#   Imagine classifying an email as 'spam' or 'not spam'.
#   If the email contains words like "offer", "discount", "buy now",
#   Naive Bayes checks how often these words appear in spam emails vs non-spam emails.
#   The class with the highest overall probability is chosen.
#
# Core idea: "The more often a word appears in a class, 
#             the more likely a new message with that word belongs to that class."
#
# -------------------------------------------------------------
# Example Output:
# ---- Naive Bayes Predictions ----
# Space shuttle launched → sci.space
# Drawing in Photoshop is fun → comp.graphics
# I don't believe in any god → alt.atheism
# -------------------------------------------------------------
