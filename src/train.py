import os
import sys

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Allow importing preprocess.py from the src folder
# --------------------------------------------------

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import clean_text


# --------------------------------------------------
# File paths
# --------------------------------------------------

DATASET_PATH = "dataset/olid-training-v1.0.tsv"
MODEL_PATH = "models/offensive_language_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

print("\nLoading OLID dataset...")

df = pd.read_csv(DATASET_PATH, sep="\t")

print("Dataset loaded successfully.")
print("Total records:", len(df))


# --------------------------------------------------
# 2. Select required columns
# --------------------------------------------------

df = df[["tweet", "subtask_a"]]

# Remove missing values
df = df.dropna()

print("\nClass distribution:")
print(df["subtask_a"].value_counts())


# --------------------------------------------------
# 3. Clean text
# --------------------------------------------------

print("\nCleaning tweets...")
print("This may take a little time...")

df["cleaned_text"] = df["tweet"].apply(clean_text)


# --------------------------------------------------
# 4. Convert labels
#
# OFF = 1
# NOT = 0
# --------------------------------------------------

df["label"] = df["subtask_a"].map({
    "OFF": 1,
    "NOT": 0
})

df = df.dropna(subset=["label"])

df["label"] = df["label"].astype(int)


# --------------------------------------------------
# 5. Define X and y
# --------------------------------------------------

X = df["cleaned_text"]
y = df["label"]


# --------------------------------------------------
# 6. Split dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nDataset split:")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 7. TF-IDF Vectorization
# --------------------------------------------------

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF transformation completed.")
print("Number of features:", len(vectorizer.get_feature_names_out()))


# --------------------------------------------------
# 8. Train Logistic Regression
# --------------------------------------------------

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_tfidf, y_train)

print("Model training completed.")


# --------------------------------------------------
# 9. Make predictions
# --------------------------------------------------

print("\nGenerating predictions...")

y_pred = model.predict(X_test_tfidf)


# --------------------------------------------------
# 10. Evaluate model
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n========================================")
print("MODEL EVALUATION")
print("========================================")

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy Percentage: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["NOT Offensive", "Offensive"]
    )
)


# --------------------------------------------------
# 11. Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# --------------------------------------------------
# 12. Save model and vectorizer
# --------------------------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)

print("\n========================================")
print("FILES SAVED")
print("========================================")

print("Model:", MODEL_PATH)
print("Vectorizer:", VECTORIZER_PATH)

print("\nTraining pipeline completed successfully!")
