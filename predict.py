import os
import sys
import joblib

# Allow importing preprocess.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import clean_text


# ----------------------------------------
# Load trained model and TF-IDF vectorizer
# ----------------------------------------

MODEL_PATH = "models/offensive_language_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


# ----------------------------------------
# Prediction function
# ----------------------------------------

def predict_text(text):

    # Clean input text
    cleaned_text = clean_text(text)

    # Convert text into TF-IDF features
    text_vector = vectorizer.transform([cleaned_text])

    # Make prediction
    prediction = model.predict(text_vector)[0]

    # Get prediction probability
    probability = model.predict_proba(text_vector)[0]

    # Convert numerical result to label
    if prediction == 1:
        label = "OFFENSIVE"
        confidence = probability[1] * 100
    else:
        label = "NOT OFFENSIVE"
        confidence = probability[0] * 100

    return label, confidence


# ----------------------------------------
# Interactive prediction
# ----------------------------------------

if __name__ == "__main__":

    print("======================================")
    print(" SOCIAL MEDIA OFFENSIVE LANGUAGE")
    print(" DETECTION SYSTEM")
    print("======================================")

    print("\nType 'exit' to close the program.\n")

    while True:

        text = input("Enter social media text: ")

        if text.lower() == "exit":
            print("\nProgram closed.")
            break

        if not text.strip():
            print("Please enter some text.\n")
            continue

        label, confidence = predict_text(text)

        print("\nPrediction:", label)
        print(f"Confidence: {confidence:.2f}%")
        print("-" * 40)