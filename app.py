import os
import sys

import joblib
import streamlit as st


# --------------------------------------------------
# Import preprocessing function
# --------------------------------------------------

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from preprocess import clean_text


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

MODEL_PATH = "models/offensive_language_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Offensive Language Detection",
    page_icon="🛡️",
    layout="centered"
)


# --------------------------------------------------
# Application title
# --------------------------------------------------

st.title("🛡️ Social Media Text Analysis")

st.subheader("Offensive Language Detection")

st.write(
    "Enter social media text below to analyze whether "
    "the content is classified as offensive or not offensive."
)


# --------------------------------------------------
# Text input
# --------------------------------------------------

text = st.text_area(
    "Enter social media text:",
    placeholder="Type your text here...",
    height=150
)


# --------------------------------------------------
# Analyze button
# --------------------------------------------------

if st.button("🔍 Analyze Text"):

    if not text.strip():

        st.warning("Please enter some text to analyze.")

    else:

        # Clean text
        cleaned_text = clean_text(text)

        # Transform text
        text_vector = vectorizer.transform([cleaned_text])

        # Predict
        prediction = model.predict(text_vector)[0]

        # Probability
        probabilities = model.predict_proba(text_vector)[0]

        if prediction == 1:

            label = "OFFENSIVE"
            confidence = probabilities[1] * 100

            st.error(
                f"🔴 Prediction: {label}"
            )

        else:

            label = "NOT OFFENSIVE"
            confidence = probabilities[0] * 100

            st.success(
                f"🟢 Prediction: {label}"
            )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )


# --------------------------------------------------
# Information section
# --------------------------------------------------

st.divider()

st.subheader("About the Project")

st.write(
    "This application uses Natural Language Processing (NLP) "
    "and Machine Learning to classify social media text."
)

st.write(
    "**Techniques used:** TF-IDF Vectorization and "
    "Logistic Regression."
)

st.write(
    "**Dataset:** Offensive Language Identification Dataset (OLID)."
)