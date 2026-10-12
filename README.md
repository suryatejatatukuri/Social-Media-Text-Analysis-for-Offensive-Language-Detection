# Social Media Text Analysis for Offensive Language Detection

## Project Overview
This project uses Natural Language Processing (NLP) and Machine Learning to classify social media text as offensive or non-offensive. It uses the Offensive Language Identification Dataset (OLID), TF-IDF feature extraction, and Logistic Regression.

## Features
- Text preprocessing and cleaning
- Offensive and non-offensive text classification
- TF-IDF feature extraction
- Machine learning model evaluation
- Interactive Streamlit web application
- Prediction confidence display

## Technologies Used
- Python
- Natural Language Processing (NLP)
- Pandas and NumPy
- NLTK
- Scikit-learn
- TF-IDF
- Logistic Regression
- Streamlit
- Joblib

## Project Structure
```text
Social-Media-Text-Analysis/
├── app.py
├── requirements.txt
├── README.md
├── dataset/
│   └── README.md
├── models/
│   ├── offensive_language_model.pkl
│   └── tfidf_vectorizer.pkl
└── src/
    ├── preprocess.py
    ├── train.py
    └── predict.py
```

## Installation and Usage

### 1. Clone the repository
```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Social-Media-Text-Analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download the required OLID dataset files and place them in the `dataset/` directory before training.

### 4. Run the application
```bash
streamlit run app.py
```

### 5. Train the model again (optional)
```bash
python src/train.py
```

## Model Details
- **Feature Extraction:** TF-IDF
- **Classification Algorithm:** Logistic Regression
- **Task:** Binary text classification
- **Labels:** OFF (Offensive), NOT (Not Offensive)

## Limitations
This model may misclassify sarcasm, slang, context-dependent language, and unfamiliar expressions. Its predictions should not be treated as definitive judgments about a person or message.

## Disclaimer
This project is intended for educational and research purposes.

## Author
Information Technology Graduate | Artificial Intelligence and Machine Learning