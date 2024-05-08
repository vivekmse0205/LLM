import re

import joblib
import nltk
import numpy as np
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_numeric, strip_short
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup

tfidf_vectorizer = TfidfVectorizer(max_features=1000)
app = Flask(__name__)

# Load the pre-trained Logistic Regression model
model = joblib.load("Logistic_Regression_model.pkl")

# Load the TF-IDF vectorizer used during training
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Define class names
class_names = ["legit email", "spam email"]


# Function to remove HTML tags from text
def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

# Function to remove hyperlinks from text
def remove_hyperlinks(text):
    return re.sub(r'http\S+', '', text)

# Additional preprocessing function for HTML tags and email IDs
def preprocess_text_with_html(text):
    text = remove_html_tags(text)
    text = remove_hyperlinks(text)
    # Remove HTML tags
    text_no_html = strip_tags(text)
    # Remove email addresses
    text_no_emails = re.sub(r'\S*@\S*\s?', '', text_no_html)
    return text_no_emails

# Preprocess text data
def preprocess_text(text):
    # Additional preprocessing for HTML and emails
    text = preprocess_text_with_html(text)
    # Tokenize and preprocess text
    tokens = preprocess_string(text)
    # Further preprocessing steps
    tokens = [strip_punctuation(token) for token in tokens]
    tokens = [strip_numeric(token) for token in tokens]
    tokens = [strip_short(token, minsize=3) for token in tokens]
    tokens = [token.lower() for token in tokens]
    tokens = ' '.join(tokens)
    return tokens


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    message = data['message']
    message = preprocess_text(message)
    # Vectorize the input text using the TF-IDF vectorizer
    text_tfidf = tfidf_vectorizer.transform([message])
    # Predict the class and probability score
    class_probabilities = model.predict_proba(text_tfidf)[0]
    prediction_index = np.argmax(class_probabilities)
    predicted_class = class_names[prediction_index]
    probability_score = class_probabilities[prediction_index]

    return jsonify({'prediction': predicted_class,
                    'score': probability_score})


if __name__ == '__main__':
    app.run(debug=True)