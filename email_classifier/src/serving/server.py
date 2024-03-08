import re

import joblib
import nltk
import numpy as np
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_features=1000)
app = Flask(__name__)

# Load the pre-trained Logistic Regression model
model = joblib.load("Logistic_Regression_model.pkl")

# Load the TF-IDF vectorizer used during training
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Define class names
class_names = ["legit email", "spam email"]


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove punctuation and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join tokens back into text
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    message = data['message']
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