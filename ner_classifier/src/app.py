from flask import Flask, request, jsonify
import spacy

app = Flask(__name__)

# Load the spaCy model
nlp_ner = spacy.load("models/model-best")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    doc = nlp_ner(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return jsonify({'entities': entities})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
