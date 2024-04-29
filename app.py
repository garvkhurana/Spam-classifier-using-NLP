import pickle
from flask import Flask, request, render_template
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

app = Flask(__name__, template_folder='templates')

model = pickle.load(open('fake_detector.pkl', 'rb'))
vectorizer = pickle.load(open('count_vectorizer.pkl', 'rb'))

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
nltk.download('wordnet')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    v2 = request.form["news"]
    corpus = []
    sentences = sent_tokenize(v2)

    for sentence in sentences:
        review = word_tokenize(sentence)
        review = re.sub('[^a-zA-Z]', " ", sentence)
        review = review.lower()
        review = review.split()
        review = [lemmatizer.lemmatize(word) for word in review if not word in stop_words]
        review = ' '.join(review)
        corpus.append(review)

    X = vectorizer.transform(corpus).toarray()
    prediction = model.predict(X)
    result = "Real News" if prediction == 1 else "Fake News"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
