# Library imports
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib
import string
from spacy.lang.en.stop_words import STOP_WORDS
from flask import Flask, request, jsonify, render_template
import unicodedata
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import nltk
import re
# from tensorflow.keras.models import load_model

# Load the model from the .h5 file in the same directory
# model2 = load_model('lstm_model.h5')


# Load trained Pipeline
model = joblib.load('nb_model.pkl')
model_fnc=joblib.load('tfidf_model3.pkl')
# model_fnc2=joblib.load('predict_sentiment_function.pkl')
# Create the app object
app = Flask(__name__)
# Download NLTK stopwords corpus if not already downloaded
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# # creating a function for data cleaning
# from custom_tokenizer_function import CustomTokenizer
contractions_dict = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it had",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there had",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'alls": "you alls",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}

def preprocess_text(text):
    # Define stopwords with exceptions
    stop_words = set(stopwords.words('english'))
    exceptions = {'couldn', "couldn't", 'haven', "haven't"}  # Add more exceptions as needed
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    # Remove HTML tags

    text = re.sub(r'<[^>]+>', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    #contractions
    for contraction, expansion in contractions_dict.items():
        text = text.replace(contraction, expansion)
    # Remove alphanumeric characters and stop words (with exceptions)
    words = text.split()  # Split text into words
    filtered_words = [word for word in words if word.isalpha() and word.lower() not in stop_words.union(exceptions)]    # Lemmatize tokens
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    # Join tokens back into a string
    preprocessed_text = ' '.join(lemmatized_words)
    return preprocessed_text

# Define predict function
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    new_review = ' '.join(str(x) for x in request.form.values())
    preprocess_text1 = preprocess_text(new_review)
    vectorized_text = model_fnc['count_vectorizer'].transform([preprocess_text1])
    predictions = model.predict(vectorized_text)
    if predictions == 0:
        return render_template('index.html', prediction_text='Positive')
    elif predictions== 1:   
        return render_template('index.html', prediction_text='Negative')
    else:
        #Render the template without passing prediction_text
        return render_template('index.html', prediction_text='')
    print(predic)

if __name__ == "__main__":
    app.run(debug=True)
