# myapp/preprocessing.py
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(sent):
    stop_words = set(stopwords.words('english'))
    sent = sent.translate(str.maketrans('', '', string.punctuation)).strip()
    words = word_tokenize(sent.lower())
    words = [word for word in words if word.isalpha() and word not in stop_words]
    sent = ' '.join(words)
    return sent

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    word_list = word_tokenize(text)
    text = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    return text
