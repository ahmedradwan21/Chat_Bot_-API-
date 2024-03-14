# myapp/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from .preprocessing import clean_text, lemmatize_text
import joblib

class PredictView(APIView):
    def post(self, request):
        text = request.data.get('text')

        # Clean and lemmatize the text
        cleaned_text = clean_text(text)
        lemmatized_text = lemmatize_text(cleaned_text)

        # Load the TF-IDF vectorizer and Decision Tree Classifier
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
        dt_classifier = joblib.load('best_svm_classifier.joblib')

        # Vectorize the text
        text_vectorized = tfidf_vectorizer.transform([lemmatized_text]).toarray()

        # Predict using the classifier
        predicted_label = dt_classifier.predict(text_vectorized)[0]

        return Response({'predicted': predicted_label})
