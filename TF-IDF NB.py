# sentiment_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
import warnings

class SentimentAnalysisModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.classifier = MultinomialNB()

    def map_sentiment(self, star_rating):
        if star_rating >= 3:
            return 'Positive'
        elif 2 <= star_rating < 3:
            return 'Neutral'
        else:
            return 'Negative'

    def preprocess_data(self, X, y):
        y_true = y.apply(lambda x: self.map_sentiment(float(x)))
        return train_test_split(X, y_true, test_size=0.2, random_state=42)

    def train_model(self, X_train, y_train):
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        self.classifier.fit(X_train_tfidf, y_train)

    def save_model(self, vectorizer_path='tfidf_vectorizer.pkl', classifier_path='sentiment_classifier.pkl'):
        joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump(self.classifier, classifier_path)

    def evaluate_model(self, X_test, y_test):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            
            X_test_tfidf = self.vectorizer.transform(X_test)
            y_pred = self.classifier.predict(X_test_tfidf)

            # Evaluate the performance
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {accuracy:.2f}")

            # Display classification report
            print("Classification Report:")
            print(classification_report(y_test, y_pred))

# Load the cleaned dataset
cleaned_data = pd.read_csv("Cleaned_data.csv")

# Assuming 'cleaned_text' is the column with the cleaned text data
X = cleaned_data['cleaned_text']
y = cleaned_data['stars']

# Create an instance of the SentimentAnalysisModel
sentiment_model = SentimentAnalysisModel()

# Preprocess data and train the model
X_train, X_test, y_train, y_test = sentiment_model.preprocess_data(X, y)
sentiment_model.train_model(X_train, y_train)

# Save the trained model
sentiment_model.save_model()

# Evaluate the model
sentiment_model.evaluate_model(X_test, y_test)
