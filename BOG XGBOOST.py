# bow_sentiment.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

class BowSentimentAnalysis:
    def __init__(self, bad_threshold=2, neutral_threshold=3):
        self.bad_threshold = bad_threshold
        self.neutral_threshold = neutral_threshold
        self.vectorizer = CountVectorizer()
        self.classifier = XGBClassifier()
        self.label_encoder = LabelEncoder()  # Move LabelEncoder to the class level

    def preprocess_data(self, data):
        data['sentiment'] = data['stars'].apply(
            lambda x: 'good' if x >= self.neutral_threshold else ('neutral' if x >= self.bad_threshold else 'bad')
        )
        return train_test_split(data, test_size=0.2, random_state=42)

    def train_model(self, train_features, train_labels):
        train_labels_encoded = self.label_encoder.fit_transform(train_labels)  # Fit and transform on training labels
        self.classifier.fit(train_features, train_labels_encoded)

    def evaluate_model(self, test_features, test_labels):
        test_labels_encoded = self.label_encoder.transform(test_labels)  # Transform test labels
        predictions = self.classifier.predict(test_features)

        accuracy = accuracy_score(test_labels_encoded, predictions)
        classification_report_output = classification_report(test_labels_encoded, predictions)

        print(f"Accuracy: {accuracy}")
        print("Classification Report:")
        print(classification_report_output)

    def save_model(self, model_path='bow_model.joblib'):
        joblib.dump(self.classifier, model_path)

    def run_pipeline(self, data):
        train_data, test_data = self.preprocess_data(data)

        # Fit and transform on the training data
        train_features = self.vectorizer.fit_transform(train_data['cleaned_text'])
        
        # Transform the test data
        test_features = self.vectorizer.transform(test_data['cleaned_text'])

        # Train the model
        self.train_model(train_features, train_data['sentiment'])

        # Evaluate and save the model
        self.evaluate_model(test_features, test_data['sentiment'])
        self.save_model()

# Load the cleaned data
cleaned_data = pd.read_csv('Cleaned_data.csv')

# Assuming 'cleaned_text' is the column with the cleaned text data
X = cleaned_data['cleaned_text']
y = cleaned_data['stars']

# Create an instance of BowSentimentAnalysis
bow_model = BowSentimentAnalysis()

# Run the complete pipeline
bow_model.run_pipeline(cleaned_data)
