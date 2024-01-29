# word2vec_sentiment.py

import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

class Word2VecSentimentAnalysis:
    def __init__(self, bad_threshold=2, neutral_threshold=3, vector_size=100, window=5, min_count=1):
        self.bad_threshold = bad_threshold
        self.neutral_threshold = neutral_threshold
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None
        self.classifier = LogisticRegression()

    def preprocess_data(self, data):
        data['sentiment'] = data['stars'].apply(
            lambda x: 'good' if x >= self.neutral_threshold else ('neutral' if x >= self.bad_threshold else 'bad')
        )
        return train_test_split(data, test_size=0.2, random_state=42)

    def train_word2vec_model(self, sentences):
        self.model = Word2Vec(sentences=sentences, vector_size=self.vector_size, window=self.window,
                              min_count=self.min_count, workers=4)

    def average_word_vectors(self, words):
        feature_vector = np.zeros((self.vector_size,), dtype="float64")
        n_words = 0.

        for word in words:
            if word in self.model.wv.index_to_key:
                n_words += 1.
                feature_vector = np.add(feature_vector, self.model.wv[word])

        if n_words:
            feature_vector = np.divide(feature_vector, n_words)

        return feature_vector

    def get_avg_feature_vectors(self, reviews):
        review_feature_vectors = []

        for review in reviews:
            review_feature_vectors.append(self.average_word_vectors(review))

        return np.array(review_feature_vectors)

    def train_classifier(self, train_features, train_labels):
        self.classifier.fit(train_features, train_labels)

    def evaluate_classifier(self, test_features, test_labels):
        predictions = self.classifier.predict(test_features)

        accuracy = accuracy_score(test_labels, predictions)
        classification_report_output = classification_report(test_labels, predictions)

        print(f"Accuracy: {accuracy}")
        print("Classification Report:")
        print(classification_report_output)

        # Save the classification report for later use
        self.classification_report_output = classification_report_output

    def save_models(self, model_path='word2vec_model.joblib', report_path='classification_report.txt'):
        joblib.dump(self.model, model_path)
        with open(report_path, 'w') as report_file:
            report_file.write(self.classification_report_output)

    def run_pipeline(self, data):
        train_data, test_data = self.preprocess_data(data)

        train_sentences = [text.split() for text in train_data['cleaned_text']]
        test_sentences = [text.split() for text in test_data['cleaned_text']]

        self.train_word2vec_model(train_sentences)

        train_vectors = self.get_avg_feature_vectors(train_sentences)
        test_vectors = self.get_avg_feature_vectors(test_sentences)

        self.train_classifier(train_vectors, train_data['sentiment'])
        self.evaluate_classifier(test_vectors, test_data['sentiment'])
        self.save_models()

# Load the cleaned data
cleaned_data = pd.read_csv('Cleaned_data.csv')

# Create an instance of Word2VecSentimentAnalysis
word2vec_model = Word2VecSentimentAnalysis()

# Run the complete pipeline
word2vec_model.run_pipeline(cleaned_data)
