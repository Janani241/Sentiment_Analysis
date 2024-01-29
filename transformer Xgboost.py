import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sentence_transformers import SentenceTransformer
import joblib

class TransformerSentimentAnalysis:
    def __init__(self, bad_threshold=2, neutral_threshold=3, model_name='distilbert-base-nli-mean-tokens'):
        self.bad_threshold = bad_threshold
        self.neutral_threshold = neutral_threshold
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.classifier = XGBClassifier()  

    def preprocess_data(self, data):
        data['sentiment'] = data['stars'].apply(
            lambda x: 'good' if x >= self.neutral_threshold else ('neutral' if x >= self.bad_threshold else 'bad')
        )
        return train_test_split(data, test_size=0.2, random_state=42)

    def encode_text(self, text_list):
        embeddings = self.model.encode(text_list)
        return pd.DataFrame(embeddings, columns=[str(i) for i in range(embeddings.shape[1])])

    def train_classifier(self, train_features, train_labels):
        self.classifier.fit(train_features, train_labels)

    def evaluate_classifier(self, test_features, test_labels):
        predictions = self.classifier.predict(test_features)

        accuracy = accuracy_score(test_labels, predictions)
        classification_report_output = classification_report(test_labels, predictions)

        print(f"Accuracy: {accuracy}")
        print("Classification Report:")
        print(classification_report_output)

        
        joblib.dump(self.classifier, 'transformer_xgboost_model.joblib')

    def run_pipeline(self, data):
        train_data, test_data = self.preprocess_data(data)

        train_features = self.encode_text(train_data['cleaned_text'].reset_index(drop=True))
        test_features = self.encode_text(test_data['cleaned_text'].reset_index(drop=True))

        self.train_classifier(train_features, train_data['sentiment'])
        self.evaluate_classifier(test_features, test_data['sentiment'])


cleaned_data = pd.read_csv('Cleaned_data.csv')


transformer_model = TransformerSentimentAnalysis()


transformer_model.run_pipeline(cleaned_data)
