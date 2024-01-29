import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load cleaned data
cleaned_data = pd.read_csv('Cleaned_data.csv')

# Assuming 'cleaned_text' is the column with the cleaned text data
X = cleaned_data['cleaned_text']
y = cleaned_data['stars']

# Load models
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
tfidf_classifier = joblib.load('sentiment_classifier.pkl')

word2vec_model = joblib.load('word2vec_model.joblib')
bow_model = joblib.load('bow_model.joblib')
transformer_model = joblib.load('transformer_model.joblib')

# Transform data with vectorizer and make predictions with classifier
tfidf_features = tfidf_vectorizer.transform(X)
tfidf_predictions = tfidf_classifier.predict(tfidf_features)

word2vec_predictions = word2vec_model.predict(X)
bow_predictions = bow_model.predict(X)
transformer_predictions = transformer_model.predict(X)

# True labels
true_labels = ['Positive' if star_rating >= 3 else ('Neutral' if 2 <= star_rating < 3 else 'Negative') for star_rating in y]

# Function to calculate metrics
def calculate_metrics(true_labels, predicted_labels):
    return classification_report(true_labels, predicted_labels, output_dict=True)

# Calculate metrics for each model
metrics_tfidf = calculate_metrics(true_labels, tfidf_predictions)
metrics_word2vec = calculate_metrics(true_labels, word2vec_predictions)
metrics_bow = calculate_metrics(true_labels, bow_predictions)
metrics_transformer = calculate_metrics(true_labels, transformer_predictions)

# Create a summary DataFrame
summary_df = pd.DataFrame({
    'TF-IDF': metrics_tfidf['weighted avg'],
    'Word2Vec': metrics_word2vec['weighted avg'],
    'Bag of Words': metrics_bow['weighted avg'],
    'Sentence Transformers': metrics_transformer['weighted avg'],
}).transpose()

# Visualization with Seaborn
metrics_to_compare = ['precision', 'recall', 'f1-score', 'support']

for metric in metrics_to_compare:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=summary_df.index, y=summary_df[metric])
    plt.title(f'{metric} Comparison')
    plt.show()
