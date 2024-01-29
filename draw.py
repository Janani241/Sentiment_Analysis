#draw pictures of experiments results

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

# Load the cleaned data
cleaned_data = pd.read_csv('Cleaned_data.csv')

# Assuming 'cleaned_text' is the column with the cleaned text data
X = cleaned_data['cleaned_text']
y = cleaned_data['stars']
neutral_threshold = 3
bad_threshold = 2
cleaned_data['sentiment'] = cleaned_data['stars'].apply(
                                                        lambda x: 'good' if x >= neutral_threshold else ('neutral' if x >= bad_threshold else 'bad')
                                                        )
train_data, test_data = train_test_split(cleaned_data, test_size=0.2, random_state=42)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load models
# models = {
#     'SentimentAnalysisModel': joblib.load('sentiment_classifier.pkl'),
#     'BowSentimentAnalysis': joblib.load('bow_model.joblib'),
#     #'Word2VecSentimentAnalysis': joblib.load('word2vec_model.joblib'),
#     'TransformerSentimentAnalysis': joblib.load('transformer_model.joblib')
# }


def plot_confusion_matrix(conf_matrix, classes, model_name):
    plt.figure(figsize=(6, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.colorbar()
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def TFIDF1():
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    classifier = joblib.load('sentiment_classifier.joblib')
    x_test = test_data['cleaned_text']
    y_test = test_data['sentiment']
    x_test = vectorizer.transform(x_test)
    y_pred = classifier.predict(x_test)
    
    sentiment_mapping1 = {'good': 1, 'neutral': 0, 'bad': -1}
    sentiment_mapping2 = {1: 1, 2: 0, 0: -1}
    y_test_numeric = list(map(sentiment_mapping1.get, y_test))
    y_pred_numeric = pd.Series([sentiment_mapping2[label] for label in y_pred])
    accuracy = accuracy_score(y_test_numeric, y_pred_numeric)
    class_report = classification_report(y_test_numeric, y_pred_numeric, output_dict=True)
    return accuracy, class_report

def TFIDF2():
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    classifier = joblib.load('tfidf_xgb.joblib')
    x_test = test_data['cleaned_text']
    y_test = test_data['sentiment']
    x_test = vectorizer.transform(x_test)
    y_pred = classifier.predict(x_test)
    
    sentiment_mapping1 = {'good': 1, 'neutral': 0, 'bad': -1}
    sentiment_mapping2 = {1: 1, 2: 0, 0: -1}
    y_test_numeric = list(map(sentiment_mapping1.get, y_test))
    y_pred_numeric = pd.Series([sentiment_mapping2[label] for label in y_pred])
    accuracy = accuracy_score(y_test_numeric, y_pred_numeric)
    class_report = classification_report(y_test_numeric, y_pred_numeric, output_dict=True)
    return accuracy, class_report

def TFIDF3():
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    classifier = joblib.load('tfidf_svc.joblib')
    x_test = test_data['cleaned_text']
    y_test = test_data['sentiment']
    x_test = vectorizer.transform(x_test)
    y_pred = classifier.predict(x_test)
    
    sentiment_mapping1 = {'good': 1, 'neutral': 0, 'bad': -1}
    sentiment_mapping2 = {1: 1, 2: 0, 0: -1}
    y_test_numeric = list(map(sentiment_mapping1.get, y_test))
    y_pred_numeric = pd.Series([sentiment_mapping2[label] for label in y_pred])
    accuracy = accuracy_score(y_test_numeric, y_pred_numeric)
    class_report = classification_report(y_test_numeric, y_pred_numeric, output_dict=True)
    return accuracy, class_report

def TFIDF4():
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    classifier = joblib.load('tfidf_lr.joblib')
    x_test = test_data['cleaned_text']
    y_test = test_data['sentiment']
    x_test = vectorizer.transform(x_test)
    y_pred = classifier.predict(x_test)
    
    sentiment_mapping1 = {'good': 1, 'neutral': 0, 'bad': -1}
    sentiment_mapping2 = {1: 1, 2: 0, 0: -1}
    y_test_numeric = list(map(sentiment_mapping1.get, y_test))
    y_pred_numeric = pd.Series([sentiment_mapping2[label] for label in y_pred])
    accuracy = accuracy_score(y_test_numeric, y_pred_numeric)
    class_report = classification_report(y_test_numeric, y_pred_numeric, output_dict=True)
    return accuracy, class_report

def BOW():
    vectorizer = joblib.load('bow_vectorizer.joblib')
    classifier = joblib.load('bow_model.joblib')
    x_test = test_data['cleaned_text']
    y_test = test_data['sentiment']
    x_test = vectorizer.transform(x_test)
    y_pred = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    sentiment_mapping = {'good': 1, 'neutral': 0, 'bad': -1}
    y_test_numeric = list(map(sentiment_mapping.get, y_test))
    y_pred_numeric = pd.Series([sentiment_mapping[label] for label in y_pred])
    class_report = classification_report(y_test_numeric, y_pred_numeric, output_dict=True)
    return accuracy, class_report

def BOW1():
    vectorizer = joblib.load('bow_vectorizer.joblib')
    classifier = joblib.load('bow_nb.joblib')
    x_test = test_data['cleaned_text']
    y_test = test_data['sentiment']
    x_test = vectorizer.transform(x_test)
    y_pred = classifier.predict(x_test)
    
    sentiment_mapping1 = {'good': 1, 'neutral': 0, 'bad': -1}
    sentiment_mapping2 = {1: 1, 2: 0, 0: -1}
    y_test_numeric = list(map(sentiment_mapping1.get, y_test))
    y_pred_numeric = pd.Series([sentiment_mapping2[label] for label in y_pred])
    accuracy = accuracy_score(y_test_numeric, y_pred_numeric)
    class_report = classification_report(y_test_numeric, y_pred_numeric, output_dict=True)
    return accuracy, class_report

def BOW2():
    vectorizer = joblib.load('bow_vectorizer.joblib')
    classifier = joblib.load('bow_xgb.joblib')
    x_test = test_data['cleaned_text']
    y_test = test_data['sentiment']
    x_test = vectorizer.transform(x_test)
    y_pred = classifier.predict(x_test)
    
    sentiment_mapping1 = {'good': 1, 'neutral': 0, 'bad': -1}
    sentiment_mapping2 = {1: 1, 2: 0, 0: -1}
    y_test_numeric = list(map(sentiment_mapping1.get, y_test))
    y_pred_numeric = pd.Series([sentiment_mapping2[label] for label in y_pred])
    accuracy = accuracy_score(y_test_numeric, y_pred_numeric)
    class_report = classification_report(y_test_numeric, y_pred_numeric, output_dict=True)
    return accuracy, class_report

def BOW3():
    vectorizer = joblib.load('bow_vectorizer.joblib')
    classifier = joblib.load('bow_svc.joblib')
    x_test = test_data['cleaned_text']
    y_test = test_data['sentiment']
    x_test = vectorizer.transform(x_test)
    y_pred = classifier.predict(x_test)
    
    sentiment_mapping1 = {'good': 1, 'neutral': 0, 'bad': -1}
    sentiment_mapping2 = {1: 1, 2: 0, 0: -1}
    y_test_numeric = list(map(sentiment_mapping1.get, y_test))
    y_pred_numeric = pd.Series([sentiment_mapping2[label] for label in y_pred])
    accuracy = accuracy_score(y_test_numeric, y_pred_numeric)
    class_report = classification_report(y_test_numeric, y_pred_numeric, output_dict=True)
    return accuracy, class_report

def BOW4():
    vectorizer = joblib.load('bow_vectorizer.joblib')
    classifier = joblib.load('bow_lr.joblib')
    x_test = test_data['cleaned_text']
    y_test = test_data['sentiment']
    x_test = vectorizer.transform(x_test)
    y_pred = classifier.predict(x_test)
    
    sentiment_mapping1 = {'good': 1, 'neutral': 0, 'bad': -1}
    sentiment_mapping2 = {1: 1, 2: 0, 0: -1}
    y_test_numeric = list(map(sentiment_mapping1.get, y_test))
    y_pred_numeric = pd.Series([sentiment_mapping2[label] for label in y_pred])
    accuracy = accuracy_score(y_test_numeric, y_pred_numeric)
    class_report = classification_report(y_test_numeric, y_pred_numeric, output_dict=True)
    return accuracy, class_report

def TransformerM():
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    classifier = joblib.load('transformer_model.joblib')
    text_list = test_data['cleaned_text'].reset_index(drop=True)
    x_test = pd.DataFrame(model.encode(text_list), columns=[str(i) for i in range(model.encode(text_list).shape[1])])
    y_test = test_data['sentiment']
    y_pred = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    sentiment_mapping = {'good': 1, 'neutral': 0, 'bad': -1}
    y_test_numeric = list(map(sentiment_mapping.get, y_test))
    y_pred_numeric = pd.Series([sentiment_mapping[label] for label in y_pred])
    class_report = classification_report(y_test_numeric, y_pred_numeric, output_dict=True)
    return accuracy, class_report

def W2V1():
    vectorizer = joblib.load('word2vec_model.joblib')
    classifier = joblib.load('w2v_nb.joblib')
    x_test = test_data['cleaned_text']
    y_test = test_data['sentiment']
    review_feature_vectors = []
    test_sentences = [text.split() for text in x_test]
    for review in test_sentences:
        feature_vector = np.zeros((100,), dtype="float64")
        n_words = 0.
        for word in review:
            if word in vectorizer.wv.index_to_key:
                n_words += 1.
                feature_vector = np.add(feature_vector, vectorizer.wv[word])
    
        if n_words:
            feature_vector = np.divide(feature_vector, n_words)

review_feature_vectors.append(feature_vector)

x_test = np.array(review_feature_vectors)
y_pred = classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
sentiment_mapping1 = {'good': 1, 'neutral': 0, 'bad': -1}
    sentiment_mapping2 = {1: 1, 2: 0, 0: -1}
    y_test_numeric = list(map(sentiment_mapping1.get, y_test))
    y_pred_numeric = pd.Series([sentiment_mapping2[label] for label in y_pred])
    accuracy = accuracy_score(y_test_numeric, y_pred_numeric)
    class_report = classification_report(y_test_numeric, y_pred_numeric, output_dict=True)
    return accuracy, class_report

def W2V2():
    vectorizer = joblib.load('word2vec_model.joblib')
    classifier = joblib.load('w2v_xgb.joblib')
    x_test = test_data['cleaned_text']
    y_test = test_data['sentiment']
    review_feature_vectors = []
    test_sentences = [text.split() for text in x_test]
    for review in test_sentences:
        feature_vector = np.zeros((100,), dtype="float64")
        n_words = 0.
        for word in review:
            if word in vectorizer.wv.index_to_key:
                n_words += 1.
                feature_vector = np.add(feature_vector, vectorizer.wv[word])
    
        if n_words:
            feature_vector = np.divide(feature_vector, n_words)

review_feature_vectors.append(feature_vector)

x_test = np.array(review_feature_vectors)
y_pred = classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
sentiment_mapping1 = {'good': 1, 'neutral': 0, 'bad': -1}
    sentiment_mapping2 = {1: 1, 2: 0, 0: -1}
    y_test_numeric = list(map(sentiment_mapping1.get, y_test))
    y_pred_numeric = pd.Series([sentiment_mapping2[label] for label in y_pred])
    accuracy = accuracy_score(y_test_numeric, y_pred_numeric)
    class_report = classification_report(y_test_numeric, y_pred_numeric, output_dict=True)
    return accuracy, class_report

def W2V3():
    vectorizer = joblib.load('word2vec_model.joblib')
    classifier = joblib.load('w2v_svc.joblib')
    x_test = test_data['cleaned_text']
    y_test = test_data['sentiment']
    review_feature_vectors = []
    test_sentences = [text.split() for text in x_test]
    for review in test_sentences:
        feature_vector = np.zeros((100,), dtype="float64")
        n_words = 0.
        for word in review:
            if word in vectorizer.wv.index_to_key:
                n_words += 1.
                feature_vector = np.add(feature_vector, vectorizer.wv[word])
    
        if n_words:
            feature_vector = np.divide(feature_vector, n_words)

review_feature_vectors.append(feature_vector)

x_test = np.array(review_feature_vectors)
y_pred = classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
sentiment_mapping1 = {'good': 1, 'neutral': 0, 'bad': -1}
    sentiment_mapping2 = {1: 1, 2: 0, 0: -1}
    y_test_numeric = list(map(sentiment_mapping1.get, y_test))
    y_pred_numeric = pd.Series([sentiment_mapping2[label] for label in y_pred])
    accuracy = accuracy_score(y_test_numeric, y_pred_numeric)
    class_report = classification_report(y_test_numeric, y_pred_numeric, output_dict=True)
    return accuracy, class_report

def W2V4():
    vectorizer = joblib.load('word2vec_model.joblib')
    classifier = joblib.load('w2v_lr.joblib')
    x_test = test_data['cleaned_text']
    y_test = test_data['sentiment']
    review_feature_vectors = []
    test_sentences = [text.split() for text in x_test]
    for review in test_sentences:
        feature_vector = np.zeros((100,), dtype="float64")
        n_words = 0.
        for word in review:
            if word in vectorizer.wv.index_to_key:
                n_words += 1.
                feature_vector = np.add(feature_vector, vectorizer.wv[word])
    
        if n_words:
            feature_vector = np.divide(feature_vector, n_words)

review_feature_vectors.append(feature_vector)

x_test = np.array(review_feature_vectors)
y_pred = classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
sentiment_mapping1 = {'good': 1, 'neutral': 0, 'bad': -1}
    sentiment_mapping2 = {1: 1, 2: 0, 0: -1}
    y_test_numeric = list(map(sentiment_mapping1.get, y_test))
    y_pred_numeric = pd.Series([sentiment_mapping2[label] for label in y_pred])
    accuracy = accuracy_score(y_test_numeric, y_pred_numeric)
    class_report = classification_report(y_test_numeric, y_pred_numeric, output_dict=True)
    return accuracy, class_report

def compare_models():
    results = []
    nb_acc, nb_crep = TFIDF1()
    xgb_acc,xgb_crep = TFIDF2()
    svc_acc, svc_crep = TFIDF3()
    lr_acc, lr_crep = TFIDF4()
    results.append({'Classifier': 'NB', 'Accuracy': nb_acc, **nb_crep['weighted avg']})
    results.append({'Classifier': 'XGB', 'Accuracy': xgb_acc, **xgb_crep['weighted avg']})
    results.append({'Classifier': 'SVC', 'Accuracy': svc_acc, **svc_crep['weighted avg']})
    results.append({'Classifier': 'LR', 'Accuracy': lr_acc, **lr_crep['weighted avg']})
    return pd.DataFrame(results)




# def compare_models(models, X_test, y_test):
#     results = []
#     for model_name, model in models.items():
#         try:
#             # Try to predict using the current model
#             y_pred = model.predict(X_test)

#             # Convert sentiment labels to numerical values
#             sentiment_mapping = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
#             y_test_numeric = y_test.map(sentiment_mapping)
#             y_pred_numeric = pd.Series([sentiment_mapping[label] for label in y_pred])

#             accuracy = accuracy_score(y_test_numeric, y_pred_numeric)
#             class_report = classification_report(y_test_numeric, y_pred_numeric, output_dict=True)
#             results.append({'Model': model_name, 'Accuracy': accuracy, **class_report['weighted avg']})
#         except Exception as e:
#             print(f"Skipping {model_name} due to the following error: {str(e)}")

#     return pd.DataFrame(results)


import seaborn as sns
import matplotlib.pyplot as plt

def plot_comparison_graphs(results_df):
    # Print the columns in results_df
    print("Columns in results_df:", results_df.columns)
    
    # Plot Accuracy
    plt.rcParams['font.size']=17
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Classifier', y='Accuracy', data=results_df)
    plt.title('Classifier Accuracy Comparison - TFIDF')
    plt.ylim([0.5, 0.8])
    plt.show()
    
    # Plot Precision
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Classifier', y='precision', data=results_df)
    plt.title('Classifier Precision Comparison - TFIDF')
    plt.ylim([0.5, 0.8])
    plt.show()
    
    # Plot Recall
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Classifier', y='recall', data=results_df)
    plt.title('Classifier Recall Comparison - TFIDF')
    plt.ylim([0.5, 0.8])
    plt.show()
    
    # Plot F1 Score
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Classifier', y='f1-score', data=results_df)
    plt.title('Classifier F1 Score Comparison - TFIDF')
    plt.ylim([0.5, 0.8])
    plt.show()


# Compare models and plot graphs
results_df = compare_models()
plot_comparison_graphs(results_df)
