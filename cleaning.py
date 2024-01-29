import pandas as pd
import re
import json
from gensim.utils import simple_preprocess

def clean_text(text, stop_words):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = simple_preprocess(text)
    tokens = [word for word in tokens if word not in stop_words]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

def read_json(file_path):
    with open(file_path, 'r') as data_file:
        json_data = json.load(data_file)
        column_names = json_data.get('columns', [])
        data = json_data.get('data', [])
    return column_names, data

def clean_and_save_data(data, columns_to_keep, stop_words, cleaned_data_file):
    review_df = pd.DataFrame(data, columns=column_names)
    review_df_cleaned = review_df[columns_to_keep]
    review_df_cleaned = review_df_cleaned.dropna()
    review_df_cleaned = review_df_cleaned.astype(str)
    review_df_cleaned = review_df_cleaned.drop_duplicates()

    if 'text' in review_df_cleaned.columns:
        review_df_cleaned['cleaned_text'] = review_df_cleaned['text'].apply(lambda x: clean_text(x, stop_words))
        review_df_cleaned.to_csv(cleaned_data_file, index=False)
        print(review_df_cleaned[['text', 'cleaned_text', 'stars']])
        print(f"\nCleaned data saved to: {cleaned_data_file}")
    else:
        print("The 'text' column is not present in the DataFrame.")

json_file_path = "Data.json"
cleaned_data_file = "Cleaned_data.csv"
columns_to_keep = ['business_id', 'date', 'stars', 'text', 'user_id']
custom_stop_words = set(['are', 'is', 'the', 'and', 'or'])

column_names, data = read_json(json_file_path)
clean_and_save_data(data, columns_to_keep, custom_stop_words, cleaned_data_file)
