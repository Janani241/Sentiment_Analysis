import json
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import tree


#def phrase2list(phrase):
#    arr = [ord(c) for c in phrase]
#    return arr, len(arr)


#load data. Only 1k reviews here for test

data_file = open("yelp_academic_dataset_review.json")
data = []
data_cnt = 0
for line in data_file:
    data.append(json.loads(line))
    data_cnt += 1
    if(data_cnt > 1000):
        break
review_df = pd.DataFrame(data)
data_file.close()
#print(data[0])
review_df.to_json('simple_review.json', orient = 'split', compression = 'infer', index = 'true')

text = review_df['text']
sentiment = review_df['stars']


#naive word bagging

word_bag = []
sentence_array = []
for i in range(data_cnt):
    sentence = text[i]
    for word in sentence:
        if word not in word_bag:
            word_bag.append(word)

for i in range(data_cnt):
    sentence = text[i]
    vector = [0 for j in range(len(word_bag))]
    sentence_array.append(vector)
    for word in sentence:
        sentence_array[i][word_bag.index(word)] += 1

#get array of sentence and sentiment

sentence_array = np.array(sentence_array)
sentiment = np.array(sentiment)

print(sentence_array.shape)
print(sentiment.shape)

#standardization

scaler = StandardScaler()
scaler = scaler.fit(sentence_array)
sentence_array = scaler.transform(sentence_array)

#800 for train, 200 for test

train_cnt =800
train_x = sentence_array[0:train_cnt]
train_y = sentiment[0:train_cnt]

test_x = sentence_array[train_cnt:-1]
test_y = sentiment[train_cnt:-1]

#naive dicision tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_x, train_y)

#0.34
score = clf.score(test_x,test_y)
print(score)


