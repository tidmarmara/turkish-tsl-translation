# import os
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split

# ROOT_PATH = 'dataset'
# PATH = os.path.join(ROOT_PATH, 'whole-dataset.txt')

# file = open(PATH, 'r', encoding='utf-8').read().splitlines()

# # Load your dataset (replace this with actual loading code)
# input_sentences = []  # List of input sentences
# target_sentences = []  # List of corresponding target sentences

# for line in file:
#     if '\t' in line:
#         tr_sentence, tid_sentence = line.split('\t')[0], line.split('\t')[1]
#         if (len(tr_sentence.split()) > 0) and (len(tid_sentence.split()) > 0):
#             input_sentences.append(tr_sentence)
#             target_sentences.append(tid_sentence)

########################################################################################3

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = []
        for line in f.readlines():
            if '\t' in line:
                tr_sentence, tid_sentence = line.split('\t')[0], line.split('\t')[1]
                if (len(tr_sentence.split()) > 0) and (len(tid_sentence.split()) > 0):
                    data.append(line.strip().split('\t'))
        # data = [line.strip().split('\t') for line in f.readlines()]
    return pd.DataFrame(data, columns=['input', 'target'])

def compute_avg_word_freq(sentence, word_freq_dict):
    tokens = sentence.split()
    avg_freq = sum(word_freq_dict.get(token, 0) for token in tokens) / len(tokens)
    return avg_freq

def create_word_freq_dict(dataset):
    word_freq_dict = {}
    for index, row in dataset.iterrows():
        for sentence in [row['input'], row['target']]:
            for token in sentence.split():
                if token in word_freq_dict:
                    word_freq_dict[token] += 1
                else:
                    word_freq_dict[token] = 1
    return word_freq_dict

def stratified_split(dataset, val_size=0.2, test_size=0.2, random_state=None, num_clusters=5):
    word_freq_dict = create_word_freq_dict(dataset)
    dataset['avg_word_freq'] = dataset['input'].apply(compute_avg_word_freq, args=(word_freq_dict,))

    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
    dataset['cluster'] = kmeans.fit_predict(dataset[['avg_word_freq']])

    train_data, remaining = train_test_split(dataset, test_size=val_size + test_size, stratify=dataset['cluster'], random_state=random_state)

    remaining_frac = 1 - (val_size / (val_size + test_size))
    val_data, test_data = train_test_split(remaining, test_size=remaining_frac, stratify=remaining['cluster'], random_state=random_state)

    return train_data.drop(columns=['avg_word_freq', 'cluster']), val_data.drop(columns=['avg_word_freq', 'cluster']), test_data.drop(columns=['avg_word_freq', 'cluster'])

dataset_file_path = 'dataset/whole-dataset.txt'
dataset = load_dataset(dataset_file_path)
train_data, val_data, test_data = stratified_split(dataset)


print(train_data.head())
print(len(train_data))
print(val_data.head())
print(len(val_data))
print(test_data.head())
print(len(test_data))

new_train = open('dataset/new_train.txt', 'w', encoding='utf-8')
new_valid = open('dataset/new_valid.txt', 'w', encoding='utf-8')
new_test = open('dataset/new_test.txt', 'w', encoding='utf-8')

for indx, sent in train_data.iterrows():
    new_train.write(sent['input'] + '\t' + sent['target'] + '\n')
for indx, sent in val_data.iterrows():
    new_valid.write(sent['input'] + '\t' + sent['target'] + '\n')
for indx, sent in test_data.iterrows():
    new_test.write(sent['input'] + '\t' + sent['target'] + '\n')