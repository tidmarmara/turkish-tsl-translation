import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')

def read_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [line.strip().split('\t') for line in file]
    return data

def get_vocabulary(data):
    vocab = set()
    for line in data:
        print("Line: ", line)
        source, target = line[0], line[1]
        words = source.split() + target.split()
        vocab.update(words)
    return vocab

def save_to_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for line in data:
            file.write('\t'.join(line) + '\n')

def filter_data_with_training_vocab(data, train_vocab):
    filtered_data = []
    for item in data:
        source, target = item[0], item[1]
        words = source.split() + target.split()
        if all(word in train_vocab for word in words):
            filtered_data.append(item)
    return filtered_data

data = read_dataset('dataset/whole-dataset-cleaned.txt')

source_sentences = [item[0] for item in data]
embeddings = model.encode(source_sentences)

kmeans = KMeans(n_clusters=3, random_state=0).fit(embeddings)
labels = kmeans.labels_

data_with_labels = list(zip(data, labels))

train, temp = train_test_split(data_with_labels, train_size=0.7, stratify=labels)
validation, test = train_test_split(temp, test_size=0.5, stratify=np.array(temp)[:, 1])

train_data = [item[0] for item in train]
validation_data = [item[0] for item in validation]
test_data = [item[0] for item in test]

train_data = [item[0] for item in train]
train_vocab = get_vocabulary(train_data)

validation_data_temp = [item[0] for item in validation]
test_data_temp = [item[0] for item in test]

validation_data = filter_data_with_training_vocab(validation_data_temp, train_vocab)
test_data = filter_data_with_training_vocab(test_data_temp, train_vocab)


save_to_file(train_data, 'train_dataset.txt')
save_to_file(validation_data, 'validation_dataset.txt')
save_to_file(test_data, 'test_dataset.txt')
