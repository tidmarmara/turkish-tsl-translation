import os
import numpy as np

ROOT_PATH = 'dataset'
TRAIN_PATH = os.path.join(ROOT_PATH, 'train-tr2tsl-wordbased.txt')
VALID_PATH = os.path.join(ROOT_PATH, 'train-tr2tsl-wordbased.txt')

train_file = open(TRAIN_PATH, 'r', encoding='utf-8').read().splitlines()
valid_file = open(VALID_PATH, 'r', encoding='utf-8').read().splitlines()

word_freq = 0

train_dict = {}
for line in train_file:
    if '\n' in line:
        tr_sent, tid_sent = line.strip().split('\t')
        if (len(tr_sent.split()) > 0) and (len(tid_sent.split()) > 0):
            for word in tr_sent.split():
                if word not in train_dict.keys():
                    train_dict[word] = 1
                else:
                    train_dict[word] += 1

new_valid = []
for line in valid_file:
    if '\n' in line:
        tr_sent, tid_sent = line.strip().split('\t')
        if (len(tr_sent.split()) > 0) and (len(tid_sent.split()) > 0):
            check = False
            for word in tr_sent.split():
                if word in train_dict.keys():
                    if train_dict[word] >= word_freq:
                        check = True
                    else:
                        check = False
                        break
                else:
                    check = False
                    break
            if check:
                new_valid.append(line)

print("After processing: ", len(new_valid))
new_valid_file = open(os.path.join(ROOT_PATH, 'new_valid_' + str(word_freq) + '.txt'), 'w', encoding='utf-8')
for line in new_valid:
    new_valid_file.write(line + '\n')
