

FILE_PATH = "dataset/valid-tr2tsl-wordbased.txt"
lines = open(FILE_PATH, 'r', encoding='utf-8').read().splitlines()

data = {1: [], 2: [], 3: [], 4: []}
count = 0
for line in lines:
    tr_sent = line.split('\t')[0]
    tid_sent = line.split('\t')[1]

    sent_length = len(tr_sent.split())
    if sent_length < (len(data.keys()) + 1):
        data[sent_length].append(line)
        count += 1

print(data)
print("Sentences: ", count)
