import tensorflow as tf
import re
from unicode_tr import unicode_tr
from loguru import logger
import yaml
import os
import json

class Dataset():
    def __init__(self, opts):
        self.opts = opts

        if self.opts.token_type.lower() == 'word':
            self.start_token = '<start>'
            self.end_token = '<end>'
        elif self.opts.token_type.lower() == 'char':
            self.start_token = '<'
            self.end_token = '>'

    def preprocess_sentence(self, sent):
        sent = unicode_tr(sent).lower()
        sent = re.sub("'", '', sent)
        sent = re.sub(" +", " ", sent)
        sent = sent.strip()
        return sent
    
    def exclude_start_end_tokens(self, sentence, start_token, end_token):
        if start_token in sentence:
            sentence = sentence[len(start_token):]
        if end_token in sentence:
            sentence = sentence[:-len(end_token)]
        return sentence.strip()

    def create_dataset(self):
        with open(self.opts.data_path) as f:
            config = yaml.safe_load(f)
        
        datasets = {}
        for data_name in config.keys():
            if data_name not in datasets.keys():
                datasets[data_name] = {'raw-data': {}, 'tensor-data': {}, 'tokenizer': {}}

            if data_name.lower() == 'path':
                continue

            data_path = os.path.join(config['path'], config[data_name])
            inp_sents_raw, targ_sents_raw = self.load_dataset_raw(data_path)
            inp_sents_tensor, inp_tokenizer, targ_sents_tensor, targ_tokenizer = self.convert_to_tensors(inp_sents_raw, targ_sents_raw)

            datasets[data_name]['raw-data']['input'] = inp_sents_raw
            datasets[data_name]['raw-data']['target'] = targ_sents_raw
            datasets[data_name]['tensor-data']['input'] = inp_sents_tensor
            datasets[data_name]['tensor-data']['target'] = targ_sents_tensor
            datasets[data_name]['tokenizer']['input'] = inp_tokenizer
            datasets[data_name]['tokenizer']['target'] = targ_tokenizer

            logger.info(f"({data_name.upper()}) Max length input: {inp_sents_tensor.shape[1]},\ttarget: {targ_sents_tensor.shape[1]}\n")
            logger.info(f"({data_name.upper()}) Vocabulary length input: {len(inp_tokenizer.word_counts)},\ttarget: {len(targ_tokenizer.word_counts)}\n")
            logger.info(f"({data_name.upper()}) Total number of sentences: {len(inp_sents_tensor)}\n")
        
        return datasets

    def save_tokenizer(self, tokenizer, save_path):
        tokenizer_json = tokenizer.to_json()
        with open(save_path, 'w') as f:
            json.dump(tokenizer_json, f)
    
    def load_tokenizer(self, json_path):
        with open(json_path) as tokenizer_file:
            tokenizer_json = json.load(tokenizer_file)
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)
        return tokenizer

    def convert_to_batch(self, input_tensor_train, target_tensor_train, batch_size):
        # Dataset slicer, each time according to the batch size it returns data slice
        dataset_batched = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train))
        dataset_batched = dataset_batched.batch(batch_size, drop_remainder=True)
        dataset_batched = dataset_batched.shuffle(batch_size, reshuffle_each_iteration=True)

        return dataset_batched

    def load_dataset_raw(self, path):
        logger.info(f"Reading from {path}")
        lines = open(path, "r", encoding='UTF-8').readlines()

        tur_data = []
        tid_data = []

        for line in lines:
            tur_sent = line.split("\t")[0].strip()
            tid_sent = line.split("\t")[1].strip()

            tur_sent = self.preprocess_sentence(tur_sent)
            tid_sent = self.preprocess_sentence(tid_sent)
            
            # Add token only to target sentences
            tid_sent = self.start_token + ' ' + tid_sent + ' ' + self.end_token

            tur_data.append(tur_sent)
            tid_data.append(tid_sent)

        return tur_data, tid_data
        
    def convert_to_tensors(self, inp_sents_raw, targ_sents_raw):
        if self.opts.token_type.lower() == 'word':
            input_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
            target_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        elif self.opts.token_type.lower() == 'char':
            input_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', char_level=True)
            target_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', char_level=True)

        input_tokenizer.fit_on_texts(inp_sents_raw)
        inp_sents_tensor = input_tokenizer.texts_to_sequences(inp_sents_raw)
        inp_sents_tensor = tf.keras.preprocessing.sequence.pad_sequences(inp_sents_tensor, padding='post')
        
        target_tokenizer.fit_on_texts(targ_sents_raw)
        targ_sents_tensor = target_tokenizer.texts_to_sequences(targ_sents_raw)
        targ_sents_tensor = tf.keras.preprocessing.sequence.pad_sequences(targ_sents_tensor, padding='post')

        return inp_sents_tensor, input_tokenizer, targ_sents_tensor, target_tokenizer