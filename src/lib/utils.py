import tensorflow as tf
import re
from unicode_tr import unicode_tr
from loguru import logger
import yaml

class Dataset():
    def __init__(self, opts):
        self.opts = opts
        self.model_type = opts.model_type.lower()

        if self.model_type == 'word':
            self.input_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
            self.target_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
            self.start_token = '<start>'
            self.end_token = '<end>'
        
        elif self.model_type == 'char':
            self.input_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', char_level=True)
            self.target_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', char_level=True)
            self.start_token = '<'
            self.end_token = '>'

    def preprocess_sentence(self, sent):
        sent = unicode_tr(sent).lower()
        sent = re.sub("'", '', sent)
        sent = re.sub(" +", " ", sent)
        sent = sent.strip()
        return sent
    
    def create_dataset(self):
        with open(self.opts.data_path) as f:
            config = yaml.safe_load(f)
        
        datasets = {}
        for data_name in config.keys():
            if data_name not in datasets.keys():
                datasets[data_name] = {'raw-data': {}, 'tensor-data': {}}

            inp_sents_raw, targ_sents_raw = self.load_dataset_raw(config[data_name])
            inp_sents_tensor, targ_sents_tensor = self.convert_to_tensors(inp_sents_raw, targ_sents_raw)

            datasets[data_name]['raw-data']['input'] = inp_sents_raw
            datasets[data_name]['raw-data']['target'] = targ_sents_raw
            datasets[data_name]['tensor-data']['input'] = inp_sents_tensor
            datasets[data_name]['tensor-data']['target'] = targ_sents_tensor
        return datasets

    def load_dataset_raw(self, path):
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
        self.input_tokenizer.fit_on_texts(inp_sents_raw)
        inp_sents_tensor = self.input_tokenizer.texts_to_sequences(inp_sents_raw)
        inp_sents_tensor = tf.keras.preprocessing.sequence.pad_sequences(inp_sents_tensor, padding='post')
        
        self.target_tokenizer.fit_on_texts(targ_sents_raw)
        targ_sents_tensor = self.target_tokenizer.texts_to_sequences(targ_sents_raw)
        targ_sents_tensor = tf.keras.preprocessing.sequence.pad_sequences(targ_sents_tensor, padding='post')

        return inp_sents_tensor, targ_sents_tensor