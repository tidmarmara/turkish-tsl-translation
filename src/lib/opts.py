import argparse

class Opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Turkish-TSL Translation')

        ################### TRAIN ###################
        self.parser.add_argument('--data-path', '-p', default='configs/data.yaml', help='YAML file that stores dataset parameters')
        self.parser.add_argument('--model-config', '-cfg', default='configs/networks/transformer.yaml', help='YAML fiel that stores the model paramters')        
        self.parser.add_argument('--token-type', '-t', default='word', help='Type of the token either word or char')
        self.parser.add_argument('--model-type', '-m', default='transformer', help="This parameter allows to choose the model type either 'rnn' or 'transformer' can be given as input")
        self.parser.add_argument('--num-epochs', '-e', default=20, type=int, help='Number of epochs for the training')
        self.parser.add_argument('--batch-size', '-b', default=64, type=int, help='Batch size for the training')
        self.parser.add_argument('--root-path', '-r', default='results', type=str, help='Root path for the saved models')
        
        ################### TEST ###################
        self.parser.add_argument('--inp-tokenizer', type=str, help='Direct path of the input tokenizer JSON file')
        self.parser.add_argument('--targ-tokenizer', type=str, help='Direct path of the target tokenizer JSON file')
        self.parser.add_argument('--ckpt-path', type=str, help='Direct path of the trained model checkpoints')

    def parse(self):
        args = self.parser.parse_args()
        return args
