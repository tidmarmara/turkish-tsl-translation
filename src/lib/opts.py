import argparse

class Opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Turkish-TSL Translation')
        self.parser.add_argument('--data-path', '-p', default='configs/data.yaml', help='YAML fiel that stores dataset parameters')
        self.parser.add_argument('--token-type', '-t', default='word', help='Type of the token either word or char')
        self.parser.add_argument('--model-type', '-m', default='transformer', help="This parameter allows to choose the model type either 'rnn' or 'transformer' can be given as input")

    def parse(self):
        args = self.parser.parse_args()
        return args
