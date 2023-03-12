import argparse

class Opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Turkish-TSL Translation')
        self.parser.add_argument('--data-path', '-p', default='configs/data.yaml', help='YAML fiel that stores dataset parameters')
        self.parser.add_argument('--model-type', '-t', default='word', help='Type of the model either word or char')
    
    def parse(self):
        args = self.parser.parse_args()
        return args
