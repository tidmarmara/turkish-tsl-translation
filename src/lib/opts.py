import argparse

class Opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Turkish-TSL Translation')

        ################### TRAIN ###################
        self.parser.add_argument('--data-path', '-p', default='configs/data.yaml', help='YAML file that stores dataset parameters')
        self.parser.add_argument('--model-config', '-cfg', default='configs/networks/transformer.yaml', help='YAML fiel that stores the model paramters')        
        self.parser.add_argument('--token-type', '-t', default='word', help='Type of the token either word or char')
        self.parser.add_argument('--num-epochs', '-e', default=20, type=int, help='Number of epochs for the training')
        self.parser.add_argument('--batch-size', '-b', default=64, type=int, help='Batch size for the training')
        self.parser.add_argument('--root-path', '-r', default='results', type=str, help='Root path for the saved models')
        self.parser.add_argument('--validation-step', '-s', type=int, default=5, help='Specifies after how many epochs to apply validation')
        
        ################### TEST ###################
        self.parser.add_argument('--exp-path', type=str, help='root path for the experiment folder that contains checkpoints, and tokenizer JSON files')

    def parse(self):
        args = self.parser.parse_args()
        return args
