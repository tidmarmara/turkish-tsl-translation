import _init_paths
import os
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import tensorflow as tf
from loguru import logger

from opts import Opts
from dataset import Dataset
from utils import Model, Evaluator
from models.transformer import create_masks

def main(opts):
    # Create the dataloader
    dataLoader = Dataset(opts)

    # Create the dataset splits in raw txt and tensor formats. This variable also stores the input and target sentence tokenizers
    datasets = dataLoader.create_dataset()

    # Convert the test data tensors to batches
    test_data_inp, test_data_tar = datasets['test']['raw-data']['input'], datasets['test']['raw-data']['target']

    input_tokenizer = dataLoader.load_tokenizer(os.path.join(opts.exp_path, 'inp_tokenizer.json'))
    target_tokenizer = dataLoader.load_tokenizer(os.path.join(opts.exp_path, 'targ_tokenizer.json'))

    model_creator = Model(opts, dataLoader)
    model = model_creator.get_model(input_tokenizer, target_tokenizer)
    model = model_creator.load_model(model, os.path.join(opts.exp_path, 'ckpts', 'last'))

    evaluator = Evaluator(model, dataLoader, model_creator)
    model_scores = evaluator.eval(test_data_inp, test_data_tar, input_tokenizer, target_tokenizer, 50)
    logger.info(f"Scores: {model_scores}")

if __name__ == '__main__':
    opts = Opts().parse()
    main(opts)