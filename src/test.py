import _init_paths
from opts import Opts
from loguru import logger
from utils import Model
from dataset import Dataset
import os
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

def main(opts):
    dataset = Dataset(opts)
    input_tokenizer = dataset.load_tokenizer(os.path.join(opts.exp_path, 'inp_tokenizer.json'))
    target_tokenizer = dataset.load_tokenizer(os.path.join(opts.exp_path, 'targ_tokenizer.json'))

    model_creator = Model(opts, dataset)
    model = model_creator.get_model(input_tokenizer, target_tokenizer)
    model = model_creator.load_model(model, os.path.join(opts.exp_path, 'ckpts', 'last'))

    model_type = model_creator.model_config['experiment-parameters']['model-type'].lower()
    
    sentence = "film nasıldı "

    if model_type == "rnn-based":
        pred, sentence, _ = model_creator.evaluate(input_tokenizer, target_tokenizer, sentence, model, 14)
    elif model_type == "transformer":
        pred, _ = model_creator.predict_sentence(model, sentence, input_tokenizer, target_tokenizer, 20)
    
    logger.success(f"Predicted sentence: {pred}")

if __name__ == '__main__':
    opts = Opts().parse()
    main(opts)
