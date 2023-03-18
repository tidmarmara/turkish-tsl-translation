import _init_paths
from opts import Opts
from loguru import logger
from utils import Model
from dataset import Dataset

def main(opts):
    dataset = Dataset(opts)
    input_tokenizer = dataset.load_tokenizer(opts.inp_tokenizer)
    target_tokenizer = dataset.load_tokenizer(opts.targ_tokenizer)

    model_creator = Model(opts, dataset)
    model = model_creator.get_model(input_tokenizer, target_tokenizer)
    model = model_creator.load_model(model, opts.ckpt_path)
    
    sentence = "sen de gelmek istiyor musun?"
    pred, _ = model_creator.predict_sentence(model, sentence, input_tokenizer, target_tokenizer, 50)
    logger.success(f"Predicted sentence: {pred}")

if __name__ == '__main__':
    opts = Opts().parse()
    main(opts)
