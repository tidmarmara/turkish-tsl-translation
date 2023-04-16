import _init_paths
from dataset import Dataset
from opts import Opts
from utils import Model, Trainer, Evaluator
import time
from loguru import logger
import tensorflow as tf
import os
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def main(opts):
    # Create the dataloader
    dataLoader = Dataset(opts)

    # Create the dataset splits in raw txt and tensor formats. This variable also stores the input and target sentence tokenizers
    datasets = dataLoader.create_dataset()

    # Convert the train data tensors to batches
    train_data = dataLoader.convert_to_batch(datasets['train']['tensor-data']['input'], datasets['train']['tensor-data']['target'], opts.batch_size)

    # Convert the validation data tensors to batches
    valid_data_inp, valid_data_tar = datasets['valid']['raw-data']['input'], datasets['valid']['raw-data']['target']
    train_inp_tokenizer, train_tar_tokenizer = datasets['train']['tokenizer']['input'], datasets['train']['tokenizer']['target']
    max_valid_tar_len = datasets['valid']['tensor-data']['target'].shape[1]
    
    # Load the model
    modelLoader = Model(opts, dataLoader)
    model = modelLoader.get_model(datasets['train']['tokenizer']['input'], datasets['train']['tokenizer']['target'])

    trainer = Trainer(model, modelLoader, opts)

    tokenizer_save_path = os.path.join(trainer.exp_save_path, 'inp_tokenizer.json')
    dataLoader.save_tokenizer(datasets['train']['tokenizer']['input'], tokenizer_save_path)

    tokenizer_save_path = os.path.join(trainer.exp_save_path, 'targ_tokenizer.json')
    dataLoader.save_tokenizer(datasets['train']['tokenizer']['target'], tokenizer_save_path)

    valid_step = 0
    for epoch in range(1, opts.num_epochs+1):
        start = time.time()

        # Validation
        if (epoch % opts.validation_step) == 0:
            trainer.valid_loss.reset_states()
            trainer.valid_accuracy.reset_states()
            logger.info('Validation started...')

            evaluator = Evaluator(trainer.model, dataLoader, modelLoader)
            model_scores = evaluator.eval(valid_data_inp, valid_data_tar, train_inp_tokenizer, train_tar_tokenizer, max_valid_tar_len)
            logger.info(f"BLEU-1: {model_scores['bleu1']}\tBLEU-4: {model_scores['bleu4']}\tTER: {model_scores['ter']}\tWER: {model_scores['wer']}\tCHRF: {model_scores['chrf']}\tAvg-Time: {model_scores['avg-time']}")

            with trainer.valid_board_writer.as_default():
                tf.summary.scalar('BLEU-1', model_scores['bleu1'], step=valid_step)
                tf.summary.scalar('BLEU-4', model_scores['bleu4'], step=valid_step)
                tf.summary.scalar('TER', model_scores['ter'], step=valid_step)
                tf.summary.scalar('WER', model_scores['wer'], step=valid_step)
                tf.summary.scalar('CHRF', model_scores['chrf'], step=valid_step)

            valid_step += 1

            if model_scores['bleu1'] > trainer.best_model_acc:
                trainer.ckpt_manager_best.save()
                logger.success("Saving best model\n")
                trainer.best_model_acc = model_scores['bleu1']

        for (batch, (inp, tar)) in enumerate(train_data):
            trainer.run(inp, tar)     
            trainer.ckpt_manager_last.save()  

            if batch % 100 == 0:
                batch_loss, batch_accuracy = trainer.train_loss.result(), trainer.train_accuracy.result()
                logger.info(f'Train --> Epoch->{epoch}\tBatch->{batch}\tLoss->{batch_loss:.4f}\tAccuracy->{batch_accuracy:.4f}')

            with trainer.train_board_writer.as_default():
                tf.summary.scalar('loss', trainer.train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', trainer.train_accuracy.result(), step=epoch)

        logger.info(f'End of Epoch-{epoch} Loss: {trainer.train_loss.result():.4f}\tAccuracy: {trainer.train_accuracy.result():.4f}')

        epoch_time = time.time() - start
        logger.info(f'Time taken for 1 epoch: {epoch_time:.2f} secs\n')
        # logger.success("Saving last epoch model")
        trainer.ckpt_manager_last_epoch.save()

if __name__ == '__main__':
    opts = Opts().parse()
    main(opts)


