import _init_paths
from dataset import Dataset
from opts import Opts
from utils import Model, Trainer
import time
from loguru import logger
import tensorflow as tf
import os

def main(opts):
    # Create the dataloader
    dataLoader = Dataset(opts)

    # Create the dataset splits in raw txt and tensor formats. This variable also stores the input and target sentence tokenizers
    datasets = dataLoader.create_dataset()

    # Convert the train data tensors to batches
    train_data = dataLoader.convert_to_batch(datasets['train']['tensor-data']['input'], datasets['train']['tensor-data']['target'], opts.batch_size)

    # Convert the validation data tensors to batches
    valid_data = dataLoader.convert_to_batch(datasets['valid']['tensor-data']['input'], datasets['valid']['tensor-data']['target'], opts.batch_size)

    # Convert the test data tensors to batches
    test_data = dataLoader.convert_to_batch(datasets['test']['tensor-data']['input'], datasets['test']['tensor-data']['target'], opts.batch_size)
    
    # Load the model
    modelLoader = Model(opts, dataLoader)
    model = modelLoader.get_model(datasets['train']['tokenizer']['input'], datasets['train']['tokenizer']['target'])

    trainer = Trainer(model, modelLoader, opts)

    tokenizer_save_path = os.path.join(trainer.exp_save_path, 'inp_tokenizer.json')
    dataLoader.save_tokenizer(datasets['train']['tokenizer']['input'], tokenizer_save_path)

    tokenizer_save_path = os.path.join(trainer.exp_save_path, 'targ_tokenizer.json')
    dataLoader.save_tokenizer(datasets['train']['tokenizer']['target'], tokenizer_save_path)

    for epoch in range(opts.num_epochs):
        start = time.time()

        trainer.train_loss.reset_states()
        trainer.train_accuracy.reset_states()
        trainer.epoch = epoch

        for (batch, (inp, tar)) in enumerate(train_data):
            # print(inp)
            trainer.train_step(inp, tar)
        
            #ter, wer, chrf, bleu1, bleu4 = evaluate_model(input_text_train, target_text_train, inp_lang_train, targ_lang_train, process, trainer, is_valid=False)
            if batch % 100 == 0:
                batch_loss, batch_accuracy = trainer.train_loss.result(), trainer.train_accuracy.result()
                logger.info(f'Epoch->{epoch + 1}\tBatch->{batch}\tLoss->{batch_loss:.4f}\tAccuracy->{batch_accuracy:.4f}')
                logger.success("Saving last model")
                trainer.ckpt_manager_last.save()

                if batch_accuracy > trainer.best_model_acc:
                    trainer.ckpt_manager_best.save()
                    logger.success("Saving best model")
                    trainer.best_model_acc = batch_accuracy

            with trainer.train_board_writer.as_default():
                tf.summary.scalar('loss', trainer.train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', trainer.train_accuracy.result(), step=epoch)

        logger.info(f'End of Epoch-{epoch + 1} Los: {trainer.train_loss.result():.4f}\tAccuracy: {trainer.train_accuracy.result():.4f}')

        epoch_time = time.time() - start
        logger.info(f'Time taken for 1 epoch: {epoch_time:.2f} secs\n')
        logger.success("Saving last epoch model")
        trainer.ckpt_manager_last_epoch.save()

if __name__ == '__main__':
    opts = Opts().parse()
    main(opts)


