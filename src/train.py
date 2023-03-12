from lib.dataset import Dataset
from lib.opts import Opts
from lib.utils import Model
import time

def main(opts):
    dataLoader = Dataset(opts)
    datasets = dataLoader.create_dataset()

    train_data = dataLoader.convert_to_batch(datasets['train']['tensor-data']['input'], datasets['train']['tensor-data']['target'], opts.batch_size)
    
    modelLoader = Model(opts)
    model = modelLoader.get_model(datasets['train']['tokenizer']['input'], datasets['train']['tokenizer']['target'])
    
    for epoch in range(opts.num_epochs):
        start = time.time()

        trainer.train_loss.reset_states()
        trainer.train_accuracy.reset_states()
        trainer.epoch = epoch

        for (batch, (inp, tar)) in enumerate(dataset_train):
            trainer.train_step(inp, tar)
        
            #ter, wer, chrf, bleu1, bleu4 = evaluate_model(input_text_train, target_text_train, inp_lang_train, targ_lang_train, process, trainer, is_valid=False)
            if batch % 100 == 0:
                logger.info(f'Epoch->{epoch + 1}\tBatch->{batch}\tLoss->{trainer.train_loss.result():.4f}\tAccuracy->{trainer.train_accuracy.result():.4f}')
                trainer.log_file.write(f'Epoch->{epoch + 1}\tBatch->{batch}\tLoss->{trainer.train_loss.result():.4f}\tAccuracy->{trainer.train_accuracy.result():.4f}\n')

            with trainer.train_summary_writer.as_default():
                tf.summary.scalar('loss', trainer.train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', trainer.train_accuracy.result(), step=epoch)

        if ((epoch+1) % validation_mod) == 0:
            logger.info("Starting model evaluation...")
            trainer.log_file.write("Starting model evaluation...\n")
            ter, wer, chrf, bleu1, bleu4, sentence_info, avg_time = evaluate_model(input_text_valid, target_text_valid, inp_lang_train, targ_lang_train, process, target_tensor_valid.shape[1], transformer, trainer, tensorboardSave='valid')

            with trainer.train_summary_writer.as_default():
                tf.summary.scalar('TER', ter, step=epoch)
                tf.summary.scalar('WER', wer, step=epoch)
                tf.summary.scalar('CHRF', chrf, step=epoch)
                tf.summary.scalar('BLEU1', bleu1, step=epoch)
                tf.summary.scalar('BLEU4', bleu4, step=epoch)

        #if (epoch + 1) % 5 == 0:
        #  ckpt_save_path = ckpt_manager.save()
        #  logger.info(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

        logger.info(f'End of Epoch-{epoch + 1} Los: {trainer.train_loss.result():.4f}\tAccuracy: {trainer.train_accuracy.result():.4f}')
        trainer.log_file.write(f'End of Epoch-{epoch + 1} Los: {trainer.train_loss.result():.4f}\tAccuracy: {trainer.train_accuracy.result():.4f}\n')

        epoch_time = time.time() - start
        logger.info(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
        trainer.log_file.write(f'Time taken for 1 epoch: {epoch_time:.2f} secs\n')


    ckpt_save_path = trainer.ckpt_manager.save()
    logger.info(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')
    trainer.log_file.write(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}\n')
    trainer.log_file.close()


if __name__ == '__main__':
    opts = Opts().parse()
    main(opts)


