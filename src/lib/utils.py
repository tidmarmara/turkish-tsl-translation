import yaml
from models.transformer import Transformer, create_masks
from models.losses import loss_function, accuracy_function
import tensorflow as tf
import os
from loguru import logger

class Model():
    def __init__(self, opts, dataset):
        self.opts = opts
        self.dataset = dataset

        with open(self.opts.model_config) as f:
            self.model_config = yaml.safe_load(f)

    def get_model(self, inp_tokenizer, targ_tokenizer):
        if self.opts.model_type.lower() == 'transformer':
            model = Transformer(num_layers=self.model_config['model-parameters']['n-layers'],
                                     d_model=self.model_config['model-parameters']['d-model'],
                                     dff=self.model_config['model-parameters']['dff'],
                                     num_heads=self.model_config['model-parameters']['n-heads'],
                                     pe_input=self.model_config['model-parameters']['input-pos-encoding'],
                                     pe_target=self.model_config['model-parameters']['target-pos-encoding'],
                                     rate=self.model_config['model-parameters']['dropout'],
                                     input_vocab_size=len(inp_tokenizer.word_index)+1,
                                     target_vocab_size=len(targ_tokenizer.word_index)+1)
        return model

    def load_model(self, model, checkpoint_path):
        learning_rate = CustomSchedule(self.model_config['model-parameters']['d-model'])
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        ckpt = tf.train.Checkpoint(transformer=model, optimizer=optimizer)

        # save_path = ckpt.save(checkpoint_path)
        ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))

        return model

    def predict_sentence(self, model, sentence, inp_lang_train, targ_lang_train, max_target_len):
        sentence = self.dataset.preprocess_sentence(sentence)

        sentence_vec = inp_lang_train.texts_to_sequences([sentence])
        sentence_vec = tf.convert_to_tensor(sentence_vec)
        encoder_input = sentence_vec

        output = tf.constant([[targ_lang_train.word_index[self.dataset.start_token]]], dtype=tf.int64)
        output = tf.convert_to_tensor(output)
        enc_check = True
        while True:
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)
            if output.shape[1] > max_target_len:
                break

            #logger.info(f"{targ_lang_train.sequences_to_texts([output.numpy()[0]])[0]}")
            predictions, attention_weights = model(encoder_input, output, False, enc_padding_mask, combined_mask, dec_padding_mask, enc_check)
            enc_check = False

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.argmax(predictions, axis=-1)

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

            # return the result if the predicted_id is equal to the end token
            if targ_lang_train.index_word[predicted_id.numpy()[0][0]] == self.dataset.end_token:
                break

        # output.shape (1, tokens)
        if self.opts.token_type == 'word':
            text = targ_lang_train.sequences_to_texts([output.numpy()[0]])[0]  # shape: ()
        elif self.opts.token_type == 'char':
            text = ''
            for value in output.numpy()[0]:
                if value != 0:
                    text += targ_lang_train.index_word[value] 

        return text, attention_weights

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

class Trainer():
    def __init__(self, model, model_loader, opts):
        self.model = model
        self.model_loader = model_loader
        self.opts = opts

        self.best_model_acc = 0

        logger.info("Creating the experiment save path...")
        self.exp_save_path = self.create_path()
        logger.info(f"Save path: {self.exp_save_path}")

        self.init_logger(os.path.join(self.exp_save_path, "train.log"))

        self.learning_rate = CustomSchedule(self.model_loader.model_config['model-parameters']['d-model'])
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none') 
        
        self.ckpt_manager_best, self.ckpt_manager_last, self.ckpt_manager_last_epoch  = self.init_checkpoint_manager(self.model, self.optimizer)

        self.train_board_writer, self.valid_board_writer = self.init_tensorboard()
        
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    def init_tensorboard(self):
        logs_root_path = os.path.join(self.exp_save_path, "logs", "gradient_tape")
        if not os.path.isdir(logs_root_path):
            os.makedirs(logs_root_path)

        train_log_dir = os.path.join(logs_root_path, 'train')
        valid_log_dir = os.path.join(logs_root_path, 'valid')

        logger.info(f"Train log: {train_log_dir}")
        logger.info(f"Valid log: {valid_log_dir}")
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

        return train_summary_writer, valid_summary_writer

    def init_checkpoint_manager(self, model, optimizer):
        self.ckpt_save_path_best = os.path.join(self.exp_save_path, 'ckpts', 'best')
        self.ckpt_save_path_last = os.path.join(self.exp_save_path, 'ckpts', 'last')
        self.ckpt_save_path_last_epoch = os.path.join(self.exp_save_path, 'ckpts', 'last-epoch')

        if not os.path.exists(self.ckpt_save_path_best):
            os.makedirs(self.ckpt_save_path_best)
        if not os.path.exists(self.ckpt_save_path_last):
            os.makedirs(self.ckpt_save_path_last)
        if not os.path.exists(self.ckpt_save_path_last_epoch):
            os.makedirs(self.ckpt_save_path_last_epoch)

        ckpt_best = tf.train.Checkpoint(transformer=model, optimizer=optimizer)
        ckpt_last = tf.train.Checkpoint(transformer=model, optimizer=optimizer)
        ckpt_last_epoch = tf.train.Checkpoint(transformer=model, optimizer=optimizer)

        ckpt_manager_best = tf.train.CheckpointManager(ckpt_best, self.ckpt_save_path_best, max_to_keep=1,  checkpoint_name='best')
        ckpt_manager_last = tf.train.CheckpointManager(ckpt_last, self.ckpt_save_path_last, max_to_keep=1, checkpoint_name='last')
        ckpt_manager_last_epoch = tf.train.CheckpointManager(ckpt_last_epoch, self.ckpt_save_path_last_epoch, max_to_keep=1, checkpoint_name='last-epoch')

        return ckpt_manager_best, ckpt_manager_last, ckpt_manager_last_epoch

    def init_logger(self, logger_name):
        logger.add(logger_name)

    def create_path(self):
        self.save_root_path = self.opts.root_path
        model_save_path = os.path.join(self.save_root_path, self.opts.model_type)
        
        if os.path.exists(model_save_path):
            self.exp_id = len(os.listdir(model_save_path)) + 1
        else:
            os.mkdir(os.path.join(self.save_root_path, self.opts.model_type))
            self.exp_id = 1
        exp_save_path = os.path.join(model_save_path, str(self.exp_id))
        return exp_save_path

    # The @tf.function trace-compiles train_step into a TF graph for faster
    # execution. The function specializes to the precise shape of the argument
    # tensors. To avoid re-tracing due to the variable sequence lengths or variable
    # batch sizes (the last batch is smaller), use input_signature to specify
    # more generic shapes.
    @tf.function
    def train_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = self.model(inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
            loss = loss_function(tar_real, predictions, self.loss_object)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(accuracy_function(tf.cast(tar_real, tf.int64), tf.cast(predictions, tf.int64)))