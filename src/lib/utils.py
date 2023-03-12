import yaml
from lib.models.transformer import Transformer

class Model():
    def __init__(self, opts):
        self.opts = opts

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

class Trainer():
    def __init__(self, model, input_tensor_train, target_tensor_train, 
                inp_lang_train, targ_lang_train, num_layers, d_model,
                batch_size, dff, dropout_rate,
                epochs, num_heads, model_type):
        self.model = model
        self.epoch = 0
        self.model_type = model_type

        learning_rate = CustomSchedule(d_model)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none') 

        RESULTS_PATH = "results"
        WEIGHTS_ROOT_PATH = "transformer"
        self.exp_name = f"exp_batch_size-{batch_size}_nlayers-{num_layers}_dmodel-{d_model}_nheads-{num_heads}_dff-{dff}_drop-{dropout_rate}"
        self.CHECKPOINTS_PATH = os.path.join(RESULTS_PATH, WEIGHTS_ROOT_PATH, self.exp_name)

        if not os.path.isdir(self.CHECKPOINTS_PATH):
            os.makedirs(self.CHECKPOINTS_PATH)
        
        train_log_name = "train_log.txt"
        self.log_file = open(os.path.join(self.CHECKPOINTS_PATH, train_log_name), 'w')
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_file.write(f"({current_time})-Log file is initialized!\n")
        
        ckpt = tf.train.Checkpoint(transformer=self.model, optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(ckpt, self.CHECKPOINTS_PATH, max_to_keep=5)

        logs_root_path = os.path.join(self.CHECKPOINTS_PATH, "logs", "gradient_tape")
        if not os.path.isdir(logs_root_path):
            os.makedirs(logs_root_path)

        # Save the training configurations to a config file
        save_configuration = open(os.path.join(self.CHECKPOINTS_PATH, 'config.cfg'), 'w', encoding='utf-8')
        save_configuration.write("MAX_IN_LEN: " + str(input_tensor_train.shape[1]) + '\n' + \
                                "MAX_TAR_LEN: " + str(target_tensor_train.shape[1]) + '\n' + \
                                "IN_VOCAB_SIZE: " + str(len(inp_lang_train.word_index)) + '\n' + \
                                "TAR_VOCAB_SIZE: " + str(len(targ_lang_train.word_index)) + '\n' + \
                                "N_LAYERS: " + str(num_layers) + '\n' + \
                                "D_MODEL: " + str(d_model) + '\n' + \
                                "BATCH_SIZE: " + str(batch_size) + '\n' + \
                                "DFF: " + str(dff) + '\n' + \
                                "DROPOUT: " + str(dropout_rate) + '\n' + \
                                "EPOCHS: " + str(epochs) + '\n' + \
                                "MODEL_TYPE: " + str(self.model_type) + '\n' + \
                                "NUM_HEADS: " + str(num_heads))
        save_configuration.close()

        train_log_dir = os.path.join(logs_root_path, 'train')
        valid_log_dir = os.path.join(logs_root_path, 'valid')

        logger.info(f"Train log: {train_log_dir}")
        logger.info(f"Valid log: {valid_log_dir}")
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)
        
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

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