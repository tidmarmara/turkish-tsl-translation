import tensorflow as tf
from loguru import logger 
from models.losses import loss_function

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, n_layers, layer_type):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.layer_type = layer_type
        self.n_layers = n_layers
        
        logger.info("Initializing the Encoder...")
        
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        
        self.encoder_layers = [0]*self.n_layers
        for i in range(self.n_layers):
            name = f"encoder_{i+1}"
            if self.layer_type.lower() == "bgru":
                self.encoder_layers[i] = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform', name=f"encoder_bgru_{i}"))
            elif self.layer_type.lower() == "gru":
                self.encoder_layers[i] = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform', name=f"encoder_gru_{i}")
            elif self.layer_type.lower() == "lstm":
                logger.info("CREATING LSTM LAYER")
                self.encoder_layers[i] = tf.keras.layers.LSTM(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform', name=f"encoder_lstm_{i}")
            elif self.layer_type.lower() == "blstm":
                self.encoder_layers[i] = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform', name=f"encoder_blstm_{i}"))
            else:
                logger.info("Wrong layer type! Correct input might be LSTM, BLSTM, GRU or BGRU")
                break
                            
    def call(self, x, hidden):
        x = self.embedding(x)
        for i in range(self.n_layers):
            if self.layer_type.lower() == "gru":
                if i == 0:
                    encoder_outputs, state_h = self.encoder_layers[i](x, initial_state=hidden)
                else:
                    encoder_outputs, state_h = self.encoder_layers[i](encoder_outputs)
            elif self.layer_type.lower() == "bgru":
                if i == 0:
                    encoder_outputs, forward_h, backward_h = self.encoder_layers[i](x, initial_state=hidden)
                else:
                    encoder_outputs, forward_h, backward_h = self.encoder_layers[i](encoder_outputs)
                state_h = tf.keras.layers.concatenate([forward_h, backward_h], axis=-1, name="concat_0")
            elif self.layer_type.lower() == "blstm":
                if i == 0:
                    encoder_outputs, forward_h, forward_c, backward_h, backward_c = self.encoder_layers[i](x, initial_state=hidden)
                else:
                    encoder_outputs, forward_h, forward_c, backward_h, backward_c = self.encoder_layers[i](encoder_outputs)
                state_h = tf.keras.layers.concatenate([forward_h, backward_h], axis=-1, name="concat_0")
                state_c = tf.keras.layers.concatenate([forward_c, backward_c], axis=-1, name="concat_0")
            elif self.layer_type.lower() == "lstm":
                if i == 0:
                    encoder_outputs, state_h, state_c = self.encoder_layers[i](x, initial_state=hidden)
                else:
                    encoder_outputs, state_h, state_c = self.encoder_layers[i](encoder_outputs)
                
        if self.layer_type.lower() == "lstm" or self.layer_type.lower() == "blstm":
            return encoder_outputs, state_h, state_c
        elif self.layer_type.lower() == "gru" or self.layer_type.lower() == "bgru":
            return encoder_outputs, state_h

    def initialize_hidden_state(self):
        if self.layer_type.lower() == "bgru":
            return [tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))]
        elif self.layer_type.lower() == "gru":
            return tf.zeros((self.batch_sz, self.enc_units))
        elif self.layer_type.lower() == "blstm":
            return [tf.zeros((self.batch_sz, self.enc_units)) for i in range(4)]
        elif self.layer_type.lower() == "lstm":
            return [tf.zeros((self.batch_sz, self.enc_units)) for i in range(2)]


class LuongAttention(tf.keras.Model):
    def __init__(self, rnn_size, layer_type):
        super(LuongAttention, self).__init__()
        if layer_type.lower() == "blstm" or layer_type.lower() == "bgru":
            self.wa = tf.keras.layers.Dense(rnn_size*2)
        elif layer_type.lower() == "lstm" or layer_type.lower() == "gru":
            self.wa = tf.keras.layers.Dense(rnn_size)
        
    def call(self, decoder_output, encoder_output):
        # Dot score: h_t (dot) Wa (dot) h_s
        # encoder_output shape: (batch_size, max_len, rnn_size)
        # decoder_output shape: (batch_size, 1, rnn_size)
        # score will have shape: (batch_size, 1, max_len)
        score = tf.matmul(decoder_output, self.wa(encoder_output), transpose_b=True)
        # alignment vector a_t
        alignment = tf.nn.softmax(score, axis=2)
        # context vector c_t is the average sum of encoder output
        context = tf.matmul(alignment, encoder_output)

        return context, alignment

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, n_layers, layer_type, attention_type):
        super(Decoder, self).__init__()
        logger.info("Initializing the Decoder...")
        
        self.n_layers = n_layers
        self.layer_type = layer_type
        self.attention_type = attention_type
        
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        
        # DEFINE EMBEDDING
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        
        # DEFINE LAYERS
        self.decoder_layers = [0]*self.n_layers
        for i in range(self.n_layers):
            if self.layer_type.lower() == "bgru":
                self.decoder_layers[i] = tf.keras.layers.GRU(self.dec_units*2, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform', name=f"decoder_gru_{i}")
                logger.info("Creating BGRU layer...")
            elif self.layer_type.lower() == "gru":
                self.decoder_layers[i] = tf.keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform', name=f"decoder_gru_{i}")
                logger.info("Creating GRU layer...")
            elif self.layer_type.lower() == "lstm":
                self.decoder_layers[i] = tf.keras.layers.LSTM(self.dec_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform', name=f"decoder_lstm_{i}")
                logger.info("Creating LSTM layer...")
            elif self.layer_type.lower() == "blstm":
                self.decoder_layers[i] = tf.keras.layers.LSTM(self.dec_units*2, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform', name=f"decoder_lstm_{i}")
                logger.info("Creating BLSTM layer...")
            else:
                logger.info("Wrong layer type!")
                break

        
        self.fc = tf.keras.layers.Dense(vocab_size)
        
        # used for attention
        if self.attention_type.lower() == "bahdanau":
            self.attention = BahdanauAttention(self.dec_units)
            logger.info("Creating Bahdanau attention...")
        elif self.attention_type.lower() == "luong":
            self.attention = LuongAttention(self.dec_units, self.layer_type)
            self.wc = tf.keras.layers.Dense(self.dec_units, activation='tanh')
            logger.info("Creating Luong attention...")
        else:
            logger.info(f"Chosen Attention Type: '{self.attention_type}'")

    def call(self, x, hidden, enc_output):
        if self.attention_type.lower() == "bahdanau":
            # enc_output shape == (batch_size, max_length, hidden_size)
            if self.layer_type.lower() == "gru" or self.layer_type.lower() == "bgru":
                context_vector, attention_weights = self.attention(hidden, enc_output)
            if self.layer_type.lower() == "lstm" or self.layer_type.lower() == "blstm":
                context_vector, attention_weights = self.attention(hidden[0], enc_output)

            # x shape after passing through embedding == (batch_size, 1, embedding_dim)
            x = self.embedding(x)

            # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
            x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

            # passing the concatenated vector to the GRU
            for i in range(self.n_layers):
                if self.layer_type.lower() == "gru" or self.layer_type.lower() == "bgru":
                    if i == 0:
                        output, state_h = self.decoder_layers[i](x, initial_state=hidden)
                    else:
                        output, state_h = self.decoder_layers[i](output, initial_state=hidden)
                elif self.layer_type.lower() == "lstm" or self.layer_type.lower() == "blstm":
                    if i == 0:
                        output, state_h, state_c = self.decoder_layers[i](x, initial_state=hidden)
                    else:
                        output, state_h, state_c = self.decoder_layers[i](output, initial_state=hidden)

            # output shape == (batch_size * 1, hidden_size)
            output = tf.reshape(output, (-1, output.shape[2]))

            # output shape == (batch_size, vocab)
            x = self.fc(output)
            
            if self.layer_type.lower() == "lstm" or self.layer_type.lower() == "blstm":
                return x, state_h, state_c, attention_weights
            elif self.layer_type.lower() == "gru" or self.layer_type.lower() == "bgru":
                return x, state_h, attention_weights
        
        elif self.attention_type.lower() == "luong":
            # Remember that the input to the decoder
            # is now a batch of one-word sequences,
            # which means that its shape is (batch_size, 1)
            x = self.embedding(x)

            # Therefore, the lstm_out has shape (batch_size, 1, rnn_size)
            for i in range(self.n_layers):
                if self.layer_type.lower() == "gru" or self.layer_type.lower() == "bgru":
                    if i == 0:
                        output, state_h = self.decoder_layers[i](x, initial_state=hidden)
                    else:
                        output, state_h = self.decoder_layers[i](output)
                elif self.layer_type.lower() == "lstm" or self.layer_type.lower() == "blstm":
                    if i == 0:
                        output, state_h, state_c = self.decoder_layers[i](x, initial_state=hidden)
                    else:
                        output, state_h, state_c = self.decoder_layers[i](output, initial_state=hidden)
            
            # Use self.attention to compute the context and alignment vectors
            # context vector's shape: (batch_size, 1, rnn_size)
            # alignment vector's shape: (batch_size, 1, source_length)

            context, alignment = self.attention(output, enc_output)

            # Combine the context vector and the LSTM output
            # Before combined, both have shape of (batch_size, 1, rnn_size),
            # so let's squeeze the axis 1 first
            # After combined, it will have shape of (batch_size, 2 * rnn_size)
            output = tf.concat([tf.squeeze(context, 1), tf.squeeze(output, 1)], 1)

            # lstm_out now has shape (batch_size, rnn_size)
            output = self.wc(output)

            # Finally, it is converted back to vocabulary space: (batch_size, vocab_size)
            logits = self.fc(output)
            
            if self.layer_type.lower() == "lstm" or self.layer_type.lower() == "blstm":
                return logits, state_h, state_c, alignment
            elif self.layer_type.lower() == "gru" or self.layer_type.lower() == "bgru":
                return logits, state_h, alignment
        
        else:
            x = self.embedding(x)

            # passing the concatenated vector to the GRU
            for i in range(self.n_layers):
                if self.layer_type.lower() == "gru" or self.layer_type.lower() == "bgru":
                    if i == 0:
                        output, state_h = self.decoder_layers[i](x, initial_state=hidden)
                    else:
                        output, state_h = self.decoder_layers[i](output, initial_state=hidden)
                elif self.layer_type.lower() == "lstm" or self.layer_type.lower() == "blstm":
                    if i == 0:
                        output, state_h, state_c = self.decoder_layers[i](x, initial_state=hidden)
                    else:
                        output, state_h, state_c = self.decoder_layers[i](output, initial_state=hidden)

            # output shape == (batch_size * 1, hidden_size)
            output = tf.reshape(output, (-1, output.shape[2]))

            # output shape == (batch_size, vocab)
            x = self.fc(output)
            
            if self.layer_type.lower() == "lstm" or self.layer_type.lower() == "blstm":
                return x, state_h, state_c
            elif self.layer_type.lower() == "gru" or self.layer_type.lower() == "bgru":
                return x, state_h

class RNN_Based(tf.keras.Model):
    def __init__(self, units, n_layers, embedding_dim, layer_type, attention_type, batch_size, input_tokenizer, target_tokenizer, dataset):
        super(RNN_Based, self).__init__()
        self.layer_type = layer_type
        self.dataset = dataset
        self.batch_size = batch_size
        self.inp_tokenizer = input_tokenizer
        self.tar_tokenizer = target_tokenizer
        self.attention_type = attention_type
        self.units = units

        self.encoder = Encoder(vocab_size=len(self.inp_tokenizer.word_index)+1, 
                  embedding_dim=embedding_dim, 
                  enc_units=units, 
                  batch_sz=batch_size, 
                  n_layers=n_layers, 
                  layer_type=layer_type)

        self.decoder = Decoder(vocab_size=len(self.tar_tokenizer.word_index)+1, 
                        embedding_dim=embedding_dim, 
                        dec_units=units, 
                        batch_sz=batch_size, 
                        n_layers=n_layers, 
                        layer_type=layer_type, 
                        attention_type=attention_type)
    
    def call(self, inp, targ):
        enc_hidden = self.encoder.initialize_hidden_state()

        if self.layer_type.lower() == "lstm" or self.layer_type.lower() == "blstm":
            enc_output, enc_hidden_h, enc_hidden_c = self.encoder(inp, enc_hidden)
            dec_hidden = [enc_hidden_h, enc_hidden_c]
        elif self.layer_type.lower() == "gru" or self.layer_type.lower() == "bgru":
            enc_output, enc_hidden_h = self.encoder(inp, enc_hidden)
            dec_hidden = enc_hidden_h
        
        # First input is given manually to the model which is the start_token
        dec_input = tf.expand_dims([self.tar_tokenizer.word_index[self.dataset.start_token]] * self.batch_size, 1)

        # Save all the model outputs in a list
        outputs = []

        # Teacher forcing - feeding the target as the next input
        # Here, we start from 1st index token due to feeding the start_token manually in the previous line
        for t in range(1, targ.shape[1]):
            # print("Token: ", t)
            # print("TARG: ", targ.shape)
            # passing enc_output to the decoder
            if (self.attention_type.lower() == "luong") | (self.attention_type.lower() == "bahdanau"):
                if self.layer_type.lower() == "gru" or self.layer_type.lower() == "bgru":
                    predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
                elif self.layer_type.lower() == "lstm" or self.layer_type.lower() == "blstm":
                    predictions, dec_hidden_h, dec_hidden_c, _ = self.decoder(dec_input, dec_hidden, enc_output)
                    dec_hidden = [dec_hidden_h, dec_hidden_c]
            else:
                if self.layer_type.lower() == "gru" or self.layer_type.lower() == "bgru":
                    predictions, dec_hidden = self.decoder(dec_input, dec_hidden, enc_output)
                elif self.layer_type.lower() == "lstm" or self.layer_type.lower() == "blstm":
                    predictions, dec_hidden_h, dec_hidden_c = self.decoder(dec_input, dec_hidden, enc_output)
                    dec_hidden = [dec_hidden_h, dec_hidden_c]
            
            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)
            outputs.append(predictions)
            # print("predictions: ", predictions.shape)
        
        # Change the order of tensor
        outputs = tf.convert_to_tensor(outputs, dtype=tf.float32)
        outputs = tf.transpose(outputs, perm=[1, 0, 2])

        return outputs