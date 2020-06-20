import tensorflow as tf
import tensorflow.keras.layers as layers


class ChatbotModel:
    """Seq2Seq model using GRU cells.
    
    References:
    Tensorflow 2 tutorial: "Neural machine translation with attention"
    SuperDataScience course: "Deep Learning and NLP A-Z: How to Create a Chatbot"
    Lilian Weng's blog: "Attention? Attention!"
    """

    def __init__(self, hparams, vocab_int, filename):
        super(ChatbotModel, self).__init__()
        self.encoder = Encoder(len(vocab_int),
                               hparams['embedding_size'],
                               hparams['units'],
                               hparams['num_layers'],
                               dropout=hparams['keep_prob'])
        self.decoder = Decoder(len(vocab_int),
                               hparams['embedding_size'],
                               hparams['units'],
                               hparams['num_layers'],
                               dropout=hparams['keep_prob'])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hparams['learning_rate'])
        self.vocab_int = vocab_int
        self.filename = filename
        self.loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                      reduction='none')
        self.checkpoint = tf.train.Checkpoint(encoder=encoder, encoder=encoder, decoder=decoder)


    def train(self, inputs, targets, epochs, batch_size):
        padded_batch_inputs = tf.keras.preprocessing.sequence.pad_sequences(batch_inputs, padding='post')
        pass


    def validate(self, inputs, targets):
        pass


    def predict(self, inputs, max_output_len):
        pass


    def _load_model(self):
        self.save_dir
        pass


    def _save_model(self):
        self.save_dir
        pass
    
    
    def _loss_fn(targets, predictions):
        loss = self.loss_obj(targets, predictions)
        # mask = tf.math.logical_not(tf.math.equal(targets, 0))
        # mask = tf.cast(mask, dtype=loss.dtype)
        # loss *= mask
        return tf.math.reduce_mean(loss)


class Encoder(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, units, num_layers, dropout=0.):
        super(Encoder, self).__init__()
        self.units = units
        self.embedding = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        gru_cells = [layers.GRUCell(units,
                                    recurrent_initializer='glorot_uniform',
                                    dropout=dropout)
                     for _ in range(num_layers)]
        self.gru = layers.Bidirectional(layers.RNN(gru_cells,
                                                   return_sequences=True,
                                                   return_state=True))


    def call(self, x, state):
        x = self.embedding(x)
        y, state = self.gru(x, initial_state=state)
        return y, state


class Decoder(tf.keras.Model):
    
    def __init__(self, vocab_size, embedding_dim, units, num_layers, dropout=0.):
        super(Decoder, self).__init__()
        self.units = units
        self.attention = BahdanauAttention(units)
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        gru_cells = [layers.GRUCell(units,
                                    recurrent_initializer='glorot_uniform',
                                    dropout=dropout)
                     for _ in range(num_layers)]
        self.gru = layers.Bidirectional(layers.RNN(gru_cells,
                                                   return_sequences=True,
                                                   return_state=True))
        self.fc = tf.keras.layers.Dense(vocab_size)


    def call(self, x, state, enc_outputs):
        # enc_output shape == (batch_size, max_len, state_size)
        context_vec, _ = self.attention(state, enc_outputs)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vec, 1), x], axis=-1)
        # x.shape == (batch_size, 1, embedding_dim + state_size)

        y, state = self.gru(x)
        y = tf.reshape(y, (-1, y.shape[2]))
        # y.shape == (batch_size, vocab)

        output = self.fc(y)
        return output, state


class BahdanauAttention(layers.Layer):

    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.hidden = layers.Dense(units)
        self.context = layers.Dense(units)
        self.score = tf.keras.layers.Dense(1)


    def call(self, hidden, enc_outputs):
        expanded_hidden = tf.expand_dims(hidden, 1)
        # expanded_hidden.shape == (batch_size, 1, hidden_size)
        # enc_outputs.shape == (batch_size, max_len, hidden_size)

        score = self.score(tf.nn.tanh(self.hidden(expanded_hidden) + self.context(enc_outputs)))
        # score.shape == (batch_size, max_len, 1)

        attn_weights = tf.nn.softmax(score, axis=1)
        # attn_weights shape == (batch_size, max_len, 1)

        context_vec = attn_weights * enc_outputs
        context_vec = tf.reduce_sum(context_vec, axis=1)
        # context_vec.shape == (batch_size, hidden_size)

        return context_vec, attn_weights
