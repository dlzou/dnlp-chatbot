import tensorflow as tf
import tensorflow.keras.layers as layers


class ChatbotModel:
    """Seq2Seq model using GRU cells.

    This implementation is inspired by the SuperDataScience tutorial "Deep Learning and NLP A-Z: How to create a 
    chatbot. The "Neural machine translation with attention" offical guide was also referenced.
    """

    def __init__(self, hparams, vocab_int, save_dir):
        super(ChatbotModel, self).__init__()
        self.vocab_int = vocab_int
        self.save_dir = save_dir
        pass


    def train_batch(self, inputs, targets):
        pass


    def validate_batch(self, inputs, targets):
        pass


    def predict_batch(self, inputs, max_output_len):
        pass


    def _build_model(self):
        pass


    def _load_model(self):
        self.save_dir
        pass


    def _save_model(self):
        self.save_dir
        pass


class Encoder(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, units, num_layers=1, dropout=0.):
        super(Encoder, self).__init__()
        self.units = units
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        gru_cells = [layers.GRUCell(units,
                                    recurrent_initializer='glorot_uniform',
                                    dropout=dropout)
                     for _ in range(num_layers)]
        self.gru = layers.Bidirectional(layers.RNN(gru_cells,
                                                   return_sequences=True,
                                                   return_state=True))


    def call(self, x, hidden):
        x = self.embedding(x)
        y, state = self.gru(x, initial_state=hidden)
        return y, state


class Decoder(tf.keras.Model):
    
    def __init__(self, vocab_size, embedding_dim, units, num_layers=1, dropout=0.):
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
        
        
    def call(self, x, hidden, enc_outputs):
        context_vec, _ = self.attention(hidden, enc_outputs)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vec, 1), x], axis=-1)
        y, state = self.gru(x)
        y = tf.reshape(y, (-1, y.shape[2]))
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
