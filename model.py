import tensorflow as tf
import tensorflow.keras.layers as layers


class ChatbotModel:
    """seq2seq model using GRU cells.

    Format for hyperparameters:
    hparams = {
        'embedding_dim': x,
        'units': x,
        'n_layers': x,
        'dropout': x,
        'learn_rate': x
    }

    Format for vocabulary:
    vocab_int = {
        '<PAD>': 0,
        '<SOS>': 1,
        '<EOS>': 2,
        '<EX>': 3,
        ...
    }

    References:
    Tensorflow 2 tutorial: "Neural machine translation with attention"
    SuperDataScience course: "Deep Learning and NLP A-Z: How to Create a Chatbot"
    Lilian Weng's blog: "Attention? Attention!"
    """

    def __init__(self, hparams, vocab_int, save_path):
        super(ChatbotModel, self).__init__()
        self.hparams = hparams
        self.encoder = Encoder(len(vocab_int),
                               hparams['embedding_dim'],
                               hparams['units'],
                               hparams['n_layers'],
                               dropout=hparams['dropout'])
        self.decoder = Decoder(len(vocab_int),
                               hparams['embedding_dim'],
                               hparams['units'],
                               hparams['n_layers'],
                               dropout=hparams['dropout'])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hparams['learn_rate']) # unused
        self.vocab_int = vocab_int
        self.save_path = save_path
        self.loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                      reduction='none')
        self.checkpoint = tf.train.Checkpoint(encoder=self.encoder,
                                              decoder=self.decoder,
                                              optimizer=self.optimizer)


    def __call__(self, batch_inputs, batch_targets, train=False):
        """Call interface to replace train_batch and test_batch.
        batch_inputs and batch_targets are zero-padded arrays.
        """
        assert len(batch_inputs) == len(batch_targets), 'batch_size is not consistent'

        batch_size = len(batch_inputs)
        hidden_state = self.encoder.get_init_state(batch_size)
        enc_outputs, hidden_state = self.encoder(batch_inputs, hidden_state)
        dec_inputs = tf.fill([batch_size, 1], self.vocab_int['<SOS>'])
        predicted_outputs = tf.fill([batch_size, 1], self.vocab_int['<SOS>'])
        grad_loss = 0

        if train:
            # teacher forcing: feeding the target (instead of prediction) as the next input
            for t in range(1, batch_targets.shape[1]):
                predictions, _ = self.decoder(dec_inputs, hidden_state, enc_outputs)
                targets = batch_targets[:, t]
                grad_loss += self.loss_fn(targets, predictions)

                predicted_ids = tf.math.argmax(predictions, axis=1, output_type=tf.dtypes.int32)
                predicted_ids = tf.expand_dims(predicted_ids, 1)
                predicted_outputs = tf.concat([predicted_outputs, predicted_ids], axis=-1)
                dec_inputs = tf.expand_dims(targets, 1)
        else:
            for t in range(1, batch_targets.shape[1]):
                predictions, _ = self.decoder(dec_inputs, hidden_state, enc_outputs)
                grad_loss += self.loss_fn(batch_targets[:, t], predictions)

                predicted_ids = tf.math.argmax(predictions, axis=1, output_type=tf.dtypes.int32)
                predicted_ids = tf.expand_dims(predicted_ids, 1)
                predicted_outputs = tf.concat([predicted_outputs, predicted_ids], axis=-1)
                dec_inputs = predicted_ids

        # batch_loss = grad_loss / int(batch_targets.shape[1])
        return predicted_outputs, grad_loss


    def get_variables(self):
        """Collect all trainable variables in the model."""
        return self.encoder.trainable_variables + self.decoder.trainable_variables


    def train_batch(self, batch_inputs, batch_targets):
        """Used for mini-batch training.
        Kept for reference; use call interface instead.
        """
        assert len(batch_inputs) == len(batch_targets), 'batch_size is not consistent'

        batch_inputs = tf.keras.preprocessing.sequence.pad_sequences(batch_inputs, padding='post')
        batch_targets = tf.keras.preprocessing.sequence.pad_sequences(batch_targets, padding='post')
        batch_size = batch_inputs.shape[0]
        loss = 0

        with tf.GradientTape() as tape:
            hidden_state = self.encoder.get_init_state(batch_size)
            enc_outputs, hidden_state = self.encoder(batch_inputs, hidden_state)
            dec_inputs = tf.expand_dims([self.vocab_int['<SOS>']] * batch_size, 1)

            # teacher forcing: feeding the target (instead of prediction) as the next input
            for t in range(1, batch_targets.shape[1]):
                predictions, _ = self.decoder(dec_inputs, hidden_state, enc_outputs)
                targets = batch_targets[:, t]
                loss += self.loss_fn(targets, predictions)
                dec_inputs = tf.expand_dims(targets, 1)

        batch_loss = loss / int(batch_targets.shape[1])
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss


    def test_batch(self, batch_inputs, batch_targets):
        """Similar to train_batch, but no gradient descent.
        Kept for reference; use call interface instead.
        """
        assert len(batch_inputs) == len(batch_targets), 'batch_size is not consistent'

        batch_inputs = tf.keras.preprocessing.sequence.pad_sequences(batch_inputs, padding='post')
        batch_targets = tf.keras.preprocessing.sequence.pad_sequences(batch_targets, padding='post')
        batch_size = batch_inputs.shape[0]
        loss = 0

        hidden_state = self.encoder.get_init_state(batch_size)
        enc_outputs, hidden_state = self.encoder(batch_inputs, hidden_state)
        dec_inputs = tf.expand_dims([self.vocab_int['<SOS>']] * batch_size, 1)

        for t in range(1, batch_targets.shape[1]):
            predictions, _ = self.decoder(dec_inputs, hidden_state, enc_outputs)
            loss += self.loss_fn(batch_targets[:, t], predictions)
            predicted_ids = tf.argmax(predictions, axis=1).numpy()
            dec_inputs = tf.expand_dims(predicted_ids, 1)

        batch_loss = loss / int(batch_targets.shape[1])
        return batch_loss


    def evaluate(self, input_seq, max_output_len):
        input_seq = tf.convert_to_tensor(input_seq)
        hidden_state = [tf.zeros((1, self.encoder.units))]
        enc_output, hidden_state = self.encoder(input_seq, hidden_state)
        dec_input = tf.expand_dims([self.vocab_int['<SOS>']], 0)
        output_seq = []

        for t in range(max_output_len):
            predictions, hidden_state = self.decoder(dec_input, hidden_state, enc_output)
            predicted_id = tf.argmax(predictions[0]).numpy()
            output_seq.append(predicted_id)
            if predicted_id == self.vocab_int['<EOS>']:
                return output_seq
            dec_input = tf.expand_dims([predicted_id], 0)

        return output_seq


    def save(self):
        self.checkpoint.save(self.save_path)


    def restore(self):
        status = self.checkpoint.restore(tf.train.latest_checkpoint(self.save_path))
        status.assert_comsumed()


    def loss_fn(self, targets, predictions):
        loss = self.loss_obj(targets, predictions)
        mask = tf.math.logical_not(tf.math.equal(targets, 0))
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.math.reduce_mean(loss)


class Encoder(tf.keras.Model):
    """Encoder portion of the seq2seq model."""

    def __init__(self, vocab_size, embedding_dim, units, n_layers, dropout=0.):
        super(Encoder, self).__init__()
        self.units = units
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.embedding = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        gru_cells = [layers.GRUCell(units,
                                    recurrent_initializer='glorot_uniform',
                                    dropout=dropout)
                     for _ in range(n_layers)]
        self.gru = layers.Bidirectional(layers.RNN(gru_cells,
                                                   return_sequences=True,
                                                   return_state=True))


    def call(self, inputs, state):
        # inputs are in mini-batches
        inputs = self.embedding(inputs)
        output_tuple = self.gru(inputs, initial_state=state)

        # not sure, pls improve docs google
        outputs = output_tuple[0]
        forward_state = output_tuple[len(output_tuple) // 2]
        reverse_state = output_tuple[-1]

        return outputs, tf.concat([forward_state, reverse_state], axis=-1)


    def get_init_state(self, batch_size):
        # doubled because bidirectional
        return [tf.zeros((batch_size, self.units)) for i in range(2 * self.n_layers)]


class Decoder(tf.keras.Model):
    """Decoder portion of the seq2seq model."""

    def __init__(self, vocab_size, embedding_dim, units, n_layers, dropout=0.):
        super(Decoder, self).__init__()
        self.units = units
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.attention = BahdanauAttention(units)
        gru_cells = [layers.GRUCell(units,
                                    recurrent_initializer='glorot_uniform',
                                    dropout=dropout)
                     for _ in range(n_layers)]
        self.gru = layers.Bidirectional(layers.RNN(gru_cells,
                                                   return_sequences=True,
                                                   return_state=True))
        self.predictor = tf.keras.layers.Dense(vocab_size)


    def call(self, inputs, state, enc_outputs):
        # enc_outputs shape == (batch_size, max_len, state_size)
        context_vec, _ = self.attention(state, enc_outputs)
        inputs = self.embedding(inputs)
        inputs = tf.concat([tf.expand_dims(context_vec, 1), inputs], axis=-1)
        # x.shape == (batch_size, 1, embedding_dim + state_size)

        output_tuple = self.gru(inputs)
        outputs = output_tuple[0]
        forward_state = output_tuple[len(output_tuple) // 2]
        reverse_state = output_tuple[-1]

        outputs = tf.reshape(outputs, (-1, outputs.shape[2]))
        predictions = self.predictor(outputs)
        # predictions.shape == (batch_size, vocab_size)

        return predictions, tf.concat([forward_state, reverse_state], axis=-1)


class BahdanauAttention(layers.Layer):
    """Implementation of the Bahdanau attention mechanism.

    An instance is maintained by the decoder.
    """

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
