import tensorflow as tf
import tensorflow.keras.layers as layers
from numpy import log


# https://github.com/tensorflow/tensorflow/issues/36508
# gpus = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], enable=True)


class ChatbotModel(tf.Module):
    """seq2seq model using GRU cells.

    Format for hyperparameters:
    hparams = {
        'embedding_dim': x,
        'units': x,
        'n_layers': x,
        'dropout': x
    }

    Format for vocabulary:
    vocab_index = {
        '<PAD>': 0,
        '<SOS>': 1,
        '<EOS>': 2,
        '<EX>': 3,
        ...
    }

    References:
    TensorFlow 2 tutorial: "Neural machine translation with attention"
    SuperDataScience course: "Deep Learning and NLP A-Z: How to Create a Chatbot"
    Lilian Weng's blog: "Attention? Attention!"
    """

    def __init__(self, hparams, vocab_index, save_path, name=None):
        super(ChatbotModel, self).__init__(name=name)
        self.encoder = Encoder(len(vocab_index),
                               hparams['embedding_dim'],
                               hparams['units'],
                               dropout=hparams['dropout'])
        self.decoder = Decoder(len(vocab_index),
                               hparams['embedding_dim'],
                               hparams['units'],
                               dropout=hparams['dropout'])
        self.beam_width = hparams['beam_width']
        self.vocab_index = vocab_index
        self.save_path = save_path
        self.loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                      reduction='none')
        self.checkpoint = tf.train.Checkpoint(encoder=self.encoder,
                                              decoder=self.decoder)


    def __call__(self, batch_inputs, batch_targets, targets_shape, train=False):
        """
        Call interface to replace train_batch and test_batch.
        batch_inputs and batch_targets are zero-padded arrays.
        """
        # assert batch_inputs.shape[0] == batch_targets.shape[0], 'batch_size not consistent'

        # only prints once when traced by tf.function to compile a new graph
        print("tracing __call__: ", batch_inputs.shape, batch_targets.shape, targets_shape)

        batch_size = targets_shape[0]
        init_state = self.encoder.get_init_state(batch_size)
        enc_output, dec_state = self.encoder(batch_inputs, init_state)
        dec_input = tf.fill([batch_size, 1], self.vocab_index['<SOS>'])
        grad_loss = 0.

        if train:
            # teacher forcing: feeding the target (instead of prediction) as the next input
            for t in range(1, targets_shape[1]):
                predictions, dec_state = self.decoder(dec_input, dec_state, enc_output)
                targets = batch_targets[:, t]
                grad_loss += self.loss_fn(targets, predictions)

                dec_input = tf.expand_dims(targets, 1)

        else:
            for t in range(1, targets_shape[1]):
                predictions, dec_state = self.decoder(dec_input, dec_state, enc_output)
                grad_loss += self.loss_fn(batch_targets[:, t], predictions)

                predicted_ids = tf.math.argmax(predictions, axis=1, output_type=tf.dtypes.int32)
                predicted_ids = tf.expand_dims(predicted_ids, 1)
                dec_input = predicted_ids

        # batch_loss = grad_loss / int(batch_targets.shape[1])
        return grad_loss


    def evaluate(self, input_seq, max_target_len):
        """
        Use beam search to decode inputs at runtime. Ideally the model should be trained with beam
        search as well, but ¯\_(ツ)_/¯
        """
        input_seq = tf.expand_dims(tf.convert_to_tensor(input_seq), 0)
        init_state = self.encoder.get_init_state(1)
        enc_output, dec_state = self.encoder(input_seq, init_state)

        # first cycle
        dec_input = tf.expand_dims([self.vocab_index['<SOS>']], 1)
        predictions, dec_state = self.decoder(dec_input, dec_state, enc_output)
        pred_vals, indices = tf.math.top_k(predictions[0], k=self.beam_width)
        pred_vals = (-tf.math.log(pred_vals)).numpy().tolist() # log to prevent underflow

        dec_input = tf.expand_dims(indices, 1)
        dec_state = tf.tile(dec_state, [self.beam_width, 1])
        enc_output = tf.tile(enc_output, [self.beam_width, 1, 1])
        sequences = [[indices[b]] for b in range(self.beam_width)]

        # beam search, taking advantage of batch inputs
        # 0.5 chance this actually works as intended
        # is there a more efficient way to do this, e.g. without Python lists?
        for t in range(max_target_len):
            predictions, dec_state = self.decoder(dec_input, dec_state, enc_output)
            predictions = predictions.numpy()
            candidates = []
            for b in range(self.beam_width):
                predictions[b] = pred_vals[b] - log(predictions[b])
                _pred_vals, _indices = tf.math.top_k(-predictions[b], k=self.beam_width)
                for bb in range(self.beam_width):
                    candidates.append([sequences[bb] + [_indices[bb]], -_pred_vals[bb], b])

            candidates.sort(reverse=True, key=lambda c: c[1]) # sort by pred_vals
            sequences, pred_vals, indices = zip(*candidates[:self.beam_width])
            if sequences[0][-1] == self.vocab_index['<EOS>']:
                return sequences[0]

            dec_input = tf.convert_to_tensor([[s[-1]] for s in sequences])
            dec_state = tf.stack([dec_state[i] for i in indices])

        return sequences[0]

        # greedy search
        # sequence = []
        # for t in range(max_target_len):
        #     predictions, dec_state = self.decoder(dec_input, dec_state, enc_output)
        #     predicted_id = tf.argmax(predictions[0]).numpy()
        #     sequence.append(predicted_id)
        #     if predicted_id == self.vocab_index['<EOS>']:
        #         return sequences
        #     dec_input = tf.expand_dims([predicted_id], 1)
        # return sequence


    def loss_fn(self, targets, predictions):
        loss = self.loss_obj(targets, predictions)
        mask = tf.math.logical_not(tf.math.equal(targets, 0))
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.math.reduce_mean(loss)


    def get_train_vars(self):
        return self.encoder.trainable_variables + self.decoder.trainable_variables


    def save(self):
        self.checkpoint.save(self.save_path)


    def restore(self):
        status = self.checkpoint.restore(tf.train.latest_checkpoint(self.save_path))
        # status.assert_comsumed()



class Encoder(tf.keras.Model):
    """Encoder portion of the seq2seq model."""

    def __init__(self, vocab_size, embedding_dim, units, dropout=0.):
        super(Encoder, self).__init__()
        self.units = units
        self.embedding = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        # self.n_layers = n_layers
        # gru_cells = [layers.GRUCell(units,
        #                             recurrent_initializer='glorot_uniform',
        #                             dropout=dropout)
        #              for _ in range(n_layers)]
        # self.gru = layers.Bidirectional(layers.RNN(gru_cells,
        #                                            return_sequences=True,
        #                                            return_state=True))
        self.gru = layers.Bidirectional(layers.GRU(self.units,
                                                   return_sequences=True,
                                                   return_state=True,
                                                   recurrent_initializer='glorot_uniform'))


    def call(self, enc_inputs, init_state):
        # inputs are in mini-batches
        enc_inputs = self.embedding(enc_inputs)
        tup = self.gru(enc_inputs, initial_state=init_state)
        output = tup[0]
        # output shape is (batch_size, seq_len, 2 * units)

        state = tf.concat(tup[1:], axis=-1)

        # concat forward and reverse hidden states
        return output, state


    def get_init_state(self, batch_size):
        # doubled because bidirectional
        return [tf.zeros((batch_size, self.units)) for _ in range(2)]



class Decoder(tf.keras.Model):
    """Decoder portion of the seq2seq model."""

    def __init__(self, vocab_size, embedding_dim, units, dropout=0.):
        super(Decoder, self).__init__()
        self.units = units
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.attention = BahdanauAttention(units)
        self.predictor = layers.Dense(vocab_size, activation='softmax')
        # self.n_layers = n_layers
        # gru_cells = [layers.GRUCell(units,
        #                             recurrent_initializer='glorot_uniform',
        #                             dropout=dropout)
        #              for _ in range(n_layers)]
        # self.gru = layers.Bidirectional(layers.RNN(gru_cells,
        #                                            return_sequences=True,
        #                                            return_state=True))
        self.gru = layers.Bidirectional(layers.GRU(self.units,
                                                   return_sequences=True,
                                                   return_state=True,
                                                   recurrent_initializer='glorot_uniform'))


    def call(self, dec_input, dec_state, enc_output):
        # enc_output shape is (batch_size, max_len, states_size)
        context_vec, _ = self.attention(dec_state, enc_output)
        dec_input = self.embedding(dec_input)

        dec_input = tf.concat([tf.expand_dims(context_vec, 1), dec_input], axis=-1)
        # inputs shape is (batch_size, 1, embedding_dim + states_size)

        tup = self.gru(dec_input)

        output = tup[0]
        output = tf.reshape(output, (-1, output.shape[2]))
        predictions = self.predictor(output)
        # predictions shape is (batch_size, vocab_size)

        state = tf.concat(tup[1:], axis=-1)

        return predictions, state



class BahdanauAttention(layers.Layer):
    """
    Implementation of the Bahdanau attention mechanism.
    An instance is maintained by the decoder.
    """

    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.state = layers.Dense(units)
        self.context = layers.Dense(units)
        self.scoring = layers.Dense(1)


    def call(self, dec_state, enc_output):
        dec_state = tf.expand_dims(dec_state, 1)
        # expanded_states shape is (batch_size, 1, states_size)
        # enc_output shape is (batch_size, max_len, states_size)

        score = self.scoring(tf.nn.tanh(self.state(dec_state) + self.context(enc_output)))
        # score shape is (batch_size, max_len, 1)

        attn_weights = tf.nn.softmax(score, axis=1)
        # attn_weights shape is (batch_size, max_len, 1)

        context_vec = attn_weights * enc_output
        context_vec = tf.reduce_sum(context_vec, axis=1)
        # context_vec shape is (batch_size, states_size)

        return context_vec, attn_weights
