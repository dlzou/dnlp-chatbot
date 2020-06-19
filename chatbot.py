import numpy as np
import tensorflow as tf
import time
import utils

with open('cmdc/movie_lines.txt', encoding='utf-8', errors='ignore') as f:
    lines = f.read().split('\n')
with open('cmdc/movie_conversations.txt', encoding='utf-8', errors='ignore') as f:
    conversations = f.read().split('\n')


############ PREPROCESS DATA #############

# Extract file data
id_to_line = {}
for line in lines:
    line = line.split(" +++$+++ ")
    if len(line) == 5:
        id_to_line[line[0]] = line[-1]

conversation_ids = []
for conv in conversations[:-1]:
    conv = conv.split(" +++$+++ ")
    if len(conv) == 4:
        conv = conv[-1][1:-1].replace("'", "").replace(" ", "")
        conversation_ids.append(conv.split(","))

questions = []
answers = []
for cid in conversation_ids:
    for i in range(len(cid) - 1):
        questions.append(id_to_line[cid[i]])
        answers.append(id_to_line[cid[i + 1]])

# Clean non-alphanumeric characters
clean_questions = []
for q in questions:
    clean_questions.append(utils.clean_text(q))

clean_answers = []
for a in answers:
    clean_answers.append(utils.clean_text(a))

# Count word occurrences, filter out uncommon words
word_counts = {}
for line in clean_questions:
    for word in line.split():
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
for line in clean_answers:
    for word in line.split():
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1

min_count = 15
word_to_int = {}
word_id = 0
for word, count in word_counts.items():
    if count >= min_count:
        word_to_int[word] = word_id
        word_id += 1

# Add tokens
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for t in tokens:
    word_to_int[t] = len(word_to_int) + 1
int_to_word = {i : w for w, i in word_to_int.items()}

for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'

# Assign unique integers to words
out_int = word_to_int['<OUT>']
questions_int = []
max_q_len = 0
for line in clean_questions:
    line = [word_to_int.get(w, out_int) for w in line.split()]
    questions_int.append(line)
    if len(line) > max_q_len:
        max_q_len = len(line)

answers_int = []
for line in clean_answers:
    answers_int.append([word_to_int.get(w, out_int) for w in line.split()])

# Sort questions by length
sorted_clean_q = []
sorted_clean_a = []
for length in range(2, max_q_len + 1):
    for i, q in enumerate(questions_int):
        if len(q) == length:
            sorted_clean_q.append(questions_int[i])
            sorted_clean_a.append(answers_int[i])


############ BUILD MODEL ############

def model_inputs():
    inputs = tf.placeholder(tf.int32, shape=(None, None), name='inputs')
    targets = tf.placeholder(tf.int32, shape=(None, None), name='targets')
    learn_rate = tf.placeholder(tf.float32, name='learn_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return inputs, targets, learn_rate, keep_prob


def preprocess_targets(targets, word_to_int, batch_size):
    left = tf.fill((batch_size, 1), word_to_int['<SOS>'])
    # right = tf.strided_slice(targets, (0, 0), (batch_size, -1), strides=(1, 1))
    right = targets[:batch_size, :-1]
    return tf.concat((left, right), 1)


def encoder_rnn(inputs, size, num_layers, keep_prob, seq_len):
    lstm = tf.contrib.rnn.BasicLSTMCell(size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell((lstm_dropout,) * num_layers)
    # enc_output can be ignored
    enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                                            cell_bw=encoder_cell,
                                                            sequence_length=seq_len,
                                                            inputs=inputs,
                                                            dtype=tf.float32)
    return enc_state


def decode_train_set(encoder_state, decoder_cell, embedded_input, seq_len,
                     decode_scope, output_fn, keep_prob, batch_size):
    attn_states = tf.zeros((batch_size, 1, decoder_cell.output_size))
    attn_keys, attn_vals, attn_score_fn, attn_construct_fn \
        = tf.contrib.seq2seq.prepare_attention(attn_states,
                                               attention_option='bahdanau',
                                               num_units=decoder_cell.output_size)
    train_dec_fn = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                 attn_keys,
                                                                 attn_vals,
                                                                 attn_score_fn,
                                                                 attn_construct_fn,
                                                                 name='attn_dec_train')
    # dec_final_state and dec_final_context can be ignored
    dec_output, dec_final_state, dec_final_context \
        = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                 train_dec_fn,
                                                 embedded_input,
                                                 seq_len,
                                                 scope=decode_scope)
    dec_output_dropout = tf.nn.dropout(dec_output, keep_prob)
    return output_fn(dec_output_dropout)


def decode_test_set(encoder_state, decoder_cell, embeddings_mtx, sos_id, eos_id,
                    max_len, num_words, seq_len, decode_scope, output_fn,
                    keep_prob, batch_size):
    attn_states = tf.zeros((batch_size, 1, decoder_cell.output_size))
    attn_keys, attn_vals, attn_score_fn, attn_construct_fn \
        = tf.contrib.seq2seq.prepare_attention(attn_states,
                                               attention_option='bahdanau',
                                               num_units=decoder_cell.output_size)
    test_dec_fn = tf.contrib.seq2seq.attention_decoder_fn_inference(output_fn,
                                                                    encoder_state[0],
                                                                    attn_keys,
                                                                    attn_vals,
                                                                    attn_score_fn,
                                                                    attn_construct_fn,
                                                                    embeddings_mtx,
                                                                    sos_id, eos_id,
                                                                    max_len, num_words,
                                                                    name='attn_dec_inf')
    # dec_final_state and dec_final_context can be ignored
    test_predictions, dec_final_state, dec_final_context \
        = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                 test_dec_fn,
                                                 scope=decode_scope)
    return test_predictions


def decoder_rnn(embedded_input, embeddings_mtx, encoder_state, num_words, 
                seq_len, size, num_layers, word_to_int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
        decoder_cell = tf.contribrnn.MultiRNNCell((lstm_dropout,) * num_layers)
        weights = tf.truncated_normal_initializer(stddev=0.1)
        biases = tf.zeros_initializer()

        def output_fn(inputs):
            return tf.contrib.layers.fully_connected(inputs, num_words, None,
                                                     scope=decoding_scope,
                                                     weights_initializer=weights,
                                                     biases_initializer=biases)

        train_predictions = decode_train_set(encoder_state,
                                             decoder_cell,
                                             embedded_input,
                                             seq_len,
                                             decoding_scope,
                                             output_fn,
                                             keep_prob,
                                             batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           embeddings_mtx,
                                           word_to_int['<SOS>'],
                                           word_to_int['<EOS>'],
                                           seq_len - 1,
                                           num_words,
                                           decoding_scope,
                                           output_fn,
                                           keep_prob,
                                           batch_size)
    return train_predictions, test_predictions


def seq2seq_model(inputs, targets, keep_prob, batch_size, seq_len, answers_num_words, questions_num_words,
                  enc_embedding_size, dec_embedding_size, rnn_size, num_layers, word_to_int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              enc_embedding_size,
                                                              initializer=tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, seq_len)
    preprocessed_targets = preprocess_targets(targets, word_to_int, batch_size)
    decoder_embeddings_mtx = tf.Variable(tf.random_uniform((questions_num_words + 1, dec_embedding_size), 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_mtx, preprocessed_targets)
    train_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                      decoder_embeddings_mtx,
                                                      encoder_state,
                                                      questions_num_words,
                                                      seq_len,
                                                      rnn_size,
                                                      word_to_int,
                                                      keep_prob,
                                                      batch_size)
    return train_predictions, test_predictions


############ TRAIN MODEL ############

# Set parameters

epochs = 100
batch_size = 32
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size = 512
learn_rate = 0.001
learn_rate_decay = 0.9
min_learn_rate = 0.0001
keep_prob = 0.5

tf.reset_default_graph()
session = tf.InteractiveSession()
inputs, targets, lr, kp = model_inputs()
seq_len = tf.placeholder_with_default(25, None, name='seq_len')
inputs_shape = tf.shape(inputs)

train_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, (-1,)),
                                                    targets,
                                                    batch_size,
                                                    seq_len,
                                                    len(word_to_int),
                                                    len(word_to_int),
                                                    encoding_embedding_size,
                                                    decoding_embedding_size,
                                                    rnn_size,
                                                    num_layers,
                                                    word_to_int)

with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(train_predictions,
                                                  targets,
                                                  tf.ones((inputs_shape[0], seq_len)))
    optimizer = tf.train.AdamOptimizer(learn_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tsr, -5., 5.), grad_var)
                         for grad_tsr, grad_var in gradients if grad_tsr is not None]
    optimizer_grad_clipping = optimizer.apply_gradients(clipped_gradients)


def apply_padding(seq_batch, word_to_int):
    max_seq_len = max([len(s) for s in seq_batch])
    pad_token = word_to_int['<PAD>']
    return [s + [pad_token] * (max_seq_len - len(s)) for s in seq_batch]


def split_batches(questions, answers, batch_size):
    for i in range(len(questions) // batch_size):
        start = i * batch_size
        q_batch = questions[start : start + batch_size]
        a_batch = answers[start : start + batch_size]
        q_batch = apply_padding(q_batch, word_to_int)
        a_batch = apply_padding(a_batch, word_to_int)
        yield q_batch, a_batch


train_validation_split = int(len(sorted_clean_q) * 0.15)
train_questions = sorted_clean_q[train_validation_split:]
train_answers = sorted_clean_a[train_validation_split:]
validation_questions = sorted_clean_q[:train_validation_split]
validation_answers = sorted_clean_a[:train_validation_split]

index_train_loss = 100
index_validation_loss = len(train_questions) // batch_size // 2
total_train_loss_error = 0
list_validation_loss_error = 0
early_stop = 100
early_stop_counter = 0
checkpoint = "chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())

# Training
for epoch in range(1, epochs + 1):
    for batch_index, (padded_q, padded_a) in enumerate(split_batches(train_questions,
                                                                     train_answers,
                                                                     batch_size)):
        start_time = time.time()
        _, batch_loss_error = session.run((optimizer_grad_clipping, loss_error),
                                          {inputs: padded_q,
                                           targets: padded_a,
                                           lr: learn_rate,
                                           seq_len: padded_a.shape[1],
                                           keep_prob: keep_prob})
        total_train_loss_error += batch_loss_error
        batch_time = time.time() - start_time
        if batch_index % index_train_loss == 0:
            print(('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, '
                  + 'Training Time on 100 Batches: {:d} seconds').format(epoch, 
                                                                         epochs, 
                                                                         batch_index, 
                                                                         len(train_questions) // batch_size,
                                                                         total_train_loss_error / index_train_loss,
                                                                         int(batch_time * index_train_loss)))
            total_train_loss_error = 0

        # Validation
        if batch_index % index_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            start_time = time.time()
            for batch_index, (padded_q, padded_a) in enumerate(split_batches(validation_questions,
                                                                             validation_answers,
                                                                             batch_size)):
                start_time = time.time()
                _, batch_loss_error = session.run((optimizer_grad_clipping, loss_error),
                                                  {inputs: padded_q,
                                                  targets: padded_a,
                                                  lr: learn_rate,
                                                  seq_len: padded_a.shape[1],
                                                  keep_prob: 1})
                total_train_loss_error += batch_loss_error
            batch_time = time.time() - start_time
            avg_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)
            print(('Validation Loss Error: {:>6.3f}, '
                  + 'Batch Validation Time: {:d} seconds').format(avg_validation_loss_error, int(batch_time)))
            
            learn_rate *= learn_rate_decay
            if learn_rate < min_learn_rate:
                learn_rate = min_learn_rate
            list_validation_loss_error.append(avg_validation_loss_error)
            if avg_validation_loss_error <= min(list_validation_loss_error):
                print('I improved!')
                early_stop_counter = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("I did not improve.")
                early_stop_counter += 1
                if early_stop_counter == early_stop:
                    break

    if early_stop_counter == early_stop:
        print("Early stopped.")
        break
print('Finished training!')


############ TESTING ############

session = tf.InteractiveSession()
session.run(tf.global_variable_initializer())
saver = tf.train.Saver()
saver.restore(session, checkpoint)


def string_to_int(question, word_to_int):
    question = utils.clean_text(question)
    out_int = word_to_int['<OUT>']
    return [word_to_int.get(w, out_int) for w in question.split()]


while True:
    question = input('>>> ')
    if question.lower() == 'bye':
        break
    question = string_to_int(question, word_to_int) + [word_to_int['<PAD>']] * (20 - len(question))
    fake_batch = np.zeros((batch_size, 20))
    fake_batch[0] = question
    predicted_answer = session.run(test_predictions,
                                   {inputs: fake_batch,
                                    keep_prob: 0.5})[0]
    answer = ''
    for i in np.argmax(predicted_answer, 1):
        if word_to_int[i] == 'i':
            token = 'I'
        elif word_to_int == '<EOS>':
            token = '.'
        else:
            token = ' ' + int_to_word[i]
        answer += token
        if token == '.':
            break
    print(answer)
