from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
import numpy as np
import time
import model
import utils


############ PREPROCESSING DATA ############

with open('cmdc/movie_lines.txt', encoding='utf-8', errors='ignore') as f:
    lines = f.read().split('\n')
with open('cmdc/movie_conversations.txt', encoding='utf-8', errors='ignore') as f:
    conversations = f.read().split('\n')

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

# Create vocab to integer mapping
MIN_WORD_COUNT = 15
vocab_int = {
    '<PAD>': 0,
    '<SOS>': 1,
    '<EOS>': 2,
    '<EX>': 3
}
word_index = 4
for word, count in word_counts.items():
    if count >= MIN_WORD_COUNT:
        vocab_int[word] = word_index
        word_index += 1

# Recreate questions and answers with integers
inputs = []
for line in clean_questions:
    inputs.append([1] + [vocab_int.get(w, 3) for w in line.split()] + [2])
targets = []
for line in clean_answers:
    targets.append([1] + [vocab_int.get(w, 3) for w in line.split()] + [2])

train_inputs, test_inputs, train_targets, test_targets = train_test_split(inputs,
                                                                          targets,
                                                                          test_size=0.5,
                                                                          shuffle=True)

############ TRAINING ############

def gen_batch_indices(indices, batch_size):
    for i in range(len(indices) // batch_size + 1):
        start = i * batch_size
        yield indices[start : start + batch_size]


MAX_EPOCHS = 20
BATCH_SIZE = 32
hparams = {
    'embedding_dim': 256,
    'units': 256,
    'n_layers': 2,
    'dropout': 0.1,
    'learn_rate': 0.001
}
REPORT_FREQ = 10 # batches between each printed report

chatbot = model.ChatbotModel(hparams, vocab_int, 'checkpoint.ckpt')
print("Starting training")

cross_val = KFold(n_splits=10)
for tr, val in cross_val.split(train_inputs, train_targets):
    start_fold = time.time()
    fold_validation_loss = []

    for epoch in range(MAX_EPOCHS):
        start_epoch = time.time()

        train_gen = gen_batch_indices(tr, BATCH_SIZE)
        total_train_loss = 0
        for (batch_i, batch_indices) in enumerate(train_gen):
            batch_inputs = [train_inputs[i] for i in batch_indices]
            batch_targets = [train_targets[i] for i in batch_indices]
            total_train_loss += chatbot.train_batch(batch_inputs, batch_targets)

            if (batch_i + 1) % REPORT_FREQ == 0:
                print('Epoch {}, batch {} loss: {:.4f}'.format(epoch + 1, 
                                                               batch_i + 1, 
                                                               total_train_loss))
                total_train_loss = 0

        validation_gen = gen_batch_indices(val, BATCH_SIZE)
        for (batch_i, batch_indices) in enumerate(validation_gen):
            batch_inputs = [train_inputs[i] for i in batch_indices]
            batch_targets = [train_targets[i] for i in batch_indices]
            fold_validation_loss.append(chatbot.test_batch(batch_inputs, batch_targets))

        print('Epoch {} time: {} sec\n'.format(epoch + 1,
                                               time.time() - start_epoch))
        if epoch % 2 == 0:
            chatbot.save()

    print('Fold time: {}'.format(time.time() - start_fold))
    print(fold_validation_loss)
    break # bamboozled, not actually performing k-fold cv
