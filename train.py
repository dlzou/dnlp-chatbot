from sklearn.model_selection import train_test_split, KFold
import numpy as np
import time
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
                                                                          test_size=0.2)

############ TRAINING ############

EPOCHS = 10
BATCH_SIZE = 32
hparams = {
    'embedding_dim': 256,
    'units': 512,
    'n_layers': 3,
    'dropout': 0.1,
    'learn_rate': 0.001
}

def get_batch(inputs, targets, batch_size):
    pass


print("Starting training")
for epoch in range(EPOCHS):
    start = time.time()
    total_loss = 0

    for (batch_i, (batch_inputs, batch_targets)) in enumerate():
        break

# k-fold cross validation
