from sklearn.model_selection import train_test_split
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

vocab_int = {
    '<PAD>': 0,
    '<SOS>': 1,
    '<EOS>': 2,
    '<EX>': 3
}

tf.keras.preprocessing.text.Tokenizer

EPOCHS = 10
BATCH_SIZE = 32


for (batch_i, (batch_input, batch_target)) in enumerate():
    pad_batch_inp = tf.keras.preprocessing.sequence.pad_sequences(batch_inp, padding='post')
    pad_batch_targ = tf.keras.preprocessing.sequence.pad_sequences(batch_targ, padding='post')
    break

# k-fold cross validation
