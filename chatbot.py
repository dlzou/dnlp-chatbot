import numpy as np
import tensorflow as tf
import time
import utils

with open('cmdc/movie_lines.txt', encoding='utf-8', errors='ignore') as f:
    lines = f.read().split('\n')
with open('cmdc/movie_conversations.txt', encoding='utf-8', errors='ignore') as f:
    conversations = f.read().split('\n')

'''
Preprocess data
'''
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

min_count = 20
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
for length in range(1, max_q_len + 1):
    for i, q in enumerate(questions_int):
        if len(q) == length:
            sorted_clean_q.append(questions_int[i])
            sorted_clean_a.append(answers_int[i])

