import numpy as np
import tensorflow as tf
import time
import utils

with open('cmdc/movie_lines.txt', encoding='utf-8', errors='ignore') as f:
    lines = f.read().split('\n')
with open('cmdc/movie_conversations.txt', encoding='utf-8', errors='ignore') as f:
    conversations = f.read().split('\n')

# Preprocess data

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

clean_q = []
for q in questions:
    clean_q.append(utils.clean_text(q))

clean_a = []
for a in answers:
    clean_a.append(utils.clean_text(a))

word_counts = {}
for line in clean_q + clean_a:
    for word in line.split():
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
