import re

def clean_text(s):
    s = s.lower()
    s = re.sub(r"i'm", "i am", s)
    s = re.sub(r"he's", "he is", s)
    s = re.sub(r"she's", "she is", s)
    s = re.sub(r"that's", "i am", s)
    s = re.sub(r"what's", "what is", s)
    s = re.sub(r"where's", "where is", s)
    s = re.sub(r"'ll", " will", s)
    s = re.sub(r"'ve", " have", s)
    s = re.sub(r"'re", " are", s)
    s = re.sub(r"'d", " would", s)
    s = re.sub(r"won't", "will not", s)
    s = re.sub(r"can't", "cannot", s)
    s = re.sub(r"[^a-zA-Z0-9 ]+", "", s)
    return s


def load_vocab_index(file_path):
    with open(file_path) as f:
        vocab = f.read().split('\n')
        vocab_index = {}
        for i, v in enumerate(vocab):
            vocab_index[v] = i
    return vocab_index


def load_train_data(file_path):
    with open(file_path) as f:
        inputs, targets = [], []
        for pair in f.read().split('\n'):
            pair = pair.split(',')
            if len(pair) < 2:
                break
            inp, targ = pair[0], pair[1]
            inputs.append(list(map(int, inp.split(' '))))
            targets.append(list(map(int, targ.split(' '))))
    return inputs, targets
