from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import model
import utils

VOCAB_FILE = 'preprocessed/vocab_index.txt'
CHECKPOINT = 'checkpoint/checkpoint'

vocab_index = utils.load_vocab_index(VOCAB_FILE)
hparams = {
    'embedding_dim': 256,
    'units': 256,
    'n_layers': 2,
    'dropout': 0.1,
    'learn_rate': 0.001
}
chatbot = model.ChatbotModel(hparams, vocab_index, CHECKPOINT)
chatbot.restore()

if __name__ == '__main__':
    while True:
        input_seq = input('> ')
        if input_seq == 'bye':
            break
        input_seq = [1] + [vocab_index.get(w, 3) for w in input_seq.split()] + [2]
        output_seq = chatbot.evaluate(input_seq, 30)
        output_seq = ' '.join([list(vocab_index.keys())[i] for i in output_seq])
        print(output_seq)
