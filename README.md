# Deep NLP Chatbot

A seq2seq model built in TensorFlow 2 and trained on movie dialogue from the [Cornell Movie-Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html). 

### Details

My model uses the standard recurrent encoder-decoder architecture. Both the encoder and the decoder are implemented as stacked bidirectional RNNs with GRU cells, and a Bahdanau-style attention mechanism serves to improve memory of long sequences during decoding. In the future, I may replace the greedy search decoder with a beam search decoder.

CMDC is organized such that collections of lines form conversations. After parsing the data, I iterated through each conversation to produce question-answer pairs, which became the input and target data for training.

### References

I initially referenced tutorials from SuperDataScience and TensorFlow. As I branched out, I came across many blogs and papers to helped me learn about RNNs, attention, and beam search. I also got a glimpse of hierarchical encoders, transformers, and other state-of-the-art techniques.
