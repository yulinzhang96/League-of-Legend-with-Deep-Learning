import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import sys

# loading
with open('./models/BiLSTM/tokenizer.pickle', 'rb') as handle:
    vocab = pickle.load(handle)

index_word = {}
for word, index in vocab.word_index.items():
    index_word[index] = word

model = tf.keras.models.load_model("./models/BiLSTM/LSTM.h5")
max_sequence_len = 6694

def runModel(text, length):
    #print(len(vocab.word_index)+1)
    #print(model.summary())

    seed_text = text
    #next_words = int(next_words_length)
    loop = int(length)

    while loop > 0:
        token_list = vocab.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        best_word = np.argmax(predicted)
        #best_word_pro = tf.nn.top_k(predicted, k=1, sorted=True, name=None)
        #best_word_prob = np.array(best_word_pro[0][0][0])
        #predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        #predicted = tf.nn.top_k(model.predict(token_list, verbose=0), k=1, sorted=True, name=None)
        output_word = index_word[best_word]

        seed_text += " " + output_word
        loop -= 1
    seed_text += "."
    print(seed_text)

text = sys.argv[1]
length = sys.argv[2]

runModel(text, length)
