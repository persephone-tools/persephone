import timit

import numpy as np
import tensorflow as tf

#def create_graph(batch_size=100, lstm_size=250, num_steps=778):
#    """ Creates RNN/CTC graphs for acoustic modeling. """

# Number of utterances in each batch
batch_size=100
# Number of hidden units in each LSTM cell.
lstm_size=250
# How far we're expecting to unroll. Currently 778 for longest TIMIT sentence
utter_length=778

# Shape is (batch_size, utter_length, feats, channels)
feats = tf.placeholder(tf.float32, [100, 26*3, utter_length])

# A slower version of BasicLSTMCell from later tensorflow versions.
lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)

initial_state = state = tf.zeros([batch_size, lstm_size])

for i in range(utter_length):
    output, state = lstm(feats[:, :, i], state)

def collapsed_timit():
    """ Converts timit into an array of format (batch_size, freq, time). Except
    where Freq is Freqxnum_deltas, so usually freq*3. Essentially multiple
    channels are collapsed to one"""

    train_feats, train_labels = timit.load_timit(batch_size=batch_size, labels="phonemes")
    print train_feats.shape
    new_train = []
    for utterance in train_feats:
        swapped = np.swapaxes(utterance,0,1)
        concatenated = np.concatenate(swapped,axis=1)
        reswapped = np.swapaxes(concatenated,0,1)
        new_train.append(reswapped)
    train_feats = np.array(new_train)
    return train_feats

if __name__ == "__main__":

    train = collapsed_timit()
    print train[:,:,1].shape
