import timit

import numpy as np
import tensorflow as tf

def collapsed_timit(batch_size=100):
    """ Converts timit into an array of format (batch_size, freq, time). Except
    where Freq is Freqxnum_deltas, so usually freq*3. Essentially multiple
    channels are collapsed to one"""

    train_feats, train_labels = timit.load_timit(batch_size=batch_size, labels="phonemes")
    new_train = []
    for utterance in train_feats:
        swapped = np.swapaxes(utterance,0,1)
        concatenated = np.concatenate(swapped,axis=1)
        reswapped = np.swapaxes(concatenated,0,1)
        new_train.append(reswapped)
    train_feats = np.array(new_train)
    return train_feats, train_labels

def create_graph(batch_size=100, utter_len=778, freqfeats=78,
        num_layers=2, hidden_size=250, keep_prob=1.0, vocab_size=61,
        learning_rate=1e-4):

    def lstm_cell(hidden_size):
        return tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=0.0,
                state_is_tuple=True)

    def dropout_cell(hidden_size, keep_prob):
        return tf.contrib.rnn.DropoutWrapper(lstm_cell(hidden_size),
                output_keep_prob=keep_prob)

    def multi_cell(num_layers, hidden_size):
        return tf.contrib.rnn.MultiRNNCell(
                [dropout_cell(hidden_size, keep_prob) for _ in range(num_layers)],
                state_is_tuple=True)

    feats = tf.Variable(tf.random_normal(
            [batch_size, freqfeats, utter_len], stddev=0.35),
            name="feats", trainable=False)

    labels = tf.sparse_placeholder(tf.int32)
    seq_len = tf.placeholder(tf.int32, [None])

    cell = multi_cell(num_layers, hidden_size)

    initial_state = cell.zero_state(batch_size, tf.float32)
    state = initial_state

    outputs = []
    with tf.variable_scope("RNN"):
        for time_step in range(utter_len):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            (cell_output, state) = cell(feats[:, :, time_step], state)
            outputs.append(cell_output)

    output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size])
    W = tf.Variable(tf.truncated_normal([hidden_size, vocab_size+1]))
    b = tf.Variable(tf.constant(0.1, shape=[vocab_size+1]))
    logits = tf.matmul(output, W) + b
    logits = tf.reshape(logits, [batch_size, -1, vocab_size+1])
    # igormq made it time major, because of an optimization in ctc_loss.
    logits = tf.transpose(logits, (1, 0, 2))
    loss = tf.nn.ctc_loss(labels, logits, seq_len)
    cost = tf.reduce_mean(loss)
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cost)

    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)
    ler = tf.reduce_mean(tf.edit_distance(
            tf.cast(decoded[0], tf.int32), labels))

    # Should engineer it so that I don't have to pick and choose ops and
    # tensors to return.
    return ler, feats

if __name__ == "__main__":
    train_x, train_y = collapsed_timit()
    ler, feats = create_graph()
    print(ler)
    print(feats)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(feats))
