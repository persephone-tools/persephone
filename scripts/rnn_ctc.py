import timit

import numpy as np
import tensorflow as tf

def SimpleSparseTensorFrom(x):
    """Create a very simple SparseTensor with dimensions (batch, time).
    Args:
        x: a list of lists of type int
    Returns:
        x_ix and x_val, the indices and values of the SparseTensor<2>.

    Note: this code was taken from
    tensorflow/python/kernel_tests/ctc_loss_op_test.py. It's purpose is to do
    magic to a batch of label sequences so it can be fed to tf.nn.ctc_loss().
    """

    from tensorflow.python.framework import constant_op
    from tensorflow.python.framework import dtypes
    from tensorflow.python.framework import sparse_tensor

    x_ix = []
    x_val = []
    for batch_i, batch in enumerate(x):
        for time, val in enumerate(batch):
            x_ix.append([batch_i, time])
            x_val.append(val)
    x_shape = [len(x), np.asarray(x_ix).max(0)[1] + 1]
    x_ix = constant_op.constant(x_ix, dtypes.int64)
    x_val = constant_op.constant(x_val, dtypes.int32)
    x_shape = constant_op.constant(x_shape, dtypes.int64)

    return sparse_tensor.SparseTensor(x_ix, x_val, x_shape)

def collapsed_timit(batch, batch_size=100):
    """ Converts timit into an array of format (batch_size, freq, time). Except
    where Freq is Freqxnum_deltas, so usually freq*3. Essentially multiple
    channels are collapsed to one"""

    train_feats, train_labels = batch
    new_train = []
    for utterance in train_feats:
        swapped = np.swapaxes(utterance,0,1)
        concatenated = np.concatenate(swapped,axis=1)
        reswapped = np.swapaxes(concatenated,0,1)
        new_train.append(reswapped)
    train_feats = np.array(new_train)
    return train_feats, train_labels

def create_graph(batch_size=100, utter_len=778, freqfeats=78,
        num_layers=2, hidden_size=250, keep_prob=1.0,
        vocab_size=timit.num_timit_phones,
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

#    feats = tf.Variable(tf.random_normal(
#            [batch_size, freqfeats, utter_len], stddev=0.35),
#            name="feats", trainable=False)
    feats = tf.placeholder(tf.float32, [None, freqfeats, utter_len])
    # The target labels
    targets = tf.sparse_placeholder(tf.int32)
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
    loss = tf.nn.ctc_loss(targets, logits, seq_len)
    cost = tf.reduce_mean(loss)
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cost)

    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)
    ler = tf.reduce_mean(tf.edit_distance(
            tf.cast(decoded[0], tf.int32), targets))

    # Should engineer it so that I don't have to pick and choose ops and
    # tensors to return.
    return feats, optimizer, ler, decoded

if __name__ == "__main__":
    batch_size=100
    feats, optimizer, ler, decoded = create_graph(batch_size=batch_size)

    init = tf.global_variables_initializer()
    batch_gen = timit.batch_gen(batch_size=batch_size, labels="phonemes")
    train_x, train_y = collapsed_timit(next(batch_gen))
    seq_lens = np.asarray([len(s) for s in train_y], dtype=np.int64)
    train_y = SimpleSparseTensorFrom(train_y)
    print(seq_lens)
    print(train_y)

    with tf.Session() as sess:
        for _ in range(1000):
            print(sess.run([optimizer, ler, decoded], feed_dict={feats: train_x,
                    targets: train_y, seq_len: seq_lens}).shape)
