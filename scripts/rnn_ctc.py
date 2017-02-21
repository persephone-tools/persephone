import timit

import numpy as np
import tensorflow as tf

def target_list_to_sparse_tensor(targetList):
    '''make tensorflow SparseTensor from list of targets, with each element
       in the list being a list or array with the values of the target sequence
       (e.g., the integer values of a character map for an ASR target string)
       See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/ctc/ctc_loss_op_test.py
       for example of SparseTensor format'''
    indices = []
    vals = []
    for tI, target in enumerate(targetList):
        for seqI, val in enumerate(target):
            indices.append([tI, seqI])
            vals.append(val)
    shape = [len(targetList), np.asarray(indices).max(0)[1]+1]
    return (np.array(indices), np.array(vals), np.array(shape))

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
    return feats, targets, seq_len, optimizer, ler, decoded

def train():
    batch_size=1
    feats, targets, seq_len, optimizer, ler, decoded = create_graph(batch_size=batch_size)

    init = tf.global_variables_initializer()
    batch_gen = timit.batch_gen(batch_size=batch_size, labels="phonemes")
    train_x, train_y = collapsed_timit(next(batch_gen))
    seq_lens = np.asarray([len(s) for s in train_y], dtype=np.int32)
    train_y = target_list_to_sparse_tensor(train_y)
    print(seq_lens)
    print(train_y)

    with tf.Session() as sess:
        sess.run(init)
        for _ in range(1000):
            print(sess.run([optimizer, ler, decoded], feed_dict={feats: train_x, targets: train_y, seq_len: seq_lens}))

if __name__ == "__main__":
    train()
