import timit
from utils import target_list_to_sparse_tensor
from nltk.metrics import distance

import logging
import numpy as np
import tensorflow as tf

def lstm_cell(hidden_size):
    return tf.contrib.rnn.LSTMCell(
            hidden_size,
            use_peepholes=True,
            state_is_tuple=True)

def multi_cell(num_layers, hidden_size):
    return tf.contrib.rnn.MultiRNNCell(
            [lstm_cell(hidden_size) for _ in range(num_layers)],
            state_is_tuple=True)

class RNNCTC:

    def __init__(self, batch_x, x_lens, batch_y, batch_seq_lens,
            num_layers=3, hidden_size=250, vocab_size=timit.num_phones,
            learning_rate=1e-4, momentum=0.9, beam_width=100):
        self.inputs = batch_x
        self.input_lens = x_lens
        self.targets = batch_y
        self.seq_lens = batch_seq_lens
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self._optimize = None
        self._error = None
        self._decoded = None

        batch_size = tf.shape(self.inputs)[0]

        cell = multi_cell(self.num_layers, self.hidden_size)

        self.outputs, _ = tf.nn.dynamic_rnn(cell, self.inputs, self.input_lens, dtype=tf.float32, 
                time_major=False)

        self.outputs = tf.reshape(self.outputs, [-1, self.hidden_size])

        W = tf.Variable(tf.truncated_normal([hidden_size, vocab_size+1]))
        b = tf.Variable(tf.constant(0.1, shape=[vocab_size+1]))
        self.logits = tf.matmul(self.outputs, W) + b
        self.logits = tf.reshape(self.logits, [batch_size, -1, vocab_size+1])
        # igormq made it time major, because of an optimization in ctc_loss.
        self.logits = tf.transpose(self.logits, (1, 0, 2))

        self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(
                self.logits, self.seq_lens, beam_width=beam_width)

        self.loss = tf.nn.ctc_loss(self.targets, self.logits, self.seq_lens,
                preprocess_collapse_repeated=True)
        self.cost = tf.reduce_mean(self.loss)
        #self.optimizer = tf.train.AdadeltaOptimizer().minimize(self.cost)
        self.optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(self.cost)

        self.ler = tf.reduce_mean(tf.edit_distance(
                tf.cast(self.decoded[0], tf.int32), self.targets))

def main(batch_size=100, total_size=4600, num_epochs=200):
    """ Run an experiment. 

        batch_size: The number of utterances in each batch.
        total_size: The number of TIMIT training examples to use.
        num_epochs: The number of times to iterate over all the training
        examples.
    """

    # A generator that pumps out batches
    batch_gen = timit.batch_gen(batch_size=batch_size, labels="phonemes",
            total_size=total_size)
    # Load the data and observe the number of feats in the numpy arrays.
    freq_feats = next(batch_gen)[0].shape[-1]

    inputs = tf.placeholder(tf.float32, [None, None, freq_feats])
    input_lens = tf.placeholder(tf.float32, [None])
    targets = tf.sparse_placeholder(tf.int32)
    # The lengths of the target sequences.
    seq_lens = tf.placeholder(tf.int32, [None])

    model = RNNCTC(inputs, input_lens, targets, seq_lens)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(num_epochs):
        batch_gen = timit.batch_gen(batch_size=batch_size, labels="phonemes",
                total_size=total_size, rand=True)

        err_total = 0
        for batch_i, batch in enumerate(batch_gen):
            batch_x, x_lens, batch_y = batch
            batch_seq_lens = np.asarray(
                    [len(s) for s in batch_y], dtype=np.int32)
            batch_y = target_list_to_sparse_tensor(batch_y)

            feed_dict={inputs: batch_x, input_lens: x_lens, targets:
                       batch_y, seq_lens: batch_seq_lens}
            _, error, decoded = sess.run(
                    [model.optimizer, model.ler, model.decoded],
                    feed_dict=feed_dict)
            #timit.error_rate(batch_y, decoded)

            err_total += error

        print("Epoch %d training error: %f" % (
                epoch, (err_total / (batch_i + 1))))

    sess.close()

if __name__ == "__main__":
    main()
