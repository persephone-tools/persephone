""" An acoustic model with a LSTM/CTC architecture. """

import os
import numpy as np
import tensorflow as tf

from . import model

def lstm_cell(hidden_size):
    """ Wrapper function to create an LSTM cell. """

    return tf.contrib.rnn.LSTMCell(
        hidden_size, use_peepholes=True, state_is_tuple=True)

class Model(model.Model):
    """ An acoustic model with a LSTM/CTC architecture. """

    def write_desc(self):
        """ Writes a description of the model to the exp_dir. """

        path = os.path.join(self.exp_dir, "model_description.txt")
        with open(path, "w") as desc_f:
            for key, val in self.__dict__.items():
                print("%s=%s" % (key, val), file=desc_f)

    def __init__(self, exp_dir, corpus_reader, num_layers=3,
                 hidden_size=250, beam_width=100, decoding_merge_repeated=True):
        super().__init__(exp_dir, corpus_reader)

        if not os.path.isdir(exp_dir):
            os.makedirs(exp_dir)

        # Increase vocab size by 2 since we need an extra for CTC blank labels
        # and another extra for dynamic padding with zeros.
        vocab_size = corpus_reader.corpus.vocab_size+2

        # Reset the graph.
        tf.reset_default_graph()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.beam_width = beam_width
        self.vocab_size = vocab_size

        # Initialize placeholders for feeding data to model.
        self.batch_x = tf.placeholder(
                tf.float32, [None, None, corpus_reader.corpus.num_feats])
        self.batch_x_lens = tf.placeholder(tf.int32, [None])
        self.batch_y = tf.sparse_placeholder(tf.int32)

        batch_size = tf.shape(self.batch_x)[0]

        layer_input = self.batch_x

        for i in range(num_layers):

            with tf.variable_scope("layer_%d" % i):

                cell_fw = lstm_cell(self.hidden_size)
                cell_bw = lstm_cell(self.hidden_size)

                (self.out_fw, self.out_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, layer_input, self.batch_x_lens, dtype=tf.float32,
                        time_major=False)

                # Self outputs now becomes [batch_num, time, hidden_size*2]
                self.outputs_concat = tf.concat((self.out_fw, self.out_bw), 2)

                # For feeding into the next layer
                layer_input = self.outputs_concat

        self.outputs = tf.reshape(self.outputs_concat, [-1, self.hidden_size*2])

        # Single-variable names are appropriate for weights an biases.
        # pylint: disable=invalid-name
        W = tf.Variable(tf.truncated_normal([hidden_size*2, vocab_size],
                stddev=np.sqrt(2.0 / (2*hidden_size))))
        b = tf.Variable(tf.zeros([vocab_size]))
        self.logits = tf.matmul(self.outputs, W) + b
        self.logits = tf.reshape(self.logits, [batch_size, -1, vocab_size])
        # igormq made it time major, because of an optimization in ctc_loss.
        self.logits = tf.transpose(self.logits, (1, 0, 2))

        # For lattice construction
        self.log_softmax = tf.nn.log_softmax(self.logits)

        self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(
                self.logits, self.batch_x_lens, beam_width=beam_width,
                merge_repeated=decoding_merge_repeated)

        # If we want to do manual PER decoding. The decoded[0] beans the best
        # hypothesis (0th) in an n-best list.
        self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0])
        self.dense_ref = tf.sparse_tensor_to_dense(self.batch_y)

        self.loss = tf.nn.ctc_loss(self.batch_y, self.logits, self.batch_x_lens,
                preprocess_collapse_repeated=False, ctc_merge_repeated=True)
        self.cost = tf.reduce_mean(self.loss)
        self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)

        self.ler = tf.reduce_mean(tf.edit_distance(
                tf.cast(self.decoded[0], tf.int32), self.batch_y))

        self.write_desc()
