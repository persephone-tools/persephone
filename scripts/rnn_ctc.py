import numpy as np
import tensorflow as tf

import timit
from utils import target_list_to_sparse_tensor

def lstm_cell(hidden_size):
    return tf.contrib.rnn.LSTMCell(
            hidden_size,
            use_peepholes=True,
            state_is_tuple=True)

class Model:

    def __init__(self, vocab_size, num_feats, num_layers,
                 hidden_size=250, beam_width=100):

        self.hidden_size = hidden_size

        # Initialize placeholders for feeding data to model.
        #num_feats = timit.num_feats(feat_type)
        self.batch_x = tf.placeholder(tf.float32, [None, None, num_feats])
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

        W = tf.Variable(tf.truncated_normal([hidden_size*2, vocab_size],
                stddev=np.sqrt(2.0 / (2*hidden_size))))
        b = tf.Variable(tf.zeros([vocab_size]))
        self.logits = tf.matmul(self.outputs, W) + b
        self.logits = tf.reshape(self.logits, [batch_size, -1, vocab_size])
        # igormq made it time major, because of an optimization in ctc_loss.
        self.logits = tf.transpose(self.logits, (1, 0, 2))

        self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(
                self.logits, self.batch_x_lens, beam_width=beam_width)

        self.loss = tf.nn.ctc_loss(self.batch_y, self.logits, self.batch_x_lens,
                preprocess_collapse_repeated=True)
        self.cost = tf.reduce_mean(self.loss)
        self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)

        self.ler = tf.reduce_mean(tf.edit_distance(
                tf.cast(self.decoded[0], tf.int32), self.batch_y))

        # If we want to do manual PER decoding. The decoded[0] beans the best
        # hypothesis (0th) in an n-best list.
        self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0])
        self.dense_ref = tf.sparse_tensor_to_dense(self.batch_y)
