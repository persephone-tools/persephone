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

    def __init__(self, batch_x, x_lens,
            batch_y, batch_seq_lens, num_layers=1, hidden_size=250, freq_feats=123,
            vocab_size=timit.num_phones):
        self.inputs = batch_x
        self.input_lens = x_lens
        self.targets = batch_y
        self.seq_lens = batch_seq_lens
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.freq_feats = freq_feats
        self._optimize = None
        self._error = None
        self._decoded = None

        batch_size = tf.shape(self.inputs)[0]

        cell = multi_cell(self.num_layers, self.hidden_size)

        self.outputs, _ = tf.nn.dynamic_rnn(cell, self.inputs, self.input_lens, dtype=tf.float32, 
                time_major=False)

        self.outputs = tf.reshape(self.outputs, [-1, self.hidden_size])

        #self.output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size])
        W = tf.Variable(tf.truncated_normal([hidden_size, vocab_size+1]))
        b = tf.Variable(tf.constant(0.1, shape=[vocab_size+1]))
        self.logits = tf.matmul(self.outputs, W) + b
        self.logits = tf.reshape(self.logits, [batch_size, -1, vocab_size+1])
        # igormq made it time major, because of an optimization in ctc_loss.
        self.logits = tf.transpose(self.logits, (1, 0, 2))

        #decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)
        self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(
                self.logits, self.seq_lens, beam_width=100)

        self.loss = tf.nn.ctc_loss(self.targets, self.logits, self.seq_lens,
                preprocess_collapse_repeated=True)
        self.cost = tf.reduce_mean(self.loss)
        # Trying Adam with defaults.
        #self.optimizer = tf.train.AdadeltaOptimizer().minimize(self.cost)
        self.optimizer = tf.train.MomentumOptimizer(1e-4, 0.9).minimize(self.cost)

        self.ler = tf.reduce_mean(tf.edit_distance(
                tf.cast(self.decoded[0], tf.int32), self.targets))

def main():
    batch_size = 50 # The size of each batch
    total_size = 100 # The total number of TIMIT training examples

    # A generator that pumps out batches
    batch_gen = timit.batch_gen(batch_size=batch_size, labels="phonemes",
            total_size=total_size, rand=True)

    freq_feats = 26 # This should be detected from data
    #utter_len = 778 # This should be detected from data

    inputs = tf.placeholder(tf.float32, [None, None, freq_feats])
    input_lens = tf.placeholder(tf.float32, [None])
    targets = tf.sparse_placeholder(tf.int32)
    # The lengths of the target sequences.
    seq_lens = tf.placeholder(tf.int32, [None])

    model = RNNCTC(inputs, input_lens, targets, seq_lens, freq_feats=freq_feats)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    num_epochs = 1000
    for epoch in range(num_epochs):
        batch_gen = timit.batch_gen(batch_size=batch_size, labels="phonemes",
                total_size=total_size, rand=False)

        for batch in batch_gen:
            batch_x, x_lens, batch_y = batch
            batch_seq_lens = np.asarray([len(s) for s in batch_y], dtype=np.int32)
            batch_y = target_list_to_sparse_tensor(batch_y)

            #print(batch_x.shape)
            #print(x_lens.shape)
            #print(batch_seq_lens.shape)

            feed_dict={inputs: batch_x, input_lens: x_lens, targets: batch_y, seq_lens: batch_seq_lens}
            #feed_dict={inputs: batch_x, input_lens: x_lens}

            #print(sess.run(model.logits, feed_dict=feed_dict).shape)
            #import sys; sys.exit()
            _, error, decoded = sess.run([model.optimizer, model.ler, model.decoded], feed_dict=feed_dict)
            timit.error_rate(batch_y, decoded)
            #sess.run(model.optimizer, feed_dict=feed_dict)
            #print(decoded[0])
            #print(targets)
            print("Epoch %d training error: %f" % (epoch, error))

    sess.close()

def create_graph(batch_size=100, utter_len=778, freqfeats=123,
        num_layers=1, hidden_size=250, keep_prob=1.0,
        vocab_size=timit.num_phones,
        learning_rate=1e-4, momentum=0.90):

    print("Creating graph...")

    def lstm_cell(hidden_size):
        return tf.contrib.rnn.LSTMCell(
                hidden_size,
                use_peepholes=True,
                state_is_tuple=True)

    #def dropout_cell(hidden_size, keep_prob): #    return tf.contrib.rnn.DropoutWrapper(lstm_cell(hidden_size),
    #            output_keep_prob=keep_prob)

    def multi_cell(num_layers, hidden_size):
        return tf.contrib.rnn.MultiRNNCell(
                [lstm_cell(hidden_size) for _ in range(num_layers)],
                state_is_tuple=True)
    #    return tf.contrib.rnn.MultiRNNCell(
    #            [dropout_cell(hidden_size, keep_prob) for _ in range(num_layers)],
    #            state_is_tuple=True)

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
    loss = tf.nn.ctc_loss(targets, logits, seq_len, preprocess_collapse_repeated=True)
    cost = tf.reduce_mean(loss)
    #optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(cost)

    # Trying Adam with defaults.
    optimizer = tf.train.AdadeltaOptimizer().minimize(cost)

    #decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, beam_width=100)
    ler = tf.reduce_mean(tf.edit_distance(
            tf.cast(decoded[0], tf.int32), targets))

    print("Graph created.")
    # Should engineer it so that I don't have to pick and choose ops and
    # tensors to return.
    return feats, targets, seq_len, optimizer, ler, decoded, logits, loss, cost

def train(batch_size=60, total_size=4620, num_epochs=1000):
    tf.reset_default_graph()
    feats, targets, seq_len, optimizer, ler, decoded, logits, loss, cost = create_graph(batch_size=batch_size)

    init = tf.global_variables_initializer()
    #saver = tf.train.Saver()

    batch_gen = timit.batch_gen(batch_size=batch_size, labels="phonemes", total_size=total_size, rand=False)
    #batches = [batch for batch in batch_gen]
    #batches = [batches[6]]

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            batch_gen = timit.batch_gen(batch_size=batch_size, labels="phonemes", total_size=total_size)
            print("Beginning epoch %d" % epoch)
            epoch_ler = 0
            for batch_i, batch in enumerate(batch_gen):
            #for batch_i, batch in enumerate(batch_gen):
                train_x, train_y = collapsed_timit(batch)
                #print("epoch %d, batch %d, train_x shape %s" % (epoch, batch_i, str(train_x.shape)))
                seq_lens = np.asarray([len(s) for s in train_y], dtype=np.int32)
#                print(train_x)
#                print(train_y[0])
#                print(timit.indices_to_phones(train_y[0]))
                train_y = target_list_to_sparse_tensor(train_y)

                _, out_ler, out_decoded = sess.run([optimizer, ler, decoded], feed_dict={feats: train_x, targets: train_y, seq_len: seq_lens})

                #hypo = timit.indices_to_phones(out_decoded[0].values)
                #print("ref: %s" % str(hypo))
                #ref = timit.indices_to_phones(train_y[1])
                #print("hypo: %s" % str(ref))
                #collapsed_hypo = timit.collapse_phones(hypo)
                #print("collapsed ref: %s" % str(collapsed_hypo))
                #collapsed_ref = timit.collapse_phones(ref)
                #print("collapsed hypo: %s" % str(collapsed_ref))
                #import pdb; pdb.set_trace()
                #print("PER: %f" % (distance.edit_distance(collapsed_hypo, collapsed_ref)/float(len(ref))))

                #print(out_logits)
                #input()
                epoch_ler += out_ler
                print("epoch %d, batch %d" % (epoch, batch_i))
                #print(timit.indices_to_phones(out_decoded[0].values))
            print("Epoch %d average LER: %f" % (epoch, epoch_ler / (batch_i+1)))
            #saver.save(sess, "model_epoch%d" % epoch)


if __name__ == "__main__":
    #train()
    main()
