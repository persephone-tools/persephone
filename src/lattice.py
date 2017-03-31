""" Ultimately a module to produce lattices from the multinomials of CTC-based
    neural networks. Includes other methods to do things such as sample and
    collapse paths.
"""

import tensorflow as tf

from corpus_reader import CorpusReader
import datasets.na
import rnn_ctc
import run

def test_sampling():
    """ As a proof of concept, first we want to sample paths from a trained CTC
    network's multinomials. After collapsing them, we want to assess their
    phoneme error rate to ensure we frequently get decent PERs."""
    pass

def test_1_best():
    """ Test my own 1-best path generator. """

    # Train the model
    exp_dir = run.prep_exp_dir()
    corpus = datasets.na.Corpus(feat_type="log_mel_filterbank",
                                target_type="phn")
    corpus_reader = CorpusReader(corpus, num_train=512)
    model = rnn_ctc.Model(exp_dir, corpus_reader)
    model.train()

    valid_x, valid_x_lens, valid_y = corpus_reader.valid_batch()
    feed_dict = {model.batch_x: valid_x,
                 model.batch_x_lens: valid_x_lens,
                 model.batch_y: valid_y}

    with tf.Session() as sess:
        logits, dense_decoded = sess.run([model.logits, model.dense_decoded],
                                         feed_dict=feed_dict)
