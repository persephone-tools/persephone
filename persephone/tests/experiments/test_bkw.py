from pathlib import Path

import pytest

from persephone import utils

@pytest.mark.experiment
class TestBKWExperiment:

    @pytest.fixture
    def clean_git(self):
        utils.is_git_directory_clean(".")

    def test_tf_gpu(self):
        import tensorflow as tf
        # Creates a graph.
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(a, b)
        # Creates a session with log_device_placement set to True.
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        # Runs the op.
        print(sess.run(c))

    def test_bkw(self, clean_git):
        """ Trains a multispeaker BKW system using default settings. """
        from persephone.run import prep_exp_dir
        import persephone.datasets.bkw as bkw
        from persephone.corpus_reader import CorpusReader
        from persephone import rnn_ctc
        from persephone import config

        exp_dir = prep_exp_dir(directory=config.TEST_EXP_PATH)
        corp = bkw.Corpus(tgt_dir=Path(config.TEST_DATA_PATH) / "bkw")
        cr = CorpusReader(corp)
        model = rnn_ctc.Model(exp_dir, cr, num_layers=2, hidden_size=250)
        model.train(min_epochs=30)
