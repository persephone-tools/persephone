""" A driver script that runs experiments. """

import os
import shutil

import config
import rnn_ctc
import datasets.na
import datasets.timit

EXP_DIR = config.EXP_DIR

def get_exp_dir_num():
    """ Gets the number of the current experiment directory."""
    return max([int(fn.split(".")[0]) for fn in os.listdir(EXP_DIR) if fn.split(".")[0].isdigit()])

def prep_exp_dir():
    """ Prepares an experiment directory by copying the code in this directory
    to it as is, and setting the logger to write to files in that
    directory.
    """

    exp_num = get_exp_dir_num()
    exp_num = exp_num + 1
    code_dir = os.path.join(EXP_DIR, str(exp_num), "code")
    os.makedirs(code_dir)
    for filename in os.listdir():
        if filename.endswith(".py"):
            shutil.copyfile(filename, os.path.join(code_dir, filename))

    return os.path.join(EXP_DIR, str(exp_num))

def run():
    """ Run an experiment.  """

    for i in range(7, 10):
        # Prepares a new experiment dir for all logging.
        exp_dir = prep_exp_dir()

        corpus_batches = datasets.na.CorpusBatches(
            feat_type="log_mel_filterbank", seg_type="phonemes",
            batch_size=8, total_size=2**i, max_samples=1000)

        model = rnn_ctc.Model(exp_dir, corpus_batches)
        model.train(corpus_batches)

    for i in range(9, 12):
        # Prepares a new experiment dir for all logging.
        exp_dir = prep_exp_dir()

        corpus_batches = datasets.na.CorpusBatches(
            feat_type="log_mel_filterbank", seg_type="phonemes",
            batch_size=64, total_size=2**i, max_samples=1000)

        model = rnn_ctc.Model(exp_dir, corpus_batches)
        model.train(corpus_batches)

def timit_test(dir_num):
    """ Tests the model in dir_num on the TIMIT test set."""

    import tensorflow as tf
    import importlib

    path = os.path.join(EXP_DIR, str(dir_num))
    model_path = os.path.join(path, "model", "model_best.ckpt")
    code_path = os.path.join(path, "code", "rnn_ctc.py")
    exec(code_path)

    #saver = tf.train.Saver()
    #with tf.Session() as sess:
    #    saver.restore(sess, path)
