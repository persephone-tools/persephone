""" A driver script that runs experiments. """

import os
import shutil

import config
import rnn_ctc
import datasets.na
import datasets.timit
from corpus_reader import CorpusReader

EXP_DIR = config.EXP_DIR

def get_exp_dir_num():
    """ Gets the number of the current experiment directory."""
    return max([int(fn.split(".")[0])
                for fn in os.listdir(EXP_DIR) if fn.split(".")[0].isdigit()])

def prep_exp_dir():
    """ Prepares an experiment directory by copying the code in this directory
    to it as is, and setting the logger to write to files in that
    directory.
    """

    exp_num = get_exp_dir_num()
    exp_num = exp_num + 1
    code_dir = os.path.join(EXP_DIR, str(exp_num), "code")
    os.makedirs(code_dir)
    shutil.copytree(os.getcwd(), code_dir)

    return os.path.join(EXP_DIR, str(exp_num))

def run():
    """ Run an experiment. """

    for i in range(7, 12):
        # Prepares a new experiment dir for all logging.
        exp_dir = prep_exp_dir()

        corpus = datasets.timit.Corpus(feat_type="log_mel_filterbank",
                                       target_type="phonemes")
        corpus_reader = CorpusReader(corpus, num_train=2**i, max_samples=1000)
        model = rnn_ctc.Model(exp_dir, corpus_reader)
        model.train()
