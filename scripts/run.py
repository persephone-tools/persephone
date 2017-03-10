""" A driver script that runs experiments. """

import os
import shutil

import config
import rnn_ctc
import timit

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

    for i in range(10,11):
        # Prepares a new experiment dir for all logging.
        exp_dir = prep_exp_dir()

        corpus_batches = timit.CorpusBatches(feat_type="mfcc13_d",
                batch_size=64, total_size=2**i)

        model = rnn_ctc.Model(exp_dir, corpus_batches)
        model.train(corpus_batches, num_epochs=100, save_n=25)
