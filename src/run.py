""" A driver script that runs experiments. """

import os
import shutil

import config
import rnn_ctc
import datasets.na
import datasets.griko
import datasets.chatino
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
    shutil.copytree(os.getcwd(), code_dir)

    return os.path.join(EXP_DIR, str(exp_num))

def train():
    """ Run an experiment. """

    for i in [128,256,512,1024,2048]:
        # Prepares a new experiment dir for all logging.
        exp_dir = prep_exp_dir()

        corpus = datasets.na.Corpus(feat_type="log_mel_filterbank",
                                    target_type="phn", tones=True)
        corpus_reader = CorpusReader(corpus, num_train=i)
        model = rnn_ctc.Model(exp_dir, corpus_reader, num_layers=3)
        model.train()

def train_chatino():
    """ Run an experiment. """

    #for i in [128,256,512,1024, 2048]:
    for i in [2048]:
        # Prepares a new experiment dir for all logging.
        exp_dir = prep_exp_dir()

        corpus = datasets.chatino.Corpus(feat_type="log_mel_filterbank",
                                    target_type="phn", tones=False)
        corpus_reader = CorpusReader(corpus, num_train=i)
        model = rnn_ctc.Model(exp_dir, corpus_reader, num_layers=3)
        model.train()

def test():
    """ Apply a previously trained model to some test data. """
    exp_dir = prep_exp_dir()
    corpus = datasets.na.Corpus(feat_type="log_mel_filterbank",
                                target_type="phn", tones=True)
    corpus_reader = CorpusReader(corpus, num_train=2048)
    model = rnn_ctc.Model(exp_dir, corpus_reader)
    restore_model_path = os.path.join(
        EXP_DIR, "131", "model", "model_best.ckpt")
    model.eval(restore_model_path)

def transcribe():
    """ Applies a trained model to the untranscribed Na data for Alexis. """

    exp_dir = prep_exp_dir()
    corpus = datasets.na.Corpus(feat_type="log_mel_filterbank",
                                target_type="phn", tones=True)
    corpus_reader = CorpusReader(corpus, num_train=2048)
    model = rnn_ctc.Model(exp_dir, corpus_reader)
    #print(corpus_reader.untranscribed_batch())

    # Model 155 is the first Na ASR model used to give transcriptions to
    # Alexis Michaud
    restore_model_path = os.path.join(
        EXP_DIR, "155", "model", "model_best.ckpt")

    #model.eval(restore_model_path, corpus_reader.)
    model.transcribe(restore_model_path)

def train_griko():

    # Prepares a new experiment dir for all logging.
    exp_dir = prep_exp_dir()

    corpus = datasets.griko.Corpus(feat_type="log_mel_filterbank",
                                   target_type="char")
    corpus_reader = CorpusReader(corpus, num_train=256)
    model = rnn_ctc.Model(exp_dir, corpus_reader)
    model.train()

def test_griko():
    # Prepares a new experiment dir for all logging.
    exp_dir = prep_exp_dir()

    corpus = datasets.griko.Corpus(feat_type="log_mel_filterbank",
                                   target_type="char")
    corpus_reader = CorpusReader(corpus, num_train=2048)
    model = rnn_ctc.Model(exp_dir, corpus_reader)
    restore_model_path = os.path.join(
        EXP_DIR, "164", "model", "model_best.ckpt")
    model.eval(restore_model_path)
