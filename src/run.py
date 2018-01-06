""" A driver script that runs experiments. """

import os
import shutil

from git import Repo

import config
import rnn_ctc
import datasets.na
#import datasets.griko
import datasets.chatino
#import datasets.timit
#import datasets.japhug
import datasets.babel
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
    exp_dir = os.path.join(EXP_DIR, str(exp_num))
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
    repo = Repo(".", search_parent_directories=True)
    with open(os.path.join(exp_dir, "git_hash.txt"), "w") as f:
        print("SHA1 hash: {hexsha}".format(hexsha=repo.head.commit.hexsha), file=f)

    return exp_dir

class DirtyRepoException(Exception):
    pass

def run():
    """
    The only function that should be called from this module. Ensures
    experiments are documented in their own dir with a reference to the hash
    of the commit used to run the experiment.
    """

    repo = Repo(".", search_parent_directories=True)
    if repo.untracked_files != []:
        raise DirtyRepoException("Untracked files. Commit them first.")
    # If there are changes to already tracked files
    if repo.is_dirty():
        raise DirtyRepoException("Changes to the index or working tree."
                                 "Commit them first .")

    prep_exp_dir()


def multi_train():
    #train("na", "fbank", "phonemes_and_tones", 3, 250,
    #      train_rec_type="text_and_wordlist")
    train("na", "fbank", "phonemes_and_tones", 3, 250,
          train_rec_type="text")
    train("na", "fbank", "phonemes_and_tones", 3, 250,
          train_rec_type="wordlist")
    #train("na", "fbank_and_pitch", "phonemes_and_tones", 3, 400,
    #      train_rec_type="text_and_wordlist", batch_size=32)

def train(language, feat_type, label_type,
          num_layers, hidden_size,
          num_train=None, batch_size=64,
          train_rec_type="text_and_wordlist"):
    """ Run an experiment. """

    if language == "chatino":
        corpus = datasets.chatino.Corpus(feat_type, label_type)
    elif language == "na":
        corpus = datasets.na.Corpus(feat_type, label_type,
                                    train_rec_type=train_rec_type)
    else:
        raise Exception("Language '%s' not supported." % language)

    # Prepares a new experiment dir for all logging.
    exp_dir = prep_exp_dir()
    if num_train:
        corpus_reader = CorpusReader(corpus, num_train=num_train, batch_size=batch_size)
    else:
        corpus_reader = CorpusReader(corpus, batch_size=batch_size)
    model = rnn_ctc.Model(exp_dir, corpus_reader,
                          num_layers=num_layers,
                          hidden_size=hidden_size,
                          decoding_merge_repeated=(False if
                                                   label_type=="tones"
                                                   else True))
    model.train()

    print("language: %s" % language)
    print("feat_type: %s" % feat_type)
    print("label_type: %s" % label_type)
    print("train_rec_type: %s" % train_rec_type)
    print("num_layers: %d" % num_layers)
    print("hidden_size: %d" % hidden_size)
    if num_train:
        print("num_train: %d" % num_train)
    print("batch_size: %d" % batch_size)
    print("Exp dir:", exp_dir)

def train_babel():
    # Prepares a new experiment dir for all logging.
    exp_dir = prep_exp_dir()
    corpus = datasets.babel.Corpus(["turkish"])
    corpus_reader = CorpusReader(corpus, num_train=len(corpus.get_train_fns()), batch_size=128)
    model = rnn_ctc.Model(exp_dir, corpus_reader, num_layers=3)
    model.train()


def calc_time():
    """ Calculates the total spoken time a given number of utterances
    corresponds to. """

    import numpy as np

    #for i in [128,256,512,1024,2048]:
    for i in [7420]:
        corpus = datasets.na.Corpus(feat_type="fbank",
                                         label_type="phonemes")
        #corpus_reader = CorpusReader(corpus, num_train=i)

        #print(len(corpus_reader.train_fns))

        total_frames = 0
        #for feat_fn in corpus.get_train_fns()[0]:
        #    frames = len(np.load(feat_fn))
        #    total_frames += frames
        #for feat_fn in corpus.get_valid_fns()[0]:
        #    frames = len(np.load(feat_fn))
        #    total_frames += frames
        for feat_fn in corpus.get_test_fns()[0]:
            frames = len(np.load(feat_fn))
            total_frames += frames

        total_time = ((total_frames*10)/1000)/60
        print(total_time)
        print("%0.3f minutes." % total_time)

def train_japhug():
    """ Run an experiment. """

    #for i in [128,256,512,1024, 2048]:
    for i in [800]:
        # Prepares a new experiment dir for all logging.
        exp_dir = prep_exp_dir()

        corpus = datasets.japhug.Corpus(feat_type="log_mel_filterbank",
                                    target_type="phn", normalize=True)
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

def produce_chatino_lattices():
    """ Apply a previously trained model to some test data. """
    exp_dir = prep_exp_dir()
    corpus = datasets.chatino.Corpus(feat_type="log_mel_filterbank",
                                target_type="phn", tones=False)
    corpus_reader = CorpusReader(corpus, num_train=2048)
    model = rnn_ctc.Model(exp_dir, corpus_reader)
    restore_model_path = os.path.join(
        EXP_DIR, "194", "model", "model_best.ckpt")
    model.output_lattices(corpus_reader.valid_batch(), restore_model_path)

def produce_na_lattices():
    """ Apply a previously trained model to some test data. """
    exp_dir = prep_exp_dir()
    corpus = datasets.na.Corpus(feat_type="log_mel_filterbank",
                                target_type="phn", tones=True)
    corpus_reader = CorpusReader(corpus, num_train=2048)
    model = rnn_ctc.Model(exp_dir, corpus_reader)
    restore_model_path = os.path.join(
        EXP_DIR, "131", "model", "model_best.ckpt")
    model.output_lattices(corpus_reader.valid_batch(), restore_model_path)

def transcribe():
    """ Applies a trained model to the untranscribed Na data for Alexis. """

    exp_dir = prep_exp_dir()
    corpus = datasets.na.Corpus(feat_type="fbank_and_pitch",
                                label_type="phonemes_and_tones")
    corpus_reader = CorpusReader(corpus, num_train=2048)
    model = rnn_ctc.Model(exp_dir, corpus_reader)
    #print(corpus_reader.untranscribed_batch())

    # Model 155 is the first Na ASR model used to give transcriptions to
    # Alexis Michaud
    restore_model_path = os.path.join(
        EXP_DIR, "552", "model", "model_best.ckpt")

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
