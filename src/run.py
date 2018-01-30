""" A driver script that runs experiments. """

import logging
import os
from os.path import join
import shutil
import sys

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

def get_exp_dir_num(parent_dir):
    """ Gets the number of the current experiment directory."""
    return max([int(fn.split(".")[0])
                for fn in os.listdir(parent_dir) if fn.split(".")[0].isdigit()]
                    + [-1])

def prep_sub_exp_dir(parent_dir):
    """ Prepares an experiment directory by copying the code in this directory
    to it as is, and setting the logger to write to files in that
    directory.
    """

    exp_num = get_exp_dir_num(parent_dir)
    exp_num = exp_num + 1
    exp_dir = os.path.join(parent_dir, str(exp_num))
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)

    return exp_dir

def prep_exp_dir():
    """ Prepares an experiment directory by copying the code in this directory
    to it as is, and setting the logger to write to files in that
    directory.
    """

    exp_num = get_exp_dir_num(EXP_DIR)
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
    exp_dir = prep_exp_dir()
    scaling_graph(exp_dir, num_train=512)

def scaling_graph(exp_dir, num_train=None):

    num_runs = 3

    #feat_types = ["fbank_and_pitch", "fbank"]
    #labels = ["phonemes", "tones", "phonemes_and_tones"]
    #for feat_type in feat_types:
    #    for label_type in labels:
    #        for i in range(num_runs):
    #            train(exp_dir, "na", feat_type, label_type, 3, 250,
    #                  train_rec_type="text", num_train=num_train)

    #feat_type = "phonemes_onehot"
    #label_type = "tones"
    #for i in range(num_runs):
    #    train(exp_dir, "na", feat_type, label_type, 3, 250,
    #          train_rec_type="text", num_train=num_train)

    feat_type = "pitch"
    label_type = "tones_notgm"
    for i in range(num_runs):
        train(exp_dir, "na", feat_type, label_type, 3, 250,
              train_rec_type="text", num_train=num_train)

def story_fold_cross_validation(exp_dir):

    with open(join(exp_dir, "storyfold_crossvalidation.txt"), "w") as f:
        texts = list(datasets.na.get_stories())
        for i, test_text in enumerate(texts):
            valid_text = texts[(i+1) % len(texts)]
            for out_f in [f, sys.stdout]:
                print(i, file=out_f)
                print("Test text: %s" % test_text, file=out_f)
                print("Valid text: %s" % valid_text, file=out_f)
                print("", file=out_f, flush=True)

            train(exp_dir, "na", "fbank_and_pitch", "phonemes_and_tones", 3, 400,
                   valid_story=valid_text, test_story=test_text)

def train(exp_dir, language, feat_type, label_type,
          num_layers, hidden_size,
          num_train=None, batch_size=64,
          train_rec_type="text_and_wordlist",
          valid_story=None, test_story=None,
          min_epochs=30):
    """ Run an experiment. """

    sub_exp_dir = prep_sub_exp_dir(exp_dir)
    print(sub_exp_dir)

    ## get TF logger
    #log = logging.getLogger('tensorflow')
    #log.setLevel(logging.DEBUG)
    ## create formatter and add it to the handlers
    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ## create file handler which logs even debug messages
    #fh = logging.FileHandler(os.path.join(sub_exp_dir, 'tensorflow.log'))
    #fh.setLevel(logging.DEBUG)
    #fh.setFormatter(formatter)
    #log.addHandler(fh)

    if language == "chatino":
        corpus = datasets.chatino.Corpus(feat_type, label_type)
    elif language == "na":
        corpus = datasets.na.Corpus(feat_type, label_type,
                                    train_rec_type=train_rec_type,
                                    valid_story=valid_story,
                                    test_story=test_story)
    elif language == "kunwinjku":
        # TODO How to choose between the Bird and Butcher corpora?
        corpus = datasets.kunwinjku.Corpus(feat_type, label_type)
    else:
        raise Exception("Language '%s' not supported." % language)

    if num_train:
        corpus_reader = CorpusReader(corpus, num_train=num_train, batch_size=batch_size)
    else:
        corpus_reader = CorpusReader(corpus, batch_size=batch_size)
    print(corpus_reader)
    model = rnn_ctc.Model(sub_exp_dir, corpus_reader,
                          num_layers=num_layers,
                          hidden_size=hidden_size,
                          decoding_merge_repeated=(False if
                                                   label_type=="tones"
                                                   else True))
    model.train(min_epochs=min_epochs)

    try:
        with open(os.path.join(sub_exp_dir, "train_desc2.txt"), "w") as desc_f:
            for f in [desc_f, sys.stdout]:
                print("language: %s" % language, file=f)
                print("feat_type: %s" % feat_type, file=f)
                print("label_type: %s" % label_type, file=f)
                print("train_rec_type: %s" % train_rec_type, file=f)
                print("num_layers: %d" % num_layers, file=f)
                print("hidden_size: %d" % hidden_size, file=f)
                if num_train:
                    print("num_train: %d" % num_train, file=f)
                print("train duration: %f" % corpus_reader.calc_time(), file=f)
                print("batch_size: %d" % batch_size, file=f)
                print("Exp dir:", sub_exp_dir, file=f)
    except:
        print("Issues with my printing train_desc2")

def train_babel():
    # Prepares a new experiment dir for all logging.
    exp_dir = prep_exp_dir()
    corpus = datasets.babel.Corpus(["turkish"])
    corpus_reader = CorpusReader(corpus, num_train=len(corpus.get_train_fns()), batch_size=128)
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
