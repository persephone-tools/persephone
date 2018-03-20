import logging
import os
from os.path import join
import shutil
import sys

import git
from git import Repo

import persephone
from . import config
from . import rnn_ctc
from .datasets import na
from .corpus_reader import CorpusReader
from .exceptions import PersephoneException, DirtyRepoException
from .utils import is_git_directory_clean

EXP_DIR = config.EXP_DIR

def get_exp_dir_num(parent_dir):
    """ Gets the number of the current experiment directory."""
    return max([int(fn.split(".")[0])
                for fn in os.listdir(parent_dir) if fn.split(".")[0].isdigit()]
                    + [-1])

def _prepare_directory(directory_path):
    """
    Prepare the directory structure required for the experiement
    :returns: returns the name of the newly created directory
    """
    exp_num = get_exp_dir_num(directory_path)
    exp_num = exp_num + 1
    exp_dir = os.path.join(directory_path, str(exp_num))
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
    return exp_dir

def prep_sub_exp_dir(parent_dir):
    """ Prepares an experiment subdirectory
    :parent_dir: the parent directory
    :returns: returns the name of the newly created subdirectory
    """
    return _prepare_directory(parent_dir)

def prep_exp_dir(directory=EXP_DIR):
    """ Prepares an experiment directory by copying the code in this directory
    to it as is, and setting the logger to write to files in that directory.
    Copies a git hash of the most changes at git HEAD into the directory to
    keep the experiment results in sync with the version control system.
    :directory: The path to directory we are preparing for the experiment,
                which will be created if it does not currently exist.
    :returns: The name of the newly created experiment directory.
    """
    if not os.path.isdir(directory):
        os.makedirs(directory)
    exp_dir = _prepare_directory(directory)
    try:
        # Get the directory this file is in, so we can grab the git repo.
        dirname = os.path.dirname(os.path.realpath(__file__))
        repo = Repo(dirname, search_parent_directories=True)
        with open(os.path.join(exp_dir, "git_hash.txt"), "w") as f:
            print("SHA1 hash: {hexsha}".format(hexsha=repo.head.commit.hexsha), file=f)
    except git.exc.InvalidGitRepositoryError: # pylint: disable=no-member
        # Then the package was probably installed via pypi. Get the version
        # number instead.
        with open(os.path.join(exp_dir, "version.txt"), "w") as f:
            print("Persphone version {}".format(persephone.__version__), file=f)

    return exp_dir

def run():
    """
    Ensures experiments are documented in their own dir with a reference to the hash
    of the commit used to run the experiment.
    """
    is_git_directory_clean(".")
    exp_dir = prep_exp_dir()

def train(exp_dir, language, feat_type, label_type,
          num_layers, hidden_size,
          num_train=None,
          train_rec_type="text_and_wordlist",
          valid_story=None, test_story=None,
          min_epochs=30, max_valid_ler=1.0, max_train_ler=0.8):
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
        raise NotImplementedError("Chatino code needs a rewrite.")
    elif language == "na":
        corpus = na.Corpus(feat_type, label_type,
                                    train_rec_type=train_rec_type,
                                    valid_story=valid_story,
                                    test_story=test_story)
    elif language == "kunwinjku":
        # TODO How to choose between the Bird and Butcher corpora?
        raise NotImplementedError("Need to finish testing.")
        #corpus = kunwinjku_steven.Corpus(feat_type, label_type)
    else:
        raise PersephoneException("Language '%s' not supported." % language)


    if num_train:
        corpus_reader = CorpusReader(corpus, num_train=num_train)
    else:
        corpus_reader = CorpusReader(corpus)
    print(corpus_reader)
    model = rnn_ctc.Model(sub_exp_dir, corpus_reader,
                          num_layers=num_layers,
                          hidden_size=hidden_size,
                          decoding_merge_repeated=(False if
                                                   label_type=="tones"
                                                   else True))
    model.train(min_epochs=min_epochs,
                max_valid_ler=max_valid_ler,
                max_train_ler=max_train_ler)

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
                print("batch_size: %d" % corpus_reader.batch_size, file=f)
                print("Exp dir:", sub_exp_dir, file=f)
    except:
        print("Issues with my printing train_desc2")


def get_simple_model(exp_dir, corpus):
    num_layers = 2
    hidden_size= 250

    def decide_batch_size(num_train):

        if num_train >= 512:
            batch_size = 16
        elif num_train < 128:
            if num_train < 4:
                batch_size = 1
            else:
                batch_size = 4
        else:
            batch_size = num_train / 32

        return batch_size

    batch_size = decide_batch_size(len(corpus.train_prefixes))

    corpus_reader = CorpusReader(corpus, batch_size=batch_size)
    model = rnn_ctc.Model(exp_dir, corpus_reader,
                          num_layers=num_layers,
                          hidden_size=hidden_size,
                          decoding_merge_repeated=True)

    return model

def train_ready(corpus, directory=EXP_DIR):

    print(directory)

    exp_dir = prep_exp_dir(directory=directory)
    model = get_simple_model(exp_dir, corpus)
    model.train(min_epochs=20, early_stopping_steps=3)
    return exp_dir

def transcribe(model_path, corpus):
    """ Applies a trained model to untranscribed data in a Corpus. """

    exp_dir = prep_exp_dir()
    model = get_simple_model(exp_dir, corpus)
    model.transcribe(model_path)
