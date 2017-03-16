""" Describes the abstract Corpus class that all interfaces to corpora should
    subclass. """

import abc
import os

def get_prefixes(set_dir):
    """ Returns a list of prefixes to files in the set (which might be a whole
    corpus, or a train/valid/test subset. The prefixes include the path leading
    up to it, but remove only the file extension.
    """

    prefixes = []
    for root, _, filenames in os.walk(set_dir):
        for filename in filenames:
            if filename.endswith(".npy"):
                # Then it's an input feature file and its prefix will
                # correspond to a training example
                prefixes.append(os.path.join(root, filename))
    return sorted(prefixes)

class Corpus(metaclass=abc.ABCMeta):
    "All interfaces to corpora are subclasses of this class."""

    feat_type = None
    target_type = None
    vocab_size = None
    num_feats = None

    def __init__(self, feat_type, target_type):
        """ feat_type: A string describing the input features. For
                       example, 'log_mel_filterbank'.
            target_type: A string describing the targets. For example,
                         'phonemes' or 'syllables'.
        """
        self.feat_type = feat_type
        self.target_type = target_type

    @abc.abstractmethod
    def prepare(self):
        """ Performs preprocessing of the raw corpus data to make it ready
        for model training.
        """

    @abc.abstractmethod
    def indices_to_phonemes(self, indices):
        """ Converts a sequence of indices representing labels into their
        corresponding (possibly collapsed) phonemes.
        """

    @abc.abstractmethod
    def get_train_prefixes(self):
        """ Returns the prefixes of all training instances."""

    @abc.abstractmethod
    def get_valid_prefixes(self):
        """ Returns the prefixes of all validation instances."""

    @abc.abstractmethod
    def get_test_prefixes(self):
        """ Returns the prefixes of all test instances."""
