""" Describes the abstract Corpus class that all interfaces to corpora should
    subclass. """

import abc

import numpy as np

class AbstractCorpus(metaclass=abc.ABCMeta):
    "All interfaces to corpora are subclasses of this class."""

    feat_type = None
    target_type = None
    vocab_size = None
    _num_feats = None

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

    @property
    def num_feats(self):
        """ The number of features per time step in the corpus. """
        if not self._num_feats:
            filename = "%s.%s.npy" % (self.get_train_prefixes()[0],
                                      self.feat_type)
            feats = np.load(filename)
            # pylint: disable=maybe-no-member
            if len(feats.shape) == 3:
                # Then there are multiple channels of multiple feats
                self._num_feats = feats.shape[1] * feats.shape[2]
            else:
                raise Exception(
                    "Feature matrix of shape %s unexpected" % str(feats.shape))
        return self._num_feats
