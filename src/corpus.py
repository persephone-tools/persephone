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
    def phonemes_to_indices(self, phonemes):
        """ Converts a sequence of phonemes their corresponding labels. """

    @abc.abstractmethod
    def get_train_fns(self):
        """ Returns a tuple of two elements representing the training set. The
        first element is a list of the filenames of all the input features. The
        second element is a list of the filenames of all the targets. There is
        a one-to-one correspondence between these two lists.
        """

    @abc.abstractmethod
    def get_valid_fns(self):
        """ Returns a tuple of two elements representing the validation set.
        The first element is a list of the filenames of all the input features.
        The second element is a list of the filenames of all the targets. There
        is a one-to-one correspondence between these two lists.
        """

    @abc.abstractmethod
    def get_test_fns(self):
        """ Returns a tuple of two elements representing the test set.
        The first element is a list of the filenames of all the input features.
        The second element is a list of the filenames of all the targets. There
        is a one-to-one correspondence between these two lists.
        """

    @property
    def num_feats(self):
        """ The number of features per time step in the corpus. """
        if not self._num_feats:
            filename = self.get_train_fns()[0][0]
            feats = np.load(filename)
            # pylint: disable=maybe-no-member
            if len(feats.shape) == 3:
                # Then there are multiple channels of multiple feats
                self._num_feats = feats.shape[1] * feats.shape[2]
            else:
                raise Exception(
                    "Feature matrix of shape %s unexpected" % str(feats.shape))
        return self._num_feats
