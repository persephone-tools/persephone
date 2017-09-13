""" Describes the abstract Corpus class that all interfaces to corpora should
    subclass. """

import abc
import os

import numpy as np

import utils

class AbstractCorpus(metaclass=abc.ABCMeta):
    "All interfaces to corpora are subclasses of this class."""

    feat_type = None
    target_type = None
    vocab_size = None
    _num_feats = None
    TGT_DIR = None
    INDEX_TO_PHONEME = None
    PHONEME_TO_INDEX = None
    train_prefixes = None
    valid_prefixes = None
    test_prefixes = None
    normalized = False

    def get_target_prefix(self, prefix):
        """ Gets the target gfn given a prefix. """

        prefix = os.path.basename(prefix)
        return os.path.join(self.TGT_DIR, "transcriptions", "utterances", prefix)

    def __init__(self, feat_type, target_type, max_samples=None):
        """ feat_type: A string describing the input features. For
                       example, 'log_mel_filterbank'.
            target_type: A string describing the targets. For example,
                         'phonemes' or 'syllables'.
            max_samples: The maximum length in samples of a train, valid, or
                         test utterance. Used to save memory in exchange for
                         reducing the number of effective training examples.
        """
        self.feat_type = feat_type
        self.target_type = target_type

    def indices_to_phonemes(self, indices):
        """ Converts a sequence of indices representing labels into their
        corresponding (possibly collapsed) phonemes.
        """

        return [(self.INDEX_TO_PHONEME[index]) for index in indices]

    def phonemes_to_indices(self, phonemes):
        """ Converts a sequence of phonemes their corresponding labels. """

        return [self.PHONEME_TO_INDEX[phoneme] for phoneme in phonemes]

    def get_train_fns(self):
        """ Returns a tuple of two elements representing the training set. The
        first element is a list of the filenames of all the input features. The
        second element is a list of the filenames of all the targets. There is
        a one-to-one correspondence between these two lists.
        """

        feat_fns = ["%s.%s.npy" % (prefix, self.feat_type)
                    for prefix in self.train_prefixes]
        if self.normalized:
            feat_fns = [os.path.splitext(fn)[0] + ".norm" + os.path.splitext(fn)[1]
                        for fn in feat_fns]
        target_fns = ["%s.%s" % (self.get_target_prefix(prefix), self.target_type)
                      for prefix in self.train_prefixes]
        return feat_fns, target_fns

    def get_valid_fns(self):
        """ Returns a tuple of two elements representing the validation set.
        The first element is a list of the filenames of all the input features.
        The second element is a list of the filenames of all the targets. There
        is a one-to-one correspondence between these two lists.
        """

        feat_fns = ["%s.%s.npy" % (prefix, self.feat_type)
                    for prefix in self.valid_prefixes]
        if self.normalized:
            feat_fns = [os.path.splitext(fn)[0] + ".norm" + os.path.splitext(fn)[1]
                        for fn in feat_fns]
        target_fns = ["%s.%s" % (self.get_target_prefix(prefix), self.target_type)
                      for prefix in self.valid_prefixes]
        return feat_fns, target_fns
    def get_test_fns(self):
        """ Returns a tuple of two elements representing the test set.
        The first element is a list of the filenames of all the input features.
        The second element is a list of the filenames of all the targets. There
        is a one-to-one correspondence between these two lists.
        """

        feat_fns = ["%s.%s.npy" % (prefix, self.feat_type)
                    for prefix in self.test_prefixes]
        if self.normalized:
            feat_fns = [os.path.splitext(fn)[0] + ".norm" + os.path.splitext(fn)[1]
                        for fn in feat_fns]
        target_fns = ["%s.%s" % (self.get_target_prefix(prefix), self.target_type)
                      for prefix in self.test_prefixes]
        return feat_fns, target_fns

#    @abc.abstractmethod
#    def prepare(self):
        """ Performs preprocessing of the raw corpus data to make it ready
        for model training.
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
            elif len(feats.shape) == 2:
                # Otherwise it is just of shape time x feats
                self._num_feats = feats.shape[1]
            else:
                raise Exception(
                    "Feature matrix of shape %s unexpected" % str(feats.shape))
        return self._num_feats
