""" Describes the abstract Corpus class that all interfaces to corpora should
    subclass. """

import abc

import numpy as np

import utils

class AbstractCorpus(metaclass=abc.ABCMeta):
    "All interfaces to corpora are subclasses of this class."""

    feat_type = None
    target_type = None
    vocab_size = None
    _num_feats = None

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

    def sort_and_filter_by_size(self, prefixes, max_samples):
        """ Sorts the files by their length and returns those with less
        than or equal to max_samples length. Returns the filename prefixes of
        those files. The main job of the method is to filter, but the sorting
        may give better efficiency when doing dynamic batching unless it gets
        shuffled downstream.

            prefixes: The prefix of the filenames, with the complete path. For
            example "/home/username/data/corpus/audiofile" minus the extension.
        """

        prefix_lens = []
        for prefix in prefixes:
            path = "%s.%s.npy" % (prefix, self.feat_type)
            _, batch_x_lens = utils.load_batch_x([path], flatten=True)
            prefix_lens.append((prefix, batch_x_lens[0]))
        prefix_lens.sort(key=lambda prefix_len: prefix_len[1])
        prefixes = [prefix for prefix, length in prefix_lens
                    if length <= max_samples]
        return prefixes

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
