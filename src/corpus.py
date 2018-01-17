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
        self.label_type = target_type

    def indices_to_labels(self, indices):
        """ Converts a sequence of indices into their corresponding labels."""

        return [(self.INDEX_TO_LABEL[index]) for index in indices]

    def labels_to_indices(self, labels):
        """ Converts a sequence of labels into their corresponding indices."""

        return [self.LABEL_TO_INDEX[label] for label in labels]

    # TODO Remove all calls to these methods:
    def indices_to_phonemes(self, indices):
        return self.indices_to_labels(indices)
    def phonemes_to_indices(self, phonemes):
        return self.labels_to_indices(phonemes)

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

    def get_train_fns(self):
        feat_fns = [os.path.join(self.FEAT_DIR, "%s.%s.npy" % (prefix, self.feat_type))
                    for prefix in self.train_prefixes]
        label_fns = [os.path.join(self.LABEL_DIR, "%s.%s" % (prefix, self.label_type))
                      for prefix in self.train_prefixes]
        return feat_fns, label_fns

    def get_valid_fns(self):
        feat_fns = [os.path.join(self.FEAT_DIR, "%s.%s.npy" % (prefix, self.feat_type))
                    for prefix in self.valid_prefixes]
        label_fns = [os.path.join(self.LABEL_DIR, "%s.%s" % (prefix, self.label_type))
                      for prefix in self.valid_prefixes]
        return feat_fns, label_fns

    def get_test_fns(self):
        feat_fns = [os.path.join(self.FEAT_DIR, "%s.%s.npy" % (prefix, self.feat_type))
                    for prefix in self.test_prefixes]
        label_fns = [os.path.join(self.LABEL_DIR, "%s.%s" % (prefix, self.label_type))
                      for prefix in self.test_prefixes]
        return feat_fns, label_fns

    def get_untranscribed_fns(self):
        feat_fns = [os.path.join(self.UNTRAN_FEAT_DIR, "%s.%s.npy" % (
                    os.path.splitext(fn)[0], self.feat_type))
                    for fn in os.listdir(self.UNTRAN_FEAT_DIR)
                    if fn.endswith(".wav")]
        return feat_fns

class ReadyCorpus(corpus.AbstractCorpus):
    """ Interface to a corpus that has WAV files and label files split into
    utterances and segregated in a directory with a "feat" and "label" dir. """

    def __init__(self, path, feat_type="fbank", label_type="phonemes"):
        super().__init__(feat_type, label_type)

        self.FEAT_DIR = os.path.join(path, "feat")
        self.LABEL_DIR = os.path.join(path, "label")

        if not os.path.isdir(self.FEAT_DIR):
            raise Exception("The supplied path requires a 'feat' subdirectory.")
        if not os.path.isdir(self.LABEL_DIR):
            raise Exception("The supplied path requires a 'label' subdirectory.")

        train, valid, test = 
