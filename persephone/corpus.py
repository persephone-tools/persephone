"""
Describes the abstract Corpus class that all interfaces to corpora should
subclass.
"""

import abc
from collections import namedtuple
import os
from os.path import join
import random

import numpy as np

from . import feat_extract
from . import utils
from .exceptions import PersephoneException

class Corpus:
    """
    All interfaces to corpora are subclasses of this class. A corpus assumes
    that WAV files are already segmented and in the tgt_dir/wav, and that
    labels are in tgt_dir/label. If additional preprocessing is required,
    then subclass Corpus, do what you need to do in __init__(), and then call
    the superclass constructor.
    """

    def __init__(self, feat_type, label_type, tgt_dir, labels,
                 max_samples=1000):
        """ feat_type: A string describing the input features. For
                       example, 'log_mel_filterbank'.
            target_type: A string describing the targets. For example,
                         'phonemes' or 'tones'.
            labels: A set of tokens used in transcription. Typically units such
                    as phonemes or characters.
            max_samples: The maximum length in samples of a train, valid, or
                         test utterance. Used to save memory in exchange for
                         reducing the number of effective training examples.
        """
        self.feat_type = feat_type
        self.label_type = label_type

        # Setting up directories
        self.set_and_check_directories(tgt_dir)

        # Label-related stuff
        self.initialize_labels(labels)

        # This is a lazy function that assumes wavs are already in the WAV dir
        # but only creates features if necessary
        self.prepare_feats()
        self._num_feats = None

        # This is also lazy if the {train,valid,test}_prefixes.txt files exist.
        self.make_data_splits(max_samples=max_samples)

        # Sort the training prefixes by size for more efficient training
        self.train_prefixes = utils.sort_by_size(
            self.feat_dir, self.train_prefixes, feat_type)

        self.untranscribed_prefixes = self.get_untranscribed_prefixes()

    def set_and_check_directories(self, tgt_dir):

        # Set the directory names
        self.tgt_dir = tgt_dir
        self.feat_dir = os.path.join(tgt_dir, "feat")
        self.wav_dir = os.path.join(tgt_dir, "wav")
        self.label_dir = os.path.join(tgt_dir, "label")

        # Check directories exist.
        if not os.path.isdir(tgt_dir):
            raise FileNotFoundError(
                "The directory {} does not exist.".format(tgt_dir))
        if not os.path.isdir(self.wav_dir):
            raise PersephoneException(
                "The supplied path requires a 'wav' subdirectory.")
        if not os.path.isdir(self.feat_dir):
            os.makedirs(self.feat_dir)
        if not os.path.isdir(self.label_dir):
            raise PersephoneException(
                "The supplied path requires a 'label' subdirectory.")

    def initialize_labels(self, labels):
        self.labels = labels
        self.vocab_size = len(self.labels)
        self.LABEL_TO_INDEX = {label: index for index, label in enumerate(
                                 ["pad"] + sorted(list(self.labels)))}
        self.INDEX_TO_LABEL = {index: phn for index, phn in enumerate(
                                 ["pad"] + sorted(list(self.labels)))}

    def indices_to_labels(self, indices):
        """ Converts a sequence of indices into their corresponding labels."""

        return [(self.INDEX_TO_LABEL[index]) for index in indices]

    def labels_to_indices(self, labels):
        """ Converts a sequence of labels into their corresponding indices."""

        return [self.LABEL_TO_INDEX[label] for label in labels]

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
                raise ValueError(
                    "Feature matrix of shape %s unexpected" % str(feats.shape))
        return self._num_feats

    def prefixes_to_fns(self, prefixes):
        feat_fns = [os.path.join(self.feat_dir, "%s.%s.npy" % (prefix, self.feat_type))
                    for prefix in prefixes]
        label_fns = [os.path.join(self.label_dir, "%s.%s" % (prefix, self.label_type))
                      for prefix in prefixes]
        return feat_fns, label_fns

    def get_train_fns(self):
        return self.prefixes_to_fns(self.train_prefixes)

    def get_valid_fns(self):
        return self.prefixes_to_fns(self.valid_prefixes)

    def get_test_fns(self):
        return self.prefixes_to_fns(self.test_prefixes)

    def get_untranscribed_prefixes(self):

        untranscribed_prefix_fn = join(self.tgt_dir, "untranscribed_prefixes.txt")
        if os.path.exists(untranscribed_prefix_fn):
            with open(untranscribed_prefix_fn) as f:
                prefixes = f.readlines()

            return [prefix.strip() for prefix in prefixes]

        return None

    def get_untranscribed_fns(self):
        feat_fns = [os.path.join(self.feat_dir, "%s.%s.npy" % (prefix, self.feat_type))
                    for prefix in self.untranscribed_prefixes]
        return feat_fns

    def prepare_feats(self):
        """ Prepares input features"""

        if not os.path.isdir(self.feat_dir):
            os.makedirs(self.feat_dir)

        should_extract_feats = False
        for fn in os.listdir(self.wav_dir):
            path = join(self.wav_dir, fn)
            if not path.endswith(".wav"):
                continue
            prefix = os.path.basename(os.path.splitext(path)[0])
            mono16k_wav_path = join(self.feat_dir, "%s.wav" % prefix)
            feat_path = join(self.feat_dir,
                             "%s.%s.npy" % (prefix, self.feat_type))
            if not os.path.isfile(feat_path):
                # Then we should extract feats
                should_extract_feats = True
                if not os.path.isfile(mono16k_wav_path):
                    feat_extract.convert_wav(path, mono16k_wav_path)

        if should_extract_feats:
            feat_extract.from_dir(self.feat_dir, self.feat_type)

    def get_prefixes(self):
        fns = [fn for fn in os.listdir(self.label_dir)
               if fn.endswith(self.label_type)]
        prefixes = [os.path.splitext(fn)[0] for fn in fns]
        return prefixes

    def make_data_splits(self, max_samples=1000, seed=0):
        """ Splits the utterances into training, validation and test sets."""

        train_prefix_fn = join(self.tgt_dir, "train_prefixes.txt")
        valid_prefix_fn = join(self.tgt_dir, "valid_prefixes.txt")
        test_prefix_fn = join(self.tgt_dir, "test_prefixes.txt")

        train_f_exists = os.path.isfile(train_prefix_fn)
        valid_f_exists = os.path.isfile(valid_prefix_fn)
        test_f_exists = os.path.isfile(test_prefix_fn)

        if train_f_exists and valid_f_exists and test_f_exists:
            with open(train_prefix_fn) as train_f:
                train_prefixes = [line.strip() for line in train_f]
            with open(valid_prefix_fn) as valid_f:
                valid_prefixes = [line.strip() for line in valid_f]
            with open(test_prefix_fn) as test_f:
                test_prefixes = [line.strip() for line in test_f]
        else:
            prefixes = self.get_prefixes()
            prefixes = utils.filter_by_size(
                self.feat_dir, prefixes, self.feat_type, max_samples)
            Ratios = namedtuple("Ratios", ["train", "valid", "test"])
            # TODO These ratios can't be hardcoded
            ratios = Ratios(.90, .05, .05)
            train_end = int(ratios.train*len(prefixes))
            valid_end = int(train_end + ratios.valid*len(prefixes))
            random.shuffle(prefixes)
            train_prefixes = prefixes[:train_end]
            valid_prefixes = prefixes[train_end:valid_end]
            test_prefixes = prefixes[valid_end:]

            # TODO Perhaps the ReadyCorpus train_prefixes variable should be a
            # property that writes this file when it is changed, then we can
            # remove the code from here. The thing is, the point of those files
            # is that they don't change, so they should be written once only,
            # unless you explicitly delete them. This isn't very clear from the
            # perspective of the user though, so it's a design decision to
            # think about.
            if train_prefixes:
                with open(train_prefix_fn, "w") as train_f:
                    for prefix in train_prefixes:
                        print(prefix, file=train_f)
            if valid_prefixes:
                with open(valid_prefix_fn, "w") as dev_f:
                    for prefix in valid_prefixes:
                        print(prefix, file=dev_f)
            if test_prefixes:
                with open(test_prefix_fn, "w") as test_f:
                    for prefix in test_prefixes:
                        print(prefix, file=test_f)

        if train_prefixes == []:
            # TODO log this as a warning
            print("""WARNING: Corpus object has no training data. Are you sure
            it's in the correct directories? WAVs should be in {} and
            transcriptions in {} with the extension .{}""".format(
                self.wav_dir, self.label_dir, self.label_type))

        self.train_prefixes = train_prefixes
        self.valid_prefixes = valid_prefixes
        self.test_prefixes = test_prefixes

class ReadyCorpus(Corpus):
    """ Interface to a corpus that has WAV files and label files split into
    utterances and segregated in a directory with a "wav" and "label" dir. """

    def __init__(self, tgt_dir, feat_type="fbank", label_type="phonemes"):

        labels = self.determine_labels(tgt_dir, label_type)

        super().__init__(feat_type, label_type, tgt_dir, labels)

    @staticmethod
    def determine_labels(tgt_dir, label_type):
        """ Returns a set of phonemes found in the corpus. """

        label_dir = os.path.join(tgt_dir, "label/")
        if not os.path.isdir(label_dir):
            raise FileNotFoundError(
                "The directory {} does not exist.".format(tgt_dir))

        phonemes = set()
        for fn in os.listdir(label_dir):
            if fn.endswith(label_type):
                with open(join(label_dir, fn)) as f:
                    try:
                        line_phonemes = set(f.readline().split())
                    except UnicodeDecodeError:
                        print("Unicode decode error on file {}".format(fn))
                        raise
                    phonemes = phonemes.union(line_phonemes)
        return phonemes
