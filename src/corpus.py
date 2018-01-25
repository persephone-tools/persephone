""" Describes the abstract Corpus class that all interfaces to corpora should
    subclass. """

import abc
from collections import namedtuple
import os
from os.path import join
import random

import numpy as np

import feat_extract
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
    FEAT_DIR = None
    LABEL_DIR = None

    def get_target_prefix(self, prefix):
        """ Gets the target gfn given a prefix. """

        prefix = os.path.basename(prefix)
        return os.path.join(self.TGT_DIR, "transcriptions", "utterances", prefix)

    def __init__(self, feat_type, target_type, max_samples=None):
        """ feat_type: A string describing the input features. For
                       example, 'log_mel_filterbank'.
            target_type: A string describing the targets. For example,
                         'phonemes' or 'tones'.
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

    def get_untranscribed_prefixes(self):

        untranscribed_prefix_fn = join(self.TGT_DIR, "untranscribed_prefixes.txt")
        if os.path.exists(untranscribed_prefix_fn):
            with open(untranscribed_prefix_fn) as f:
                prefixes = f.readlines()

            return [prefix.strip() for prefix in prefixes]

        return None

    def get_untranscribed_fns(self):
        feat_fns = [os.path.join(self.FEAT_DIR, "%s.%s.npy" % (prefix, self.feat_type))
                    for prefix in self.untranscribed_prefixes]
        return feat_fns

        # TODO Figure out how to interface with untranscribed files.
        # Temporarily just using the validation set for testing
        #return self.get_valid_fns()[0]

        #feat_fns = [os.path.join(self.UNTRAN_FEAT_DIR, "%s.%s.npy" % (
        #            os.path.splitext(fn)[0], self.feat_type))
        #            for fn in os.listdir(self.UNTRAN_FEAT_DIR)
        #            if fn.endswith(".wav")]
        #return feat_fns

    def calc_time(self):
        """
        Prints statistics about the the total duration of recordings in the
        corpus.
        """

        def get_number_of_frames(feat_fns):
            """ fns: A list of numpy files which contain a number of feature
            frames. """

            total = 0
            for feat_fn in feat_fns:
                num_frames = len(np.load(feat_fn))
                total += num_frames

            return total

        def numframes_to_minutes(num_frames):
            # TODO Assumes 10ms strides for the frames. This should generalize to
            # different frame stride widths, as should feature preparation.
            minutes = ((num_frames*10)/1000)/60
            return minutes

        total_frames = 0

        num_train_frames = get_number_of_frames(self.get_train_fns()[0])
        total_frames += num_train_frames
        num_valid_frames = get_number_of_frames(self.get_valid_fns()[0])
        total_frames += num_valid_frames
        num_test_frames = get_number_of_frames(self.get_test_fns()[0])
        total_frames += num_test_frames

        print("Train duration: %0.3f" % numframes_to_minutes(num_train_frames))
        print("Validation duration: %0.3f" % numframes_to_minutes(num_valid_frames))
        print("Test duration: %0.3f" % numframes_to_minutes(num_test_frames))
        print("Total duration: %0.3f" % numframes_to_minutes(total_frames))

    def prepare_feats(self, org_dir):
        """ Prepares input features"""

        if not os.path.isdir(self.FEAT_DIR):
            os.makedirs(self.FEAT_DIR)

        for fn in os.listdir(org_dir):
            path = join(org_dir, fn)
            if path.endswith(".wav"):
                prefix = os.path.basename(os.path.splitext(path)[0])
                mono16k_wav_path = join(self.FEAT_DIR, prefix+".wav")
                if not os.path.isfile(mono16k_wav_path):
                    feat_extract.convert_wav(path, mono16k_wav_path)

        feat_extract.from_dir(self.FEAT_DIR, self.feat_type)

    def get_prefixes(self):
        fns = [fn for fn in os.listdir(self.LABEL_DIR)
               if fn.endswith(self.label_type)]
        prefixes = [os.path.splitext(fn)[0] for fn in fns]
        return prefixes

    def make_data_splits(self, max_samples=1000, seed=0):
        """ Splits the utterances into training, validation and test sets."""

        train_prefix_fn = join(self.TGT_DIR, "train_prefixes.txt")
        valid_prefix_fn = join(self.TGT_DIR, "valid_prefixes.txt")
        test_prefix_fn = join(self.TGT_DIR, "test_prefixes.txt")

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

            return train_prefixes, valid_prefixes, test_prefixes

        prefixes = self.get_prefixes()
        # TODO Note that I'm shuffling after sorting; this could be better.
        # TODO Remove explicit reference to "fbank"
        prefixes = utils.filter_by_size(
            self.FEAT_DIR, prefixes, "fbank", max_samples)
        Ratios = namedtuple("Ratios", ["train", "valid", "test"])
        # TODO These ratios can't be hardcoded
        ratios = Ratios(.90, .05, .05)
        train_end = int(ratios.train*len(prefixes))
        valid_end = int(train_end + ratios.valid*len(prefixes))
        random.shuffle(prefixes)
        train_prefixes = prefixes[:train_end]
        valid_prefixes = prefixes[train_end:valid_end]
        test_prefixes = prefixes[valid_end:]

        with open(train_prefix_fn, "w") as train_f:
            for prefix in train_prefixes:
                print(prefix, file=train_f)
        with open(valid_prefix_fn, "w") as dev_f:
            for prefix in valid_prefixes:
                print(prefix, file=dev_f)
        with open(test_prefix_fn, "w") as test_f:
            for prefix in test_prefixes:
                print(prefix, file=test_f)

        return train_prefixes, valid_prefixes, test_prefixes

    def determine_labels(self):
        """ Returns a set of phonemes found in the corpus. """
        phonemes = set()
        for fn in os.listdir(self.LABEL_DIR):
            with open(join(self.LABEL_DIR, fn)) as f:
                line_phonemes = set(f.readline().split())
                phonemes = phonemes.union(line_phonemes)
        return phonemes

class ReadyCorpus(AbstractCorpus):
    """ Interface to a corpus that has WAV files and label files split into
    utterances and segregated in a directory with a "feat" and "label" dir. """

    def __init__(self, tgt_dir, feat_type="fbank", label_type="phonemes"):
        super().__init__(feat_type, label_type)

        self.TGT_DIR = tgt_dir
        self.FEAT_DIR = os.path.join(tgt_dir, "feat")
        self.LABEL_DIR = os.path.join(tgt_dir, "label")

        if not os.path.isdir(self.FEAT_DIR):
            raise Exception("The supplied path requires a 'feat' subdirectory.")
        if not os.path.isdir(self.LABEL_DIR):
            raise Exception("The supplied path requires a 'label' subdirectory.")

        self.prepare_feats(self.FEAT_DIR) # In this case the feat_dir is the same as the org_dir
        self.labels = self.determine_labels()
        train, valid, test = self.make_data_splits()

        self.train_prefixes = train
        self.valid_prefixes = valid
        self.test_prefixes = test

        # TODO just testing model.transcribe() by using the test set here
        self.untranscribed_prefixes = self.get_untranscribed_prefixes()

        # Sort the training prefixes by size for more efficient training
        self.train_prefixes = utils.sort_by_size(
            self.FEAT_DIR, self.train_prefixes, feat_type)

        # TODO Should be in the abstract corpus. It's common to all corpora but
        # it needs to be set after self.labels. Perhaps I should use a label
        # setter which creates this, then indices_to_phonemes/indices_to_labels
        # will automatically call it.
        self.LABEL_TO_INDEX = {label: index for index, label in enumerate(
                                 ["pad"] + sorted(list(self.labels)))}
        self.INDEX_TO_LABEL = {index: phn for index, phn in enumerate(
                                 ["pad"] + sorted(list(self.labels)))}
        self.vocab_size = len(self.labels)
