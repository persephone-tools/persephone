"""
Describes the abstract Corpus class that all interfaces to corpora should
subclass.
"""

from collections import namedtuple
import os
from pathlib import Path
from os.path import join
import random
import subprocess

import numpy as np

from .preprocess import feat_extract
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

    def get_wav_dir(self):
        return self.tgt_dir / "wav"

    def get_feat_dir(self):
        return self.tgt_dir / "feat"

    def get_label_dir(self):
        return self.tgt_dir / "label"

    def set_and_check_directories(self, tgt_dir):

        # Set the directory names
        self.tgt_dir = tgt_dir
        self.feat_dir = self.get_feat_dir()
        self.wav_dir = self.get_wav_dir()
        self.label_dir = self.get_label_dir()

        # Check directories exist.
        if not tgt_dir.is_dir():
            raise FileNotFoundError(
                "The directory {} does not exist.".format(tgt_dir))
        if not self.wav_dir.is_dir():
            raise PersephoneException(
                "The supplied path requires a 'wav' subdirectory.")
        self.feat_dir.mkdir(parents=True, exist_ok=True)
        if not self.label_dir.is_dir():
            raise PersephoneException(
                "The supplied path requires a 'label' subdirectory.")

    def initialize_labels(self, labels):
        self.labels = labels
        self.vocab_size = len(self.labels)
        self.LABEL_TO_INDEX = {label: index for index, label in enumerate(
                                 ["pad"] + sorted(list(self.labels)))}
        self.INDEX_TO_LABEL = {index: phn for index, phn in enumerate(
                                 ["pad"] + sorted(list(self.labels)))}

    def prepare_feats(self):
        """ Prepares input features"""

        self.feat_dir.mkdir(parents=True, exist_ok=True)

        should_extract_feats = False
        for path in self.wav_dir.iterdir():
            if not path.suffix == ".wav":
                continue
            # TODO use pathlib.Path
            prefix = os.path.basename(os.path.splitext(str(path))[0])
            mono16k_wav_path = join(str(self.feat_dir), "%s.wav" % prefix)
            feat_path = join(str(self.feat_dir),
                             "%s.%s.npy" % (prefix, self.feat_type))
            if not os.path.isfile(feat_path):
                # Then we should extract feats
                should_extract_feats = True
                if not os.path.isfile(mono16k_wav_path):
                    feat_extract.convert_wav(str(path), mono16k_wav_path)

        # TODO Should be extracting feats on a per-file basis. Right now we
        # check if any feats files don't exist and then do all the feature
        # extraction.
        if should_extract_feats:
            feat_extract.from_dir(str(self.feat_dir), self.feat_type)

    def make_data_splits(self, max_samples):
        """ Splits the utterances into training, validation and test sets."""

        # TODO Change this to being paths everywhere
        train_prefix_fn = join(str(self.tgt_dir), "train_prefixes.txt")
        valid_prefix_fn = join(str(self.tgt_dir), "valid_prefixes.txt")
        test_prefix_fn = join(str(self.tgt_dir), "test_prefixes.txt")

        train_f_exists = os.path.isfile(train_prefix_fn)
        valid_f_exists = os.path.isfile(valid_prefix_fn)
        test_f_exists = os.path.isfile(test_prefix_fn)

        if train_f_exists and valid_f_exists and test_f_exists:
            self.train_prefixes = self.read_prefixes(train_prefix_fn)
            self.valid_prefixes = self.read_prefixes(valid_prefix_fn)
            self.test_prefixes = self.read_prefixes(test_prefix_fn)
            return

        # Otherwise we now need to load prefixes for other cases addressed
        # below
        prefixes = self.determine_prefixes()
        prefixes = utils.filter_by_size(
            self.feat_dir, prefixes, self.feat_type, max_samples)

        if not train_f_exists and not valid_f_exists and not test_f_exists:
            train_prefixes, valid_prefixes, test_prefixes = self.divide_prefixes(prefixes)
            self.train_prefixes = train_prefixes
            self.valid_prefixes = valid_prefixes
            self.test_prefixes = test_prefixes
            self.write_prefixes(train_prefixes, train_prefix_fn)
            self.write_prefixes(valid_prefixes, valid_prefix_fn)
            self.write_prefixes(test_prefixes, test_prefix_fn)
        elif not train_f_exists and valid_f_exists and test_f_exists:
            # Then we just make all other prefixes training prefixes.
            self.valid_prefixes = self.read_prefixes(valid_prefix_fn)
            self.test_prefixes = self.read_prefixes(test_prefix_fn)
            train_prefixes = list(
                set(prefixes) - set(self.valid_prefixes))
            self.train_prefixes = list(
                set(train_prefixes) - set(self.test_prefixes))
            self.write_prefixes(prefixes, train_prefix_fn)
        else:
            raise NotImplementedError(
                "The following case has not been implemented:" + 
                "{} exists - {}\n".format(train_prefix_fn, train_f_exists) +
                "{} exists - {}\n".format(valid_prefix_fn, valid_f_exists) +
                "{} exists - {}\n".format(test_prefix_fn, test_f_exists))

    @staticmethod
    def read_prefixes(prefix_fn):
        with open(prefix_fn) as prefix_f:
            prefixes = [line.strip() for line in prefix_f]
        if prefixes == []:
            raise PersephoneException(
                "Empty prefix file {}. Either delete it\
                or put something in it".format(prefix_fn))
        return prefixes

    @staticmethod
    def write_prefixes(prefixes, prefix_fn):
        if prefixes == []:
            raise PersephoneException(
                "No prefixes. Will not write {}".format(prefix_fn))
        with open(prefix_fn, "w") as prefix_f:
            for prefix in prefixes:
                print(prefix, file=prefix_f)

    @staticmethod
    def divide_prefixes(prefixes, seed=0):
        Ratios = namedtuple("Ratios", ["train", "valid", "test"])
        ratios=Ratios(.90, .05, .05)
        train_end = int(ratios.train*len(prefixes))
        valid_end = int(train_end + ratios.valid*len(prefixes))
        # TODO Add a context manager to govern randomness here.
        random.shuffle(prefixes)

        train_prefixes = prefixes[:train_end]
        valid_prefixes = prefixes[train_end:valid_end]
        test_prefixes = prefixes[valid_end:]

        # TODO Adjust code to cope properly with toy datasets where these
        # subsets might actually be empty.
        assert train_prefixes
        assert valid_prefixes
        assert test_prefixes

        return train_prefixes, valid_prefixes, test_prefixes

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
        # TODO Return pathlib.Paths
        feat_fns = [str(self.feat_dir / ("%s.%s.npy" % (prefix, self.feat_type)))
                    for prefix in prefixes]
        label_fns = [str(self.label_dir / ("%s.%s" % (prefix, self.label_type)))
                      for prefix in prefixes]
        return feat_fns, label_fns

    def get_train_fns(self):
        return self.prefixes_to_fns(self.train_prefixes)

    def get_valid_fns(self):
        return self.prefixes_to_fns(self.valid_prefixes)

    def get_test_fns(self):
        return self.prefixes_to_fns(self.test_prefixes)

    def get_untranscribed_prefixes(self):

        # TODO Change to pathlib.Path
        untranscribed_prefix_fn = join(str(self.tgt_dir), "untranscribed_prefixes.txt")
        if os.path.exists(untranscribed_prefix_fn):
            with open(untranscribed_prefix_fn) as f:
                prefixes = f.readlines()

            return [prefix.strip() for prefix in prefixes]

        return None

    def get_untranscribed_fns(self):
        feat_fns = [os.path.join(self.feat_dir, "%s.%s.npy" % (prefix, self.feat_type))
                    for prefix in self.untranscribed_prefixes]
        return feat_fns

    def determine_prefixes(self):
        label_fns = [fn for fn in os.listdir(str(self.label_dir))
               if fn.endswith(self.label_type)]
        label_prefixes = set([os.path.splitext(fn)[0] for fn in label_fns])
        wav_fns = [fn for fn in os.listdir(str(self.wav_dir))
               if fn.endswith(".wav")]
        wav_prefixes = set([os.path.splitext(fn)[0] for fn in wav_fns])

        # Take the intersection
        prefixes = list(label_prefixes & wav_prefixes)

        if prefixes == []:
            raise PersephoneException("""WARNING: Corpus object has no data. Are you sure
            it's in the correct directories? WAVs should be in {} and
            transcriptions in {} with the extension .{}""".format(
                self.wav_dir, self.label_dir, self.label_type))

        return prefixes

    def review(self):
        """ Used to play the WAV files and compare with the transcription. """

        for prefix in self.determine_prefixes():
            print("Utterance: {}".format(prefix))
            wav_fn = self.feat_dir / "{}.wav".format(prefix)
            label_fn = self.label_dir / "{}.{}".format(prefix,self.label_type)
            with label_fn.open() as f:
                transcript = f.read().strip()
            print("Transcription: {}".format(transcript))
            subprocess.run(["play", str(wav_fn)])

class ReadyCorpus(Corpus):
    """ Interface to a corpus that has WAV files and label files split into
    utterances and segregated in a directory with a "wav" and "label" dir. """

    def __init__(self, tgt_dir, feat_type="fbank", label_type="phonemes"):

        labels = self.determine_labels(tgt_dir, label_type)

        super().__init__(feat_type, label_type, Path(tgt_dir), labels)

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
