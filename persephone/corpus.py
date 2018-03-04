"""
Describes the abstract Corpus class that all interfaces to corpora should
subclass.
"""

from collections import namedtuple
import logging.config
import os
from pathlib import Path
import pickle
from os.path import join
import random
import subprocess
from typing import List, Callable, Type, TypeVar

import numpy as np

from . import config
from .preprocess import feat_extract
from . import utils
from .exceptions import PersephoneException
from .preprocess import elan, wav
from . import utterance
from .utterance import Utterance
from .preprocess.labels import LabelSegmenter

logging.config.fileConfig(config.LOGGING_INI_PATH)

CorpusT = TypeVar("CorpusT", bound="Corpus")

class Corpus:
    """ Represents a preprocessed corpus that is ready to be used in model
    training.

    Construction of a Corpus object involves preprocessing data if the data has
    not previously already been preprocessed. Once a Corpus object is created
    it should be considered immutable. At this point feature extraction from
    WAVs will have been performed, with feature files in tgt_dir/feat/.
    Transcriptions will have been segmented into appropriate tokens (labels)
    and will be stored in tgt_dir/label/.
    """

    def __init__(self, feat_type, label_type, tgt_dir, labels,
                 max_samples=1000, speakers=None):
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

        #: A string representing the type of speech feature (eg. "fbank"
        #: for log filterbank energies).
        self.feat_type = feat_type

        #: An arbitrary string representing the transcription tokenization
        #: used (eg. "phonemes", "tones", "joint", or "characters").
        self.label_type = label_type

        # Setting up directories
        self.set_and_check_directories(tgt_dir)

        # Label-related stuff
        self.initialize_labels(labels)
        logging.info("Corpus label set: \n\t{}".format(labels))

        # This is a lazy function that assumes wavs are already in the WAV dir
        # but only creates features if necessary
        self.prepare_feats()
        self._num_feats = None

        # This is also lazy if the {train,valid,test}_prefixes.txt files exist.
        self.make_data_splits(max_samples=max_samples)

        # Sort the training prefixes by size for more efficient training
        self.train_prefixes = utils.sort_by_size(
            self.feat_dir, self.train_prefixes, feat_type)

        # Ensure no overlap between training and test sets
        self.ensure_no_set_overlap()

        self.untranscribed_prefixes = self.get_untranscribed_prefixes()

        # TODO Need to contemplate whether Corpus objects have Utterance
        # objects or # not. Some of the TestBKW tests currently rely on this
        # for testing.
        self.utterances = None # type: List[Utterance]

        self.pickle()

    @classmethod
    def from_elan(cls: Type[CorpusT], org_dir: Path, tgt_dir: Path,
                 feat_type: str = "fbank", label_type: str = "phonemes",
                 utterance_filter: Callable[[Utterance], bool] = None,
                 label_segmenter: LabelSegmenter = None,
                 speakers: List[str] = None, lazy: bool = True,
                 tier_prefixes: List[str] = ["xv", "rf"]) -> CorpusT:

        # Read utterances from org_dir.
        utterances = elan.utterances_from_dir(org_dir,
                                              tier_prefixes=tier_prefixes)

        # Filter utterances based on some criteria (such as codeswitching).
        utterances = [utter for utter in utterances if utterance_filter(utter)]
        utterances = utterance.remove_duplicates(utterances)

        # Segment the labels in the utterances appropriately
        utterances = [label_segmenter.segment_labels(utter) for utter in utterances]

        # Remove utterances without transcriptions.
        utterances = utterance.remove_empty_text(utterances)

        # Remove utterances with exceptionally short wav_files that are too
        # short for CTC to work.
        utterances = utterance.remove_too_short(utterances)

        tgt_dir.mkdir(parents=True, exist_ok=True)

        # TODO A lot of these methods aren't ELAN-specific. preprocess.elan was
        # only used to get the utterances. There could be another Corpus
        # factory method that takes Utterance objects. the fromElan and
        # fromPangloss constructors could call this.

        # Writes the transcriptions to the tgt_dir/label/ dir
        utterance.write_transcriptions(utterances, (tgt_dir / "label"),
                               label_type, lazy=lazy)
        # Extracts utterance level WAV information from the input file.
        wav.extract_wavs(utterances, (tgt_dir / "wav"), lazy=lazy)

        corpus = cls(feat_type, label_type, tgt_dir,
                     label_segmenter.labels, speakers=speakers)
        corpus.utterances = utterances
        return corpus

    def get_wav_dir(self) -> Path:
        return self.tgt_dir / "wav"

    def get_feat_dir(self) -> Path:
        return self.tgt_dir / "feat"

    def get_label_dir(self) -> Path:
        return self.tgt_dir / "label"

    @property
    def train_prefix_fn(self) -> Path:
        return self.tgt_dir / "train_prefixes.txt"

    @property
    def valid_prefix_fn(self) -> Path:
        return self.tgt_dir / "valid_prefixes.txt"

    @property
    def test_prefix_fn(self) -> Path:
        return self.tgt_dir / "test_prefixes.txt"

    def set_and_check_directories(self, tgt_dir: Path) -> None:

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
            prefix = os.path.basename(os.path.splitext(str(path))[0])
            mono16k_wav_path = self.feat_dir / "{}.wav".format(prefix)
            feat_path = self.feat_dir / "{}.{}.npy".format(prefix, self.feat_type)
            if not feat_path.is_file():
                # Then we should extract feats
                should_extract_feats = True
                if not mono16k_wav_path.is_file():
                    feat_extract.convert_wav(path, mono16k_wav_path)

        # TODO Should be extracting feats on a per-file basis. Right now we
        # check if any feats files don't exist and then do all the feature
        # extraction.
        if should_extract_feats:
            feat_extract.from_dir(self.feat_dir, self.feat_type)

    def make_data_splits(self, max_samples):
        """ Splits the utterances into training, validation and test sets."""

        train_f_exists = self.train_prefix_fn.is_file()
        valid_f_exists = self.valid_prefix_fn.is_file()
        test_f_exists = self.test_prefix_fn.is_file()

        if train_f_exists and valid_f_exists and test_f_exists:
            self.train_prefixes = self.read_prefixes(self.train_prefix_fn)
            self.valid_prefixes = self.read_prefixes(self.valid_prefix_fn)
            self.test_prefixes = self.read_prefixes(self.test_prefix_fn)
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
            self.write_prefixes(train_prefixes, self.train_prefix_fn)
            self.write_prefixes(valid_prefixes, self.valid_prefix_fn)
            self.write_prefixes(test_prefixes, self.test_prefix_fn)
        elif not train_f_exists and valid_f_exists and test_f_exists:
            # Then we just make all other prefixes training prefixes.
            self.valid_prefixes = self.read_prefixes(self.valid_prefix_fn)
            self.test_prefixes = self.read_prefixes(self.test_prefix_fn)
            train_prefixes = list(
                set(prefixes) - set(self.valid_prefixes))
            self.train_prefixes = list(
                set(train_prefixes) - set(self.test_prefixes))
            self.write_prefixes(self.train_prefixes, self.train_prefix_fn)
        else:
            raise NotImplementedError(
                "The following case has not been implemented:" + 
                "{} exists - {}\n".format(self.train_prefix_fn, train_f_exists) +
                "{} exists - {}\n".format(self.valid_prefix_fn, valid_f_exists) +
                "{} exists - {}\n".format(self.test_prefix_fn, test_f_exists))

    @staticmethod
    def read_prefixes(prefix_fn: Path) -> List[str]:
        assert prefix_fn.is_file()
        with prefix_fn.open() as prefix_f:
            prefixes = [line.strip() for line in prefix_f]
        if prefixes == []:
            raise PersephoneException(
                "Empty prefix file {}. Either delete it\
                or put something in it".format(prefix_fn))
        return prefixes

    @staticmethod
    def write_prefixes(prefixes: List[str], prefix_fn: Path) -> None:
        if prefixes == []:
            raise PersephoneException(
                "No prefixes. Will not write {}".format(prefix_fn))
        with prefix_fn.open("w") as prefix_f:
            for prefix in prefixes:
                print(prefix, file=prefix_f)

    @staticmethod
    def divide_prefixes(prefixes, seed=0):
        Ratios = namedtuple("Ratios", ["train", "valid", "test"])
        ratios=Ratios(.90, .05, .05)
        train_end = int(ratios.train*len(prefixes))
        valid_end = int(train_end + ratios.valid*len(prefixes))
        random.seed(seed)
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
        """ Fetches the training set of the corpus.

        Outputs a Tuple of size 2, where the first element is a list of paths
        to input features files, one per utterance. The second element is a list
        of paths to the transcriptions.
        """
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

    def determine_prefixes(self) -> List[str]:
        label_prefixes = [str(path.relative_to(self.label_dir).with_suffix(""))
                          for path in 
                          self.label_dir.glob("**/*.{}".format(self.label_type))]
        wav_prefixes = [str(path.relative_to(self.wav_dir).with_suffix(""))
                          for path in 
                          self.wav_dir.glob("**/*.{}".format("wav"))]

        # Take the intersection; sort for determinism.
        prefixes = sorted(list(set(label_prefixes) & set(wav_prefixes)))

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

    def ensure_no_set_overlap(self):
        """ Ensures no test set data has creeped into the training set."""

        train = set(self.get_train_fns()[0])
        valid = set(self.get_valid_fns()[0])
        test = set(self.get_test_fns()[0])
        assert train - valid == train
        assert train - test == train
        assert valid - train == valid
        assert valid - test == valid
        assert test - train == test
        assert test - valid == test

    def pickle(self):
        """ Pickles the Corpus object in a file in tgt_dir. """

        pickle_path = self.tgt_dir / "corpus.p"
        with pickle_path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls: Type[CorpusT], tgt_dir: Path) -> CorpusT:
        pickle_path = tgt_dir / "corpus.p"
        with pickle_path.open("rb") as f:
            return pickle.load(f)


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
