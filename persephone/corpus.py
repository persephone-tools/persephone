"""
Describes the abstract Corpus class that all interfaces to corpora should
subclass.
"""

from collections import namedtuple
import logging.config
import os
from pathlib import Path
import pickle
import random
import subprocess
from typing import Any, List, Callable, Optional, Set, Sequence, Tuple, Type, TypeVar

import numpy as np

from .preprocess import feat_extract
from . import utils
from .exceptions import PersephoneException
from .preprocess import elan, wav
from . import utterance
from .utterance import Utterance
from .preprocess.labels import LabelSegmenter

logger = logging.getLogger(__name__) # type: ignore

CorpusT = TypeVar("CorpusT", bound="Corpus")

def ensure_no_set_overlap(train: Sequence[str], valid: Sequence[str], test: Sequence[str]) -> None:
    """ Ensures no test set data has creeped into the training set."""

    logger.debug("Ensuring that the training, validation and test data sets have no overlap")
    train_s = set(train)
    valid_s = set(valid)
    test_s = set(test)

    if train_s & valid_s:
        logger.warning("train and valid have overlapping items: {}".format(train_s & valid_s))
        raise PersephoneException("train and valid have overlapping items: {}".format(train_s & valid_s))
    if train_s & test_s:
        logger.warning("train and test have overlapping items: {}".format(train_s & test_s))
        raise PersephoneException("train and test have overlapping items: {}".format(train_s & test_s))
    if valid_s & test_s:
        logger.warning("valid and test have overlapping items: {}".format(valid_s & test_s))
        raise PersephoneException("valid and test have overlapping items: {}".format(valid_s & test_s))

def find_untranscribed_wavs(wav_path: Path, transcription_path: Path, label_type: str) -> List[str]:
    """Find the prefixes for all the wav files that do not have an associated transcription
    Args:
        wav_path: Path to search for wav files in
        transcription_path: Path to search for transcriptions in
        label_type: The type of labels for transcriptions. Eg "phonemes" "phonemes_and_tones"
    Returns:
        A list of all untranscribed prefixes
    """
    audio_files = wav_path.glob("**/*.wav")
    transcription_files = transcription_path.glob("**/*.{}".format(label_type))
    
    transcription_file_prefixes = [t_file.stem for t_file in transcription_files]

    untranscribed_prefixes = [] # type: List[str]
    for a_file in audio_files:
        if a_file.stem not in transcription_file_prefixes:
            untranscribed_prefixes.append(a_file.stem)
    return untranscribed_prefixes

def get_untranscribed_prefixes_from_file(target_directory: Path) -> List[str]:
    """
    The file "untranscribed_prefixes.txt" will specify prefixes which
    do not have an associated transcription file if placed in the target directory.

    This will fetch those prefixes from that file and will return an empty
    list if that file does not exist.

    See find_untranscribed_wavs function for finding untranscribed prefixes in an
    experiment directory.

    Returns:
        A list of all untranscribed prefixes as specified in the file
    """

    untranscribed_prefix_fn = target_directory / "untranscribed_prefixes.txt"
    if untranscribed_prefix_fn.exists():
        with untranscribed_prefix_fn.open() as f:
            prefixes = f.readlines()

        return [prefix.strip() for prefix in prefixes]
    else:
        #logger.warning("Attempting to get untranscribed prefixes but the file ({})"
        #                " that should specify these does not exist".format(untranscribed_prefix_fn))
        pass
    return []

class Corpus:
    """ Represents a preprocessed corpus that is ready to be used in model
    training.

    Construction of a `Corpus` instance involves preprocessing data if the data has
    not previously already been preprocessed. The extent of the preprocessing
    depends on which constructor is used. If the default constructor,
    `__init__()` is used, transcriptions are assumed to already be preprocessed
    and only speech feature extraction from WAV files is performed. In other
    constructors such as `from_elan()`, preprocessing of the transcriptions is
    performed. See the documentation of the relevant constructors for more
    information.

    Once a Corpus object is created it should be considered immutable. At this
    point feature extraction from WAVs will have been performed, with feature
    files in `tgt_dir/feat/`.  Transcriptions will have been segmented into
    appropriate tokens (labels) and will be stored in `tgt_dir/label/`.

    """

    def __init__(self, feat_type: str, label_type: str, tgt_dir: Path, labels: Optional[Any] = None,
                 max_samples:int=1000, speakers: Optional[Sequence[str]] = None) -> None:
        """ Construct a `Corpus` instance from preprocessed data.

        Assumes that the corpus data has been preprocessed and is
        structured as follows: (1) WAVs for each utterance are found in
        `<tgt_dir>/wav/` with the filename `<prefix>.wav`, where `prefix` is
        some string uniquely identifying the utterance; (2) For each WAV file,
        there is a corresponding transcription found in `<tgt_dir>/label/` with
        the filename `<prefix>.<label_type>`, where `label_type` is some string
        describing the type of label used (for example, "phonemes" or "tones").

        If the data is found in the format, WAV normalization and speech
        feature extraction will be performed during `Corpus` construction, and
        the utterances will be randomly divided into training, validation and
        test_sets. If you would like to define these datasets yourself, include
        files named `train_prefixes.txt`, `valid_prefixes.txt` and
        `test_prefixes.txt` in `<tgt_dir>`. Each file should be a list of
        prefixes (utterance IDs), one per line. If these are found during
        `Corpus` construction, those sets will be used instead.

        Args:
            feat_type: A string describing the input speech features. For
                       example, "fbank" for log Mel filterbank features.
            label_type: A string describing the transcription labels. For example,
                         "phonemes" or "tones".
            labels: A set of strings representing labels (tokens) used in
                    transcription. For example: {"a", "o", "th", ...}.
                    If this parameter is not provided the experiment directory is
                    scanned for labels present in the transcription files.
            max_samples: The maximum number of samples an utterance in the
                corpus may have. If an utterance is longer than this, it is not
                included in the corpus.

        """

        if speakers:
            raise NotImplementedError("Speakers not implemented")

        logger.debug("Creating a new Corpus object with feature type %s, label type %s,"
                     "target directory %s, label set %s, ms, max_samples, speakers")

        # In case path is supplied as a string, make it a Path
        self.tgt_dir = Path(tgt_dir)

        #: A string representing the type of speech feature (eg. "fbank"
        #: for log filterbank energies).
        self.feat_type = feat_type

        #: An arbitrary string representing the transcription tokenization
        #: used (eg. "phonemes", "tones", "joint", or "characters").
        self.label_type = label_type

        # Setting up directories
        # Set the directory names
        self.feat_dir = self.get_feat_dir()
        self.wav_dir = self.get_wav_dir()
        self.label_dir = self.get_label_dir()

        logger.debug("Setting up directories for this Corpus object at %s", self.tgt_dir)
        self.set_and_check_directories(self.tgt_dir)

        # Label-related stuff
        if labels is not None:
            self.labels = labels
        else:
            self.labels = determine_labels(self.tgt_dir, label_type)
        self.vocab_size = len(self.labels)
        self.LABEL_TO_INDEX, self.INDEX_TO_LABEL = self.initialize_labels(self.labels)
        logger.info("Corpus label set: \n\t{}".format(self.labels))

        # This is a lazy function that assumes wavs are already in the WAV dir
        # but only creates features if necessary
        logger.debug("Preparing features")
        self.prepare_feats()
        self._num_feats = None

        self.train_prefixes = [] # type: List[str]
        self.valid_prefixes = [] # type: List[str]
        self.test_prefixes = [] # type: List[str]
        # This is also lazy if the {train,valid,test}_prefixes.txt files exist.
        self.make_data_splits(max_samples=max_samples)

        # Sort the training prefixes by size for more efficient training
        logger.debug("Training prefixes")
        self.train_prefixes = utils.sort_by_size(
            self.feat_dir, self.train_prefixes, feat_type)

        # Ensure no overlap between training and test sets
        try:
            ensure_no_set_overlap(
                self.get_train_fns()[0],
                self.get_valid_fns()[0],
                self.get_test_fns()[0]
            )
        except PersephoneException:
            logger.error("Got overlap between train valid and test data sets")
            raise

        untranscribed_from_file = self.get_untranscribed_prefixes()
        untranscribed_found = find_untranscribed_wavs(self.get_wav_dir(), self.get_label_dir(), self.label_type)

        self.untranscribed_prefixes = list(set(untranscribed_from_file) & set(untranscribed_found))

        # TODO Need to contemplate whether Corpus objects have Utterance
        # objects or # not. Some of the TestBKW tests currently rely on this
        # for testing.
        self.utterances = []  # type: List[Utterance]

        self.pickle()

    @classmethod
    def from_elan(cls: Type[CorpusT], org_dir: Path, tgt_dir: Path,
                  feat_type: str = "fbank", label_type: str = "phonemes",
                  utterance_filter: Callable[[Utterance], bool] = None,
                  label_segmenter: LabelSegmenter = None,
                  speakers: List[str] = None, lazy: bool = True,
                  tier_prefixes: Tuple[str, ...] = ("xv", "rf")) -> CorpusT:
        """ Construct a `Corpus` from ELAN files.

        Args:
            org_dir: A path to the directory containing the unpreprocessed
                data.
            tgt_dir: A path to the directory where the preprocessed data will
                be stored.
            feat_type: A string describing the input speech features. For
                       example, "fbank" for log Mel filterbank features.
            label_type: A string describing the transcription labels. For example,
                         "phonemes" or "tones".
            utterance_filter: A function that returns False if an utterance
                should not be included in the corpus and True otherwise. This
                can be used to remove undesirable utterances for training, such as
                codeswitched utterances.
            label_segmenter: An object that has an attribute `segment_labels`,
                which is creates new `Utterance` instances from old ones,
                by segmenting the tokens in their `text` attribute. Note,
                `LabelSegmenter` might be better as a function, the only issue
                is it needs to carry with it a list of labels. This could
                potentially be a function attribute.
            speakers: A list of speakers to filter for. If `None`, utterances
                from all speakers are included.
            tier_prefixes: A collection of strings that prefix ELAN tiers to
                filter for. For example, if this is `("xv", "rf")`, then tiers
                named "xv", "xv@Mark", "rf@Rose" would be extracted if they
                existed.

        """
        # This currently bails out if label_segmenter is not provided
        if not label_segmenter:
            raise ValueError("A label segmenter must be provided via label_segmenter")

        # In case path is supplied as a string, make it a Path
        self.tgt_dir = Path(tgt_dir)

        # Read utterances from org_dir.
        utterances = elan.utterances_from_dir(org_dir,
                                              tier_prefixes=tier_prefixes)

        # Filter utterances based on some criteria (such as codeswitching).
        if utterance_filter:
            utterances = [utter for utter in utterances if utterance_filter(utter)]
        utterances = utterance.remove_duplicates(utterances)

        # Segment the labels in the utterances appropriately
        if label_segmenter:
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
        """
        Make sure that the required directories exist in the target directory.
        set variables accordingly.
        """
        logger.info("Setting up directories for corpus in %s", tgt_dir)

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

    def initialize_labels(self, labels: Sequence[str]) -> Tuple[dict, dict]:
        """Create mappings from label to index and index to label"""
        logger.debug("Creating mappings for labels")

        label_to_index = {label: index for index, label in enumerate(
                                 ["pad"] + sorted(list(labels)))}
        index_to_label = {index: phn for index, phn in enumerate(
                                 ["pad"] + sorted(list(labels)))}

        return label_to_index, index_to_label

    def prepare_feats(self) -> None:
        """ Prepares input features"""

        logger.debug("Preparing input features")
        self.feat_dir.mkdir(parents=True, exist_ok=True)

        should_extract_feats = False
        for path in self.wav_dir.iterdir():
            if not path.suffix == ".wav":
                logger.info("Non wav file found in wav directory: %s", path)
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

    def make_data_splits(self, max_samples: int) -> None:
        """ Splits the utterances into training, validation and test sets."""

        train_f_exists = self.train_prefix_fn.is_file()
        valid_f_exists = self.valid_prefix_fn.is_file()
        test_f_exists = self.test_prefix_fn.is_file()

        if train_f_exists and valid_f_exists and test_f_exists:
            logger.debug("Split for training, validation and tests specified by files")
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
            logger.debug("No files supplied to define the split for training, validation"
                         " and tests. Using default.")
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
        if not prefix_fn.is_file():
            logger.critical("Expected a prefix file at path {}, but this path is"
                           " not a file".format(prefix_fn))
        assert prefix_fn.is_file(), "Path {} was not a file".format(prefix_fn)
        with prefix_fn.open() as prefix_f:
            prefixes = [line.strip() for line in prefix_f]
        if not prefixes:
            raise PersephoneException(
                "Empty prefix file {}. Either delete it\
                or put something in it".format(prefix_fn))
        return prefixes

    @staticmethod
    def write_prefixes(prefixes: List[str], prefix_fn: Path) -> None:
        if not prefixes:
            raise PersephoneException(
                "No prefixes. Will not write {}".format(prefix_fn))
        with prefix_fn.open("w") as prefix_f:
            for prefix in prefixes:
                print(prefix, file=prefix_f)

    @staticmethod
    def divide_prefixes(prefixes: List[str], seed:int=0) -> Tuple[List[str], List[str], List[str]]:
        """Divide data into training, validation and test subsets"""
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
        assert train_prefixes, "Got empty set for training data"
        assert valid_prefixes, "Got empty set for validation data"
        assert test_prefixes, "Got empty set for testing data"

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

    def prefixes_to_fns(self, prefixes: List[str]) -> Tuple[List[str], List[str]]:
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
        """ Fetches the validation set of the corpus."""
        return self.prefixes_to_fns(self.valid_prefixes)

    def get_test_fns(self):
        """ Fetches the test set of the corpus."""
        return self.prefixes_to_fns(self.test_prefixes)

    def get_untranscribed_prefixes(self) -> List[str]:
        """
        The file "untranscribed_prefixes.txt" will specify prefixes which
        do not have an associated transcription file if placed in the target directory.

        This will fetch those prefixes from that file and will return an empty
        list if that file does not exist.

        See find_untranscribed_wavs function for finding untranscribed prefixes in an
        experiment directory.
        """
        return get_untranscribed_prefixes_from_file(self.tgt_dir)

    def get_untranscribed_fns(self) -> List[str]:
        feat_fns = [os.path.join(str(self.feat_dir), "untranscribed", "%s.%s.npy" % (prefix, self.feat_type))
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

        if not prefixes:
            raise PersephoneException("""WARNING: Corpus object has no data. Are you sure
            it's in the correct directories? WAVs should be in {} and
            transcriptions in {} with the extension .{}""".format(
                self.wav_dir, self.label_dir, self.label_type))

        return prefixes

    def review(self) -> None:
        """ Used to play the WAV files and compare with the transcription. """

        for prefix in self.determine_prefixes():
            print("Utterance: {}".format(prefix))
            wav_fn = self.feat_dir / "{}.wav".format(prefix)
            label_fn = self.label_dir / "{}.{}".format(prefix,self.label_type)
            with label_fn.open() as f:
                transcript = f.read().strip()
            print("Transcription: {}".format(transcript))
            subprocess.run(["play", str(wav_fn)])

    def pickle(self) -> None:
        """ Pickles the Corpus object in a file in tgt_dir. """

        pickle_path = self.tgt_dir / "corpus.p"
        logger.debug("pickling %r object and saving it to path %s", self, pickle_path)
        with pickle_path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls: Type[CorpusT], tgt_dir: Path) -> CorpusT:
        pickle_path = tgt_dir / "corpus.p"
        logger.debug("Creating Corpus object from pickle file path %s", pickle_path)
        with pickle_path.open("rb") as f:
            return pickle.load(f)


def determine_labels(target_dir: Path, label_type: str) -> set:
    """ Returns a set of all phonemes found in the corpus. Assumes that WAV files and
    label files are split into utterances and segregated in a directory which contains a
    "wav" subdirectory and "label" subdirectory.

    Arguments:
        target_dir: A `Path` to the directory where the corpus data is found
        label_type: The type of label we are creating the label set from. For example
                    "phonemes" would only search for labels for that type.
    """
    logger.info("Finding phonemes of type %s in directory %s", label_type, target_dir)

    label_dir = target_dir / "label/"
    if not label_dir.is_dir():
        raise FileNotFoundError(
            "The directory {} does not exist.".format(target_dir))

    phonemes = set() # type: set
    for fn in os.listdir(str(label_dir)):
        if fn.endswith(str(label_type)):
            with (label_dir / fn).open("r") as f:
                try:
                    line_phonemes = set(f.readline().split())
                except UnicodeDecodeError:
                    logger.error("Unicode decode error on file %s", fn)
                    print("Unicode decode error on file {}".format(fn))
                    raise
                phonemes = phonemes.union(line_phonemes)
    return phonemes
