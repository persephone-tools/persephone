""" Provides a Corpus class that can read ELAN .eaf XML files. """

import logging
from pathlib import Path
from typing import List, Union, Dict, Tuple, Callable, Set

import pympi.Elan

from .. import corpus
from .. import utterance
from ..utterance import Utterance
from ..utterance import write_utters
from ..preprocess.wav import extract_wavs
from ..preprocess.labels import LabelSegmenter

class Eaf(pympi.Elan.Eaf):
    """ This subclass exists because eaf MEDIA_DESCRIPTOR elements typically
    have RELATIVE_MEDIA_URL that contains the path of the media file relative
    to the eaf file. However, pympi.Elan.Eaf doesn't have a path attribute
    pointing to the eaf file itself. This class here adds this attribute so
    that the path to the media file can be reconstructed."""

    def __init__(self, eaf_path: Path) -> None:
        super().__init__(str(eaf_path))
        self.eaf_path = eaf_path
        self.initialize_media_descriptor()

    @property
    def media_path(self) -> Path:
        """ The URL of the associated media file."""
        return self.get_media_path(self.media_descriptor)

    @property
    def time_origin(self) -> int:
        """ The time offset of the annotations."""
        try:
            return int(self.media_descriptor["TIME_ORIGIN"])
        except KeyError:
            # Since it wasn't specified, we don't need to adjust 
            # utterance offsets.
            return 0

    def get_media_path(self, media_descriptor: Dict) -> Path:
        return self.eaf_path.parent / media_descriptor["RELATIVE_MEDIA_URL"]

    def initialize_media_descriptor(self) -> None:
        """
        Returns the media descriptor for the first media descriptor where
        the file can be found.
        """

        for md in self.media_descriptors:
            media_path = self.get_media_path(md)
            if media_path.is_file():
                self.media_descriptor = md
                return

        raise FileNotFoundError(
            """Cannot find media file corresponding to {}.
            Tried looking for the following files: {}.
            """.format(self.eaf_path, [self.get_media_path(md)
                                       for md in self.media_descriptors]))

def sort_annotations(annotations: List[Tuple[int, int, str]]
                     ) -> List[Tuple[int, int, str]]:
    """ Sorts the annotations by their start_time. """
    return sorted(annotations, key=lambda x: x[0])

def utterances_from_tier(eafob: Eaf, tier_name: str) -> List[Utterance]:
    """ Returns utterances found in the given Eaf object in the given tier."""

    try:
        speaker = eafob.tiers[tier_name][2]["PARTICIPANT"]
    except KeyError:
        speaker = None # We don't know the name of the speaker.

    tier_utterances = []

    annotations = sort_annotations(
        list(eafob.get_annotation_data_for_tier(tier_name)))

    for i, annotation in enumerate(annotations):
        eaf_stem = eafob.eaf_path.stem
        utter_id = "{}.{}.{}".format(eaf_stem, tier_name, i)
        start_time = eafob.time_origin + annotation[0]
        end_time = eafob.time_origin + annotation[1]
        text = annotation[2]
        utterance = Utterance(eafob.media_path, eafob.eaf_path, utter_id,
                              start_time, end_time, text, speaker)
        tier_utterances.append(utterance)

    return tier_utterances

def utterances_from_eaf(eaf_path: Path, tier_prefixes: List[str]) -> List[Utterance]:
    """
    Extracts utterances in tiers that start with tier_prefixes found in the ELAN .eaf XML file
    at eaf_path.

    For example, if xv@Mark is a tier in the eaf file, and
    tier_prefixes = ["xv"], then utterances from that tier will be gathered.
    """

    if not eaf_path.is_file():
        raise FileNotFoundError("Cannot find {}".format(eaf_path))

    eaf = Eaf(eaf_path)
    utterances = []
    for tier_name in sorted(list(eaf.tiers)): # Sorting for determinism
        for tier_prefix in tier_prefixes:
            if tier_name.startswith(tier_prefix):
                utterances.extend(utterances_from_tier(eaf, tier_name))
                break
    return utterances

def utterances_from_dir(eaf_dir: Path, tier_prefixes: List[str]) -> List[Utterance]:
    """ Returns the utterances found in a directory. """

    logging.info(
        "EAF from directory: {}, searching with tier_prefixes {}".format(
            eaf_dir, tier_prefixes))

    utterances = []
    for eaf_path in eaf_dir.glob("**/*.eaf"):
        eaf_utterances = utterances_from_eaf(eaf_path, tier_prefixes)
        utterances.extend(eaf_utterances)
    return utterances

class Corpus(corpus.Corpus):
    def __init__(self, org_dir: Path, tgt_dir: Path,
                 feat_type: str = "fbank", label_type: str = "phonemes",
                 utterance_filter: Callable[[Utterance], bool] = None,
                 label_segmenter: LabelSegmenter = None,
                 speakers: List[str] = None, lazy: bool = True) -> None:
        """
        Need to think about this constructor. Ideally it should take a
        function:

            label_preprocess(utter: String) -> String

        which takes a unpreprocessed utterance (perhaps with spaces delimiting
        words in some orthography), and outputs a string where spaces delimit
        things like phonemes and tones.

        corpus.Corpus and corpus.ReadyCorpus could also take such an argument.

        There's probably also going to need to be a way for such a function to
        mark utterances so that they aren't included in the corpus. For
        example, for filtering out code-switched sentences.

        Currently the corpus.Corpus superclass takes a labels argument which is
        just a collection of phonemes etc which the corpus assumes have already
        been segmented correctly, so that it doesn't have to read the whole
        corpus to figure them out. corpus.Corpus should also have the option of
        taking the label_preprocess argument, in which case it would not take
        labels.

        This constructor, elan.Corpus, shouldn't be taking a labels
        argument, since ELAN files are unlikely ever going to
        phoneme-segmented and we don't really want to encourage linguists to do that
        and add tiers or anything. There could be a function that takes labels
        and produces a label_preprocess function that is based on the greedy
        left-to-right phoneme segmentation. So one would do:

            > labels = {"a", "x", "b", ..., etc}
            > greedy_segmenter = create_greedy_segmenter(labels)
            > corp = elan.Corpus(tgt_dir, label_preprocess=greedy_segmenter,
                                 feat_type="fbank", label_type="phonemes")

        I notice now that this means that label_type and label_preprocess needs
        to be manually coordinated by the creater of the elan.Corpus. Ideally 
        the label_preprocess function would dictate the label_type somehow.

        From the end user's perspective, what happens if the orthography cannot
        be automatically segmented with the greedy algorithm AND the user
        doesn't want to do character-level prediction AND they can't write
        their own segmentation algorithm because of lack of technical
        expertise. Then I guess that is a situation where they might want to
        do manual segmentation as it's the only option. In such cases, they
        should be able to create another ELAN tier as is and there can be a
        label_segmenter that is just the identity function.
        """

        self.tgt_dir = tgt_dir

        # Read utterances from org_dir.
        utterances = utterances_from_dir(org_dir,
                                         tier_prefixes=["xv", "rf"])

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

        self.utterances = utterances

        tgt_dir.mkdir(parents=True, exist_ok=True)
        utterance.write_utt2spk(self.utterances, self.tgt_dir)

        # Writes the utterances to the tgt_dir/label/ dir
        write_utters(self.utterances, self.get_label_dir(), label_type, lazy=lazy)
        # Extracts utterance level WAV information from the input file.
        extract_wavs(self.utterances, self.get_wav_dir(), lazy=lazy)

        super().__init__(feat_type, label_type, tgt_dir,
                         label_segmenter.labels, speakers=speakers)
