""" Provides a Corpus class that can read ELAN .eaf XML files. """

import logging
from pathlib import Path
from typing import List, Union, Dict, Tuple, Callable, Set

import pympi.Elan

from .. import corpus
from .. import utterance
from ..utterance import Utterance
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

def utterances_from_dir(eaf_dir: Path,
                        tier_prefixes: List[str]) -> List[Utterance]:
    """ Returns the utterances found in ELAN files in a directory.

    Recursively explores the directory, gathering ELAN files and extracting
    utterances from them for tiers that start with the specified prefixes.

    Args:
        eaf_dir: A path to the directory to be searched
        tier_prefixes: Stings matching the start of ELAN tier names that are to
            be extracted. For example, if you want to extract from tiers "xv-Jane"
            and "xv-Mark", then tier_prefixes = ["xv"] would do the job.

    Returns:
        A list of Utterance objects.

    """

    logging.info(
        "EAF from directory: {}, searching with tier_prefixes {}".format(
            eaf_dir, tier_prefixes))

    utterances = []
    for eaf_path in eaf_dir.glob("**/*.eaf"):
        eaf_utterances = utterances_from_eaf(eaf_path, tier_prefixes)
        utterances.extend(eaf_utterances)
    return utterances
