""" Provides a Corpus class that can read ELAN .eaf XML files. """

from pathlib import Path
from typing import List, Union, Dict, Tuple, Callable

import pympi.Elan

from .. import corpus
from ..utterance import Utterance

class Eaf(pympi.Elan.Eaf):
    """ This subclass exists because eaf MEDIA_DESCRIPTOR elements typically
    have RELATIVE_MEDIA_URL that contains the path of the media file relative
    to the eaf file. However, pympi.Elan.Eaf doesn't have a path attribute
    pointing to the eaf file itself. This class here adds this attribute so
    that the path to the media file can be reconstructed."""

    def __init__(self, eaf_path: Path, author: str = "persephone") -> None:
        super().__init__(str(eaf_path), author=author)
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

def utterances_from_eaf(eaf_path: Path, tier_prefixes: List[str]) -> None:#List[Utterance]:
    """
    Extracts utterances in tiers that start with tier_prefixes found in the ELAN .eaf XML file
    at eaf_path.

    For example, if xv@Mark is a tier in the eaf file, and
    tier_prefixes = ["xv"], then utterances from that tier will be gathered.
    """

    if not eaf_path.is_file():
        raise FileNotFoundError("Cannot find {}".format(eaf_path))

    # This could actually only take a tier_name argument and then be moved
    # further below to take the eafob and media_path of the enclosing function.
    # Not sure whether to do that (at one extreme), or to take more arguments
    # and move out of the enclosing function.
    def utterances_from_tier(eafob: Eaf, tier_name: str) -> List[str]:
        """ Returns utterances found in the given Eaf object in the given tier."""

        try:
            participant = eafob.tiers[tier_name][2]["PARTICIPANT"]
        except KeyError:
            participant = None # We don't know the name of the speaker.

        tier_utterances = []
        for i, annotation in enumerate(eafob.get_annotation_data_for_tier(tier_name)):
            eaf_stem = eaf_path.stem
            utter_id = "{}.{}.{}".format(eaf_stem, tier_name, i)
            start_time = eafob.time_origin + annotation[0]
            end_time = eafob.time_origin + annotation[1]
            text = annotation[2]
            utterance = Utterance(eafob.media_path, eaf_path, utter_id,
                                  start_time, end_time, text, participant)
            tier_utterances.append(utterance)

        return tier_utterances

    eaf = Eaf(eaf_path)
    utterances = []
    # TODO potential bug if tier_prefixes has prefixes that are common to a
    # given tier_name.
    for tier_name in eaf.tiers:
        for tier_prefix in tier_prefixes:
            if tier_name.startswith(tier_prefix):
                utterances.extend(utterances_from_tier(eaf, tier_name))
    return utterances

def utterances_from_dir(eaf_dir: Path, tier_prefixes: List[str]) -> List[Utterance]:
    """ Returns the utterances found in a directory. """

    utterances = []
    for eaf_path in eaf_dir.glob("**/*.eaf"):
        eaf_utterances = utterances_from_eaf(eaf_path, tier_prefixes)
        utterances.extend(eaf_utterances)
    return utterances

class Corpus(corpus.Corpus):
    def __init__(self, org_dir: Path, tgt_dir: Path,
                 feat_type: str = "fbank", label_type: str = "phonemes",
                 utterance_filter: Callable[Utterance, bool] = None,
                 label_segmenter: Callable[Utterance, Utterance] = None) -> None:
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

        # Is this conditional the right way to do it? I could also test for
        # each file, but perhaps just testing the existence of the directory is
        # best.
        if not tgt_dir.is_dir():
            # Read utterances from org_dir.
            utterances = utterances_from_dir(tgt_dir)

            # Filter utterances based on some criteria (such as codeswitching).
            utterances = [utter for utter in utterances if utterance_filter(utter)]

            # Segment the labels in the utterances appropriately
            utterances = [label_segmenter(utter) for utter in utterances]

            # Writes the utterances to the tgt_dir/label/ dir
            utterance.write_text(utterances, label_dir, label_type)

            # Extracts utterance level WAV information from the input file.
            wav.extract_wavs(utterances, wav_dir)

            # If we're being fed a segment_labels function rather than the actual
            # labels, then we do actually have to determine all the labels by
            # reading the utterances. A natural way around this is to make the
            # label_segmenter an immutable class (say, a NamedTuple) which stores
            # the labels etc.

            # labels = determine_labels(utterances)

        super().__init__(feat_type, label_type, tgt_dir, label_segmenter.labels)
