""" Testing Persephone on Chatino """

import logging
import os
from os.path import splitext
from pathlib import Path
import pprint
import subprocess
from typing import List, Set

import pint
import pytest
from pympi.Elan import Eaf

from persephone import config
from persephone import corpus
from persephone import model
from persephone import utterance
from persephone.utterance import Utterance
from persephone.datasets import bkw
from persephone.model import Model
from persephone.preprocess import elan
from persephone.preprocess.labels import LabelSegmenter
from persephone.corpus_reader import CorpusReader
from persephone.experiment import prep_exp_dir
from persephone import rnn_ctc
from persephone import utils

ureg = pint.UnitRegistry()

logging.config.fileConfig(config.LOGGING_INI_PATH)

@pytest.fixture
def chatino_labels():
    """ Hard coding chatino labels so that decoding can be done without Corpus objects."""

    # Hardcode the phoneme set
    ONE_CHAR_PHONEMES = set(["p", "t", "d", "k", "q", "s", "x", "j", "m", "n", "r", "l",
                             "y", "w", "i", "u", "e", "o", "a"])
    TWO_CHAR_PHONEMES = set(["ty", "dy", "kw", "ts", "dz", "ch", "ny", "ly",
                             "in", "en", "On", "an"])
    PHONEMES = ONE_CHAR_PHONEMES.union(TWO_CHAR_PHONEMES)

    # Hardcode the tone set
    ONE_CHAR_TONES = set(["0", "3", "1", "4", "2"])
    TWO_CHAR_TONES = set(["04", "14", "24", "20", "42", "32", "40", "10"])
    THREE_CHAR_TONES = set(["140"])
    TONES = ONE_CHAR_TONES.union(TWO_CHAR_TONES.union(THREE_CHAR_TONES))
    LABELS = PHONEMES.union(TONES)

    return LABELS

@pytest.mark.experiment
def test_decode(chatino_labels):
    model_path_prefix = "/home/oadams/mam/exp/609/0/model/model_best.ckpt"
    valid_prefix_path = "/home/oadams/mam/data/chatino/valid_prefixes.txt"
    with open(valid_prefix_path) as valid_prefix_f:
        valid_prefixes = [prefix.strip() for prefix in valid_prefix_f.readlines()]
    valid_feat_paths = ["/home/oadams/mam/data/chatino/feat/{}.fbank.npy".format(prefix)
                      for prefix in valid_prefixes]
    logging.debug("valid_feat_paths: {}".format(valid_feat_paths[:10]))
    transcripts = model.decode(model_path_prefix,
                               valid_feat_paths,
                               chatino_labels)
    logging.debug("transcripts: {}".format(pprint.pformat(
        [" ".join(transcript) for transcript in transcripts])))

def segment_utterance(utterance: Utterance) -> Utterance:
    fields = utterance._asdict()
    fields["text"] = segment_str(fields["text"], phoneme_inventory)
    return Utterance(**fields)

def segment_str(text: str, phoneme_inventory: Set[str]) -> str:
    """
    Takes as input a string in Kunwinjku and segments it into phoneme-like
    units based on the standard orthographic rules specified at
    http://bininjgunwok.org.au/
    """

    #text = text.lower()
    text = segment_into_tokens(text, phoneme_inventory)
    return text

@pytest.fixture
def joes_data(chatino_labels):
    chatino_label_segmenter = LabelSegmenter(segment_utterance, chatino_labels)
    joe_corpus = corpus.Corpus.from_elan(
                  org_dir = Path("/home/oadams/mam/org_data/joes_chatino/"),
                  tgt_dir = Path("/home/oadams/mam/data/joes_chatino/"),
                  feat_type = "fbank", label_type = "phonemes_and_tones",
                  label_segmenter = chatino_label_segmenter,
                  tier_prefixes = ("Chatino",))
    return joe_corpus

def test_transcribe_joes_data(chatino_labels, joes_data):
    """ This test uses a model stored on a Unimelb server."""
    model_path_prefix = "/home/oadams/mam/exp/609/0/model/model_best.ckpt"
