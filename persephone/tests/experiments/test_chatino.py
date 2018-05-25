""" Testing Persephone on Chatino """

import logging
import os
from os.path import splitext
from pathlib import Path
import pprint
import subprocess
from typing import List

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

#@pytest.mark.experiment
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
