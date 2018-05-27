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
import persephone.preprocess.labels
from persephone.preprocess.labels import LabelSegmenter
from persephone.corpus_reader import CorpusReader
from persephone.experiment import prep_exp_dir
from persephone import rnn_ctc
from persephone import utils
import persephone.results

ureg = pint.UnitRegistry()

logging.config.fileConfig(config.LOGGING_INI_PATH)

def chatino_phonemes():
    # Hardcode the phoneme set
    ONE_CHAR_PHONEMES = set(["p", "t", "d", "k", "q", "s", "x", "j", "m", "n", "r", "l",
                             "y", "w", "i", "u", "e", "o", "a"])
    TWO_CHAR_PHONEMES = set(["ty", "dy", "kw", "ts", "dz", "ch", "ny", "ly",
                             "in", "en", "On", "an"])
    PHONEMES = ONE_CHAR_PHONEMES.union(TWO_CHAR_PHONEMES)
    return PHONEMES

def chatino_tones():
    # Hardcode the tone set
    ONE_CHAR_TONES = set(["0", "3", "1", "4", "2"])
    TWO_CHAR_TONES = set(["04", "14", "24", "20", "42", "32", "40", "10"])
    THREE_CHAR_TONES = set(["140"])
    TONES = ONE_CHAR_TONES.union(TWO_CHAR_TONES.union(THREE_CHAR_TONES))
    return TONES

def chatino_labels():
    """ Hard coding chatino labels so that decoding can be done without Corpus objects."""

    LABELS = chatino_phonemes().union(chatino_tones())
    return LABELS

@pytest.mark.experiment
def test_decode():
    labels = chatino_labels()
    model_path_prefix = "/home/oadams/mam/exp/609/0/model/model_best.ckpt"
    valid_prefix_path = "/home/oadams/mam/data/chatino/valid_prefixes.txt"
    with open(valid_prefix_path) as valid_prefix_f:
        valid_prefixes = [prefix.strip() for prefix in valid_prefix_f.readlines()]
    valid_feat_paths = ["/home/oadams/mam/data/chatino/feat/{}.fbank.npy".format(prefix)
                      for prefix in valid_prefixes]
    logging.debug("valid_feat_paths: {}".format(valid_feat_paths[:10]))
    transcripts = model.decode(model_path_prefix,
                               valid_feat_paths,
                               labels)
    logging.debug("transcripts: {}".format(pprint.pformat(
        [" ".join(transcript) for transcript in transcripts])))

def segment_utterance(utterance: Utterance) -> Utterance:
    fields = utterance._asdict()
    fields["text"] = persephone.preprocess.labels.segment_into_tokens(fields["text"], chatino_labels())
    return Utterance(**fields)

@pytest.fixture
def joes_data():
    labels = chatino_labels()
    chatino_label_segmenter = LabelSegmenter(segment_utterance, labels)
    joe_corpus = corpus.Corpus.from_elan(
                  org_dir = Path("/home/oadams/mam/org_data/joes_chatino/"),
                  tgt_dir = Path("/home/oadams/mam/data/joes_chatino/"),
                  feat_type = "fbank", label_type = "phonemes_and_tones",
                  label_segmenter = chatino_label_segmenter,
                  tier_prefixes = ("Chatino",))
    return joe_corpus

def test_transcribe_joes_data(joes_data):
    """ This test uses a model stored on a Unimelb server."""
    model_path_prefix = "/home/oadams/mam/exp/609/0/model/model_best.ckpt"
    labels = chatino_labels()
    joe_corpus = joes_data
    # Create a list of fbank files in Joe's corpus for transcription.
    joe_fbank_fns = (joe_corpus.get_train_fns()[0] +
                     joe_corpus.get_valid_fns()[0] + 
                     joe_corpus.get_test_fns()[0])
    #joe_corpus_fbank = joe_corpus.
    logging.debug("joe_fbank_fns: {}".format(joe_fbank_fns))
    hyps = model.decode(model_path_prefix,
                               joe_fbank_fns,
                               labels)
    logging.debug("transcripts: {}".format(pprint.pformat(
        [" ".join(transcript) for transcript in hyps])))
    label_fns = (joe_corpus.get_train_fns()[1] +
                 joe_corpus.get_valid_fns()[1] +
                 joe_corpus.get_test_fns()[1])
    prefixes = (joe_corpus.train_prefixes +
                 joe_corpus.valid_prefixes +
                 joe_corpus.test_prefixes)
    refs = []
    for fn in label_fns:
        with open(fn) as f:
            ref = f.read().strip()
            refs.append(ref.split())
    with open("hyps.txt", "w") as hyps_f:
        for hyp in hyps:
            print(" ".join(hyp), file=hyps_f)
    with open("refs.txt", "w") as refs_f:
        for ref in refs:
            print(" ".join(ref), file=refs_f)
    logging.debug("hyps refs: {}".format(list(zip(hyps, refs))))
    ler = persephone.results.filtered_error_rate("hyps.txt", "refs.txt", labels)
    per = persephone.results.filtered_error_rate("hyps.txt", "refs.txt", chatino_phonemes())
    #ter = persephone.results.filtered_error_rate("hyps.txt", "refs.txt", chatino_tones())
    logging.debug("LER: {}".format(ler))
    logging.debug("PER: {}".format(per))
    #logging.debug("TER: {}".format(ter))
    persephone.results.fmt_latex_output(hyps, refs, prefixes, Path("fmt_output.tex"))

