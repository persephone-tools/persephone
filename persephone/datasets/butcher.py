"""
Provides a Corpus interface to Kunwinjku data (the Butcher corpus for now;
Steven's data soon).
"""

from collections import namedtuple
import os
from os.path import join
import random


from .. import config
from .. import corpus
from .. import feat_extract
from .. import utils

TGT_DIR = join(config.TGT_DIR, "butcher/kun")
LABEL_DIR = join(TGT_DIR, "labels")
FEAT_DIR = join(TGT_DIR, "feats")
#prepare_butcher_feats("fbank", feat_dir)
#prepare_butcher_labels(label_dir)

def prepare_butcher_labels(label_dir=LABEL_DIR):
    """ Prepares target labels as phonemes """
    # TODO offer label format that is consistent with Steven's
    # data; perhaps by using the orthographic form and lowercasing.

    from nltk_contrib.textgrid import TextGrid
    ## TODO: change import to use
    # from pympi.praat import TextGrid

    if not os.path.isdir(label_dir):
        os.makedirs(label_dir)

    for fn in os.listdir(config.BUTCHER_DIR):
        path = join(config.BUTCHER_DIR, fn)
        if path.endswith(".TextGrid"):
            prefix = os.path.basename(os.path.splitext(path)[0])
            out_path = join(label_dir, prefix+".phonemes")
            with open(path) as f, open(out_path, "w") as out_f:
                tg = TextGrid(f.read())
                for tier in tg:
                    if tier.nameid == "etic":
                        transcript = [text for _, _, text in tier.simple_transcript
                                      if text != "<p:>"]
                        print(" ".join(transcript).strip(), file=out_f)

def get_prefixes():
    """ Gets the prefixes for utterances in this corpus."""

    fns = [prefix for prefix in os.listdir(config.BUTCHER_DIR)
           if prefix.endswith(".TextGrid")]
    prefixes = [os.path.splitext(fn)[0] for fn in fns]
    return prefixes

def filter_unlabelled(prefixes, label_type):
    filtered_prefixes = []
    for prefix in prefixes:
        with open(join(LABEL_DIR, "%s.%s" % (prefix, label_type))) as f:
            if len(f.readline().split()) != 0:
                filtered_prefixes.append(prefix)
            else:
                print('The labels of utterance "{prefix}" '\
                      'with label type "{label_type}" are empty. Removed ' \
                      'from utterance set.'.format(prefix=prefix,
                                               label_type=label_type))
    return filtered_prefixes

class Corpus(corpus.AbstractCorpus):
    """ Interface to the Kunwinjku data. """

    # TODO Do I make these attributes non-caps?
    FEAT_DIR = FEAT_DIR
    LABEL_DIR = LABEL_DIR

    def __init__(self, feat_type="fbank", label_type="phonemes"):
        super().__init__(feat_type, label_type)

        self.labels = butcher_phonemes(LABEL_DIR)
        train, valid, test = make_data_splits(LABEL_DIR)

        # Filter out prefixes that have no transcription. It's probably better
        # to have this after the splitting between train/valid/test sets,
        # because having it before means minor changes in the way labelling
        # occurs would lead to drastically different training sets.
        self.train_prefixes = filter_unlabelled(train, label_type)
        self.valid_prefixes = filter_unlabelled(valid, label_type)
        self.test_prefixes = filter_unlabelled(test, label_type)

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
