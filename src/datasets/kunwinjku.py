"""
Provides a Corpus interface to Kunwinjku data (the Butcher corpus for now;
Steven's data soon).
"""

from collections import namedtuple
import os
from os.path import join
import random

from nltk_contrib.textgrid import TextGrid

import config
import corpus
import feat_extract
import utils

TGT_DIR = join(config.TGT_DIR, "butcher/kun")
LABEL_DIR = join(TGT_DIR, "labels")
FEAT_DIR = join(TGT_DIR, "feats")
#prepare_butcher_feats("fbank", feat_dir)
#prepare_butcher_labels(label_dir)

def prepare_butcher_labels(label_dir=LABEL_DIR):
    """ Prepares target labels as phonemes """
    # TODO offer label format that is consistent with Steven's
    # data; perhaps by using the orthographic form and lowercasing.

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

def prepare_butcher_feats(feat_type, feat_dir):
    """ Prepares input features"""
    # TODO Could probably be factored out; there's nothing so corpus-specific
    # here.

    if not os.path.isdir(feat_dir):
        os.makedirs(feat_dir)

    for fn in os.listdir(config.BUTCHER_DIR):
        path = join(config.BUTCHER_DIR, fn)
        if path.endswith(".wav"):
            prefix = os.path.basename(os.path.splitext(path)[0])
            mono16k_wav_path = join(feat_dir, prefix+".wav")
            if not os.path.isfile(mono16k_wav_path):
                feat_extract.convert_wav(path, mono16k_wav_path)

    feat_extract.from_dir(feat_dir, feat_type)

def make_data_splits(label_dir, max_samples=1000, seed=0):
    """ Splits the utterances into training, validation and test sets."""

    train_prefix_fn = join(TGT_DIR, "train_prefixes.txt")
    valid_prefix_fn = join(TGT_DIR, "valid_prefixes.txt")
    test_prefix_fn = join(TGT_DIR, "test_prefixes.txt")

    train_f_exists = os.path.isfile(train_prefix_fn)
    valid_f_exists = os.path.isfile(valid_prefix_fn)
    test_f_exists = os.path.isfile(test_prefix_fn)

    if train_f_exists and valid_f_exists and test_f_exists:
        with open(train_prefix_fn) as train_f:
            train_prefixes = [line.strip() for line in train_f]
        with open(valid_prefix_fn) as valid_f:
            valid_prefixes = [line.strip() for line in valid_f]
        with open(test_prefix_fn) as test_f:
            test_prefixes = [line.strip() for line in test_f]

        return train_prefixes, valid_prefixes, test_prefixes

    prefixes = get_prefixes()
    # TODO Note that I'm shuffling after sorting; this could be better.
    # TODO Remove explicit reference to "fbank"
    prefixes = utils.filter_by_size(
        FEAT_DIR, prefixes, "fbank", max_samples)
    Ratios = namedtuple("Ratios", ["train", "valid", "test"])
    ratios = Ratios(.80, .10, .10)
    train_end = int(ratios.train*len(prefixes))
    valid_end = int(train_end + ratios.valid*len(prefixes))
    random.shuffle(prefixes)
    train_prefixes = prefixes[:train_end]
    valid_prefixes = prefixes[train_end:valid_end]
    test_prefixes = prefixes[valid_end:]

    with open(train_prefix_fn, "w") as train_f:
        for prefix in train_prefixes:
            print(prefix, file=train_f)
    with open(valid_prefix_fn, "w") as dev_f:
        for prefix in valid_prefixes:
            print(prefix, file=dev_f)
    with open(test_prefix_fn, "w") as test_f:
        for prefix in test_prefixes:
            print(prefix, file=test_f)

    return train_prefixes, valid_prefixes, test_prefixes

def butcher_phonemes(label_dir=LABEL_DIR):
    """ Returns a set of phonemes found in the corpus. """
    phonemes = set()
    for fn in os.listdir(label_dir):
        with open(join(label_dir, fn)) as f:
            line_phonemes = set(f.readline().split())
            phonemes = phonemes.union(line_phonemes)
    return phonemes

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
