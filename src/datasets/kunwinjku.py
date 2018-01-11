from collections import namedtuple
import os
from os.path import join
import random

from nltk_contrib.textgrid import TextGrid

import config
import corpus
import feat_extract
import utils

def prepare_butcher_labels(label_dir):
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
                        transcript = [text for _, _, text in tier.simple_transcript]
                        print(" ".join(transcript).strip(), file=out_f)

def prepare_butcher_feats(feat_type, feat_dir):
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

TGT_DIR = join(config.TGT_DIR, "butcher/kun")
label_dir = join(TGT_DIR, "labels")
feat_dir = join(TGT_DIR, "feats")
#prepare_butcher_feats("fbank", feat_dir)
#prepare_butcher_labels(label_dir)

def make_data_splits(label_dir, max_samples=1000, seed=0):

    fns = [prefix for prefix in os.listdir(label_dir)
                if prefix.endswith("phonemes")]
    prefixes = [os.path.splitext(fn)[0] for fn in fns]
    # Note that I'm shuffling after sorting; this could be better.
    prefixes = utils.sort_and_filter_by_size(
        feat_dir, prefixes, "fbank", max_samples)
    Ratios = namedtuple("Ratios", ["train", "dev", "test"])
    ratios = Ratios(.80, .10, .10)
    train_end = int(ratios.train*len(prefixes))
    dev_end = int(train_end + ratios.dev*len(prefixes))
    random.shuffle(prefixes)
    train_prefixes = prefixes[:train_end]
    dev_prefixes = prefixes[train_end:dev_end]
    test_prefixes = prefixes[dev_end:]

    with open(join(TGT_DIR, "train_prefixes.txt"), "w") as train_f:
        for prefix in train_prefixes:
            print(prefix, file=train_f)
    with open(join(TGT_DIR, "valid_prefixes.txt"), "w") as dev_f:
        for prefix in train_prefixes:
            print(prefix, file=dev_f)
    with open(join(TGT_DIR, "test_prefixes.txt"), "w") as test_f:
        for prefix in test_prefixes:
            print(prefix, file=test_f)

make_data_splits(label_dir)

class Corpus(corpus.AbstractCorpus):
    """ Interface to the Kunwinjku data. """
    pass
