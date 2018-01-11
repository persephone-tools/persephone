import os
from os.path import join

from nltk_contrib.textgrid import TextGrid

import config
import corpus
import feat_extract

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

label_dir = join(config.TGT_DIR, "butcher/kun/labels")
feat_dir = join(config.TGT_DIR, "butcher/kun/feats")
#prepare_butcher_feats("fbank", feat_dir)
prepare_butcher_labels(label_dir)

class Corpus(corpus.AbstractCorpus):
    """ Interface to the Kunwinjku data. """
    pass
