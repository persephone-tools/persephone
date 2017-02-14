""" Preprocesses the TIMIT data into a format for end-to-end phoneme recognition."""

import python_speech_features
import os
import numpy
import scipy.io.wavefile as wav
import shutil
import subprocess
from os.path import join

ORG_DIR = "/lt/data/timit/timit"
TGT_DIR = "/home/oadams/mam/data/timit"
SOX_PATH = "/home/oadams/tools/sox-14.4.2/src/sox"

train_dirs = [d for d in os.scandir(join(ORG_DIR,"train")) if d.is_dir()]

if not os.path.exists(join(TGT_DIR)):
    os.makedirs(join(TGT_DIR, "train"))

def preprocess_phones(org_path, tgt_path):
    """ Takes the org_path .phn file, preprocesses and writes to tgt_path."""
    with open(org_path) as in_f, open(tgt_path, "w") as out_f:
        phones = []
        for line in in_f:
            phones.append(line.strip().split()[-1])
        print(" ".join(phones), file=out_f)

def sph2wav(sphere_path, wav_path):
    """ Calls sox to convert the sphere file to wav."""
    args = [SOX_PATH, sphere_path, wav_path]
    subprocess.run(args)

def feature_extraction(wav_path):
    """ Currently grabs log Mel filterbank, deltas and double deltas."""

    (rate, sig) = wav.read(wav_path)
    fbank_feat = logfbank(sig, rate)
    delta_feat = delta(fbank_feat, 2)
    delta_delta_feat = delta(delta_feat, 2)
    all_feats = numpy.concatenate(
            (fbank_feat, delta_feat, delta_delta_feat), axis=1)

def preprocess():
    """ Calls all the preprocessing over the TIMIT data."""
    dialect_dirs = [entry for entry in os.scandir(join(ORG_DIR,"train"))]
    for dialect_dir in dialect_dirs:
        speaker_dirs = [entry for entry in os.scandir(dialect_dir.path)]
        for speaker_dir in speaker_dirs:
            tgt_speaker_path = join(TGT_DIR, "train", dialect_dir.name, speaker_dir.name)
            if not os.path.exists(tgt_speaker_path):
                os.makedirs(tgt_speaker_path)
            for f in os.scandir(speaker_dir.path):
                fn = f.name
                org_path = join(ORG_DIR, "train", dialect_dir.name, speaker_dir.name, fn)
                tgt_path = join(TGT_DIR, "train", dialect_dir.name, speaker_dir.name, fn)
                if fn.endswith("phn"):
                    preprocess_phones(org_path, tgt_path)
                elif fn.endswith("wav"):
                    # It's actually in sphere format
                    sphere_path = tgt_path[:-3]+"sph"
                    shutil.copyfile(org_path, sphere_path)
                    sph2wav(sphere_path, tgt_path)
                    feature_extraction(tgt_path)

if __name__ == "__main__":
    preprocess()
