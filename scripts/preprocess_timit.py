""" Preprocesses the TIMIT data into a format for end-to-end phoneme recognition."""

import python_speech_features
import os
import numpy as np
import scipy.io.wavfile as wav
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

def all_feature_extraction():
    """ Walk over all the wav files in the TIMIT dir and extract features. """

    def feature_extraction(wav_path):
        """ Currently grabs log Mel filterbank, deltas and double deltas."""
        (rate, sig) = wav.read(wav_path)
        fbank_feat = python_speech_features.logfbank(sig, rate)
        delta_feat = python_speech_features.delta(fbank_feat, 2)
        delta_delta_feat = python_speech_features.delta(delta_feat, 2)
        all_feats = np.concatenate(
                (fbank_feat, delta_feat, delta_delta_feat), axis=1)
        # Log Mel Filterbank, with delta, and double delta
        feat_fn = wav_path[:-3] + "lmfb_d_dd.feat"
        np.savetxt(feat_fn, all_feats)

    for root, dirs, fns in os.walk(TGT_DIR):
        for fn in fns:
            if fn.endswith(".wav"):
                print(join(root, fn))
                feature_extraction(join(root, fn))

def all_zero_pad():

    def len_longest_timit_utterance(path=TGT_DIR):
        """ Finds the number of frames in the longest utterance in TIMIT, so that
        we can zero-pad the other utterances appropriately."""

        if os.path.exists("max_len.txt"):
            with open("max_len.txt") as f:
                return int(f.readline())

        max_len = 0

        for root, dirnames, filenames in os.walk(path):
            for fn in filenames:
                prefix = fn.split(".")[0] # Get the file prefix
                if fn.endswith(".lmfb_d_dd.feat"):
                    path = join(root,fn)
                    x = np.loadtxt(path)
                    if x.shape[0] > max_len:
                        max_len = x.shape[0]
        print(max_len)
        with open("max_len.txt", "w") as f:
            f.write(str(max_len))

    def zero_pad(a, to_length):
        """ Zero pads along the 0th dimension to make sure the utterance array
        x is of length to_length."""

        assert a.shape[0] <= to_length
        result = np.zeros((to_length,) + a.shape[1:])
        result[:a.shape[0]] = a
        return result

    max_len = len_longest_timit_utterance()

    for root, dirs, fns in os.walk(TGT_DIR):
        print("Preprocessing for speaker %s" % "/".join(root.split("/")[-3:]))
        for fn in fns:
            if fn.endswith(".lmfb_d_dd.feat"):
                path = join(root, fn)
                x = np.loadtxt(path)
                x = zero_pad(x, max_len)
                np.savetxt(join(
                        root, fn.split(".")[0] + ".lmfb_d_dd.zero_pad.feat"),
                        x)

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

if __name__ == "__main__":
    #preprocess()
    all_feature_extraction()
