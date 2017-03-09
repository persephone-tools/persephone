""" Preprocesses the TIMIT data into a format for end-to-end phoneme recognition."""

import python_speech_features
import os
import numpy as np
import scipy.io.wavfile as wav
import shutil
import subprocess
from os.path import join
import config
from utils import zero_pad

def all_feature_extraction():
    """ Walk over all the wav files in the TIMIT dir and extract features. """

    def extract_energy(rate, sig):
        """ Extracts the energy of frames. """
        mfcc = python_speech_features.mfcc(sig, rate, appendEnergy=True)
        energy_row_vec = mfcc[:,0]
        energy_col_vec = energy_row_vec[:, np.newaxis]
        return energy_col_vec

    def logfbank_feature_extraction(wav_path):
        """ Currently grabs log Mel filterbank, deltas and double deltas."""
        (rate, sig) = wav.read(wav_path)
        fbank_feat = python_speech_features.logfbank(sig, rate, nfilt=40)
        energy = extract_energy(rate, sig)
        feat = np.hstack([energy, fbank_feat])
        delta_feat = python_speech_features.delta(feat, 2)
        delta_delta_feat = python_speech_features.delta(delta_feat, 2)
        l = [feat, delta_feat, delta_delta_feat]
        all_feats = np.array(l)
        # Make time the first dimension for easy length normalization padding later.
        all_feats = np.swapaxes(all_feats, 0, 1)
        all_feats = np.swapaxes(all_feats, 1, 2)

        # Log Mel Filterbank, with delta, and double delta
        feat_fn = wav_path[:-3] + "log_mel_filterbank.npy"
        np.save(feat_fn, all_feats)

    def feature_extraction(wav_path):
        """ Grabs MFCC features with energy and derivates. """

        (rate, sig) = wav.read(wav_path)
        feat = python_speech_features.mfcc(sig, rate, appendEnergy=True)
        delta_feat = python_speech_features.delta(feat, 2)
        l = [feat, delta_feat]
        all_feats = np.array(l)
        # Make time the first dimension for easy length normalization padding later.
        all_feats = np.swapaxes(all_feats, 0, 1)
        all_feats = np.swapaxes(all_feats, 1, 2)

        feat_fn = wav_path[:-3] + "mfcc13_d.npy"
        np.save(feat_fn, all_feats)

    for root, dirs, fns in os.walk(config.TGT_DIR):
        print("Processing speaker %s" % root)
        for fn in fns:
            if fn.endswith(".wav"):
                logfbank_feature_extraction(join(root, fn))

def all_zero_pad():
    """ Pads all the utterance features with zeros along the time dimension so
    that each utterance is as long as the longest one, with silence added."""

    def len_longest_timit_utterance(path=config.TGT_DIR):
        """ Finds the number of frames in the longest utterance in TIMIT, so that
        we can zero-pad the other utterances appropriately."""

        if os.path.exists("max_len.txt"):
            with open("max_len.txt") as f:
                return int(f.readline())

        max_len = 0

        for root, dirnames, filenames in os.walk(path):
            for fn in filenames:
                prefix = fn.split(".")[0] # Get the file prefix
                if fn.endswith(".log_mel_filterbank.npy"):
                    path = join(root,fn)
                    x = np.load(path)
                    if x.shape[0] > max_len:
                        max_len = x.shape[0]

        with open("max_len.txt", "w") as f:
            f.write(str(max_len))

        return max_len

    def zero_pad(a, to_length):
        """ Zero pads along the 0th dimension to make sure the utterance array
        x is of length to_length."""

        assert a.shape[0] <= to_length
        result = np.zeros((to_length,) + a.shape[1:])
        result[:a.shape[0]] = a
        return result

    max_len = len_longest_timit_utterance()

    for root, dirs, fns in os.walk(config.TGT_DIR):
        print("Padding utterances for speaker %s" % "/".join(root.split("/")[-3:]))
        for fn in fns:
            if fn.endswith(".log_mel_filterbank.npy"):
                path = join(root, fn)
                x = np.load(path)
                x = zero_pad(x, max_len)
                np.save(join(
                        root, fn.split(".")[0] + ".log_mel_filterbank.zero_pad.npy"),
                        x)

def create_raw_data():
    """ Copies the original TIMIT data to a working directory and does basic
    preprocessing such as removing phone timestamps and converting NIST files
    to WAV."""

    def preprocess_phones(org_path, tgt_path):
        """ Takes the org_path .phn file, preprocesses and writes to tgt_path."""
        with open(org_path) as in_f, open(tgt_path, "w") as out_f:
            phones = []
            for line in in_f:
                phones.append(line.strip().split()[-1])
            print(" ".join(phones), file=out_f)

    def sph2wav(sphere_path, wav_path):
        """ Calls sox to convert the sphere file to wav."""
        args = [config.SOX_PATH, sphere_path, wav_path]
        subprocess.run(args)

    for root, dirs, fns in os.walk(config.ORG_DIR):
        for fn in fns:
            org_path = join(root, fn)
            sub_path = join(root[len(config.ORG_DIR):], fn)
            tgt_path = join(config.TGT_DIR, sub_path)

            # Create parent directory
            parent_path = os.path.dirname(tgt_path)
            if not os.path.exists(parent_path):
                os.makedirs(parent_path)

            print("Creating raw data for %s" % tgt_path)

            if fn.endswith("phn"):
                preprocess_phones(org_path, tgt_path)
            elif fn.endswith("wav"):
                # It's actually in sphere format
                sphere_path = tgt_path[:-3]+"sph"
                shutil.copyfile(org_path, sphere_path)
                sph2wav(sphere_path, tgt_path)

if __name__ == "__main__":
    #create_raw_data()
    all_feature_extraction()
    #all_zero_pad()
