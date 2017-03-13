""" Preprocesses the TIMIT data into a format for end-to-end phoneme recognition."""

import numpy as np
import os
from os.path import join
import python_speech_features
import scipy.io.wavfile as wav
import shutil
import subprocess

import config

def all_feature_extraction():
    """ Walk over all the wav files in the TIMIT dir and extract features. """

    def extract_energy(rate, sig):
        """ Extracts the energy of frames. """
        mfcc = python_speech_features.mfcc(sig, rate, appendEnergy=True)
        energy_row_vec = mfcc[:, 0]
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
        all_feats = [feat, delta_feat, delta_delta_feat]
        all_feats = np.array(all_feats)
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
        for filename in fns:
            if filename.endswith(".wav"):
                logfbank_feature_extraction(join(root, filename))

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
