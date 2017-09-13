""" Performs feature extraction of WAV files for acoustic modelling."""

import os
import subprocess

import numpy as np
import python_speech_features
import scipy.io.wavfile as wav

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
    all_feats = [feat, delta_feat]
    all_feats = np.array(all_feats)
    # Make time the first dimension for easy length normalization padding later.
    all_feats = np.swapaxes(all_feats, 0, 1)
    all_feats = np.swapaxes(all_feats, 1, 2)

    feat_fn = wav_path[:-3] + "mfcc13_d.npy"
    np.save(feat_fn, all_feats)

def from_dir(dirname, feat_type):
    for filename in os.listdir(dirname):
        path = os.path.join(dirname, filename)
        if path.endswith(".wav"):
            if feat_type == "log_mel_filterbank":
                logfbank_feature_extraction(path)
            elif feat_type == "mfcc13_d":
                feature_extraction(wav)
            else:
                raise Exception("Feature type not found: %s" % feat_type)

def convert_wav(org_wav_fn, tgt_wav_fn):
    """ Converts the wav into a 16bit mono 16000Hz wav."""
    home = os.path.expanduser("~")
    args = [os.path.join(home, "tools", "ffmpeg-3.3", "ffmpeg"),
            "-i", org_wav_fn, "-ac", "1", "-ar", "16000", tgt_wav_fn]
    subprocess.run(args)
