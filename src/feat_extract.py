""" Performs feature extraction of WAV files for acoustic modelling."""

import os
import subprocess

import numpy as np
import python_speech_features
import scipy.io.wavfile as wav

import config

def extract_energy(rate, sig):
    """ Extracts the energy of frames. """

    mfcc = python_speech_features.mfcc(sig, rate, appendEnergy=True)
    energy_row_vec = mfcc[:, 0]
    energy_col_vec = energy_row_vec[:, np.newaxis]
    return energy_col_vec

def fbank(wav_path, flat=True):
    """ Currently grabs log Mel filterbank, deltas and double deltas."""

    (rate, sig) = wav.read(wav_path)
    fbank_feat = python_speech_features.logfbank(sig, rate, nfilt=40)
    energy = extract_energy(rate, sig)
    feat = np.hstack([energy, fbank_feat])
    delta_feat = python_speech_features.delta(feat, 2)
    delta_delta_feat = python_speech_features.delta(delta_feat, 2)
    all_feats = [feat, delta_feat, delta_delta_feat]
    if flat == False:
        all_feats = np.array(all_feats)
        # Make time the first dimension for easy length normalization padding
        # later.
        all_feats = np.swapaxes(all_feats, 0, 1)
        all_feats = np.swapaxes(all_feats, 1, 2)
    else:
        all_feats = np.concatenate(all_feats, axis=1)

    # Log Mel Filterbank, with delta, and double delta
    feat_fn = wav_path[:-3] + "fbank.npy"
    np.save(feat_fn, all_feats)

def mfcc(wav_path):
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

# TODO I'm operating at the directory level for pitch feats, but at the file
# level for other things. Change this? Currently wav normalization occurs
# elsewhere too (in say, datasets.chatin.prepare_feats()). Kaldi expects
# specific wav format, so that should be coupled together with pitch extraction
# here.
def from_dir(dirname, feat_type):
    # If pitch features are needed as part of this, extract them
    if feat_type == "pitch" or feat_type == "fbank_and_pitch":
        kaldi_pitch(FEAT_DIR, FEAT_DIR)

    # Then apply file-wise feature extraction
    for filename in os.listdir(dirname):
        path = os.path.join(dirname, filename)
        if path.endswith(".wav"):
            if feat_type == "fbank":
                fbank(path)
            elif feat_type == "fbank_and_pitch":
                fbank(path)
                prefix = os.path.splitext(filename)[0]
                # TODO
                combine_fbank_pitch(prefix)
            elif feat_type == "pitch":
                # Already extracted pitch at the start of this function.
                pass
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

def kaldi_pitch(wav_dir, feat_dir):
    """ Extract Kaldi pitch features. """

    # Make wav.scp and pitch.scp files
    prefixes = []
    for fn in os.listdir(wav_dir):
        prefix, ext = os.path.splitext(fn)
        if ext == ".wav":
            prefixes.append(prefix)

    wav_scp_path = os.path.join(feat_dir, "wavs.scp")
    with open(wav_scp_path, "w") as wav_scp:
        for prefix in prefixes:
            print(prefix, os.path.join(wav_dir, prefix + ".wav"), file=wav_scp)

    pitch_scp_path = os.path.join(feat_dir, "pitch_feats.scp")
    with open(pitch_scp_path, "w") as pitch_scp:
        for prefix in prefixes:
            print(prefix, os.path.join(feat_dir, prefix + ".pitch.txt"), file=pitch_scp)

    # Call Kaldi pitch feat extraction
    args = [os.path.join(config.KALDI_ROOT, "src/featbin/compute-kaldi-pitch-feats"),
            "scp:%s" % (wav_scp_path), "scp,t:%s" % pitch_scp_path]
    subprocess.run(args)
