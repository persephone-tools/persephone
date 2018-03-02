""" Performs feature extraction of WAV files for acoustic modelling."""

import os
from pathlib import Path
import subprocess

import numpy as np
import python_speech_features
import scipy.io.wavfile as wav

from .. import config
from ..exceptions import PersephoneException

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

def combine_fbank_and_pitch(feat_dir, prefix):

    fbank_fn = os.path.join(feat_dir, prefix + ".fbank.npy")
    fbanks = np.load(fbank_fn)
    pitch_fn = os.path.join(feat_dir, prefix + ".pitch.npy")
    pitches = np.load(pitch_fn)

    # Check that the fbanks are flat
    if len(fbanks.shape) == 3:
        def flatten(feats_3d):
            swapped = np.swapaxes(feats_3d, 0, 1)
            concatenated = np.concatenate(swapped, axis=1)
            return concatenated
        fbanks = flatten(fbanks)
    elif len(fbanks.shape) == 2:
        pass
    else:
        raise PersephoneException("Invalid fbank array shape %s" % (str(fbanks.shape)))

    diff = len(fbanks) - len(pitches)

    # When using python_speech_features fbank extraction and kaldi pitch
    # extraction, there are slightly differing numbers of frames. Usually only
    # by 1 or so. Just pad with zeros to match.
    # TODO This could be a sign that it might just be best to use Kaldi for
    # fbank feature extraction as well (I kind of want to see how using those
    # features goes anyway). But I'm currently keeping it this way for
    # experimental consistency.
    if diff > 2:
        raise PersephoneException("Excessive difference in number of frames. %d" % diff)
    elif diff > 0:
        pitches = np.concatenate((np.array([[0,0]]*(len(fbanks) - len(pitches))), pitches))
    fbank_pitch_feats = np.concatenate((fbanks, pitches), axis=1)

    out_fn = os.path.join(feat_dir, prefix + ".fbank_and_pitch.npy")
    np.save(out_fn, fbank_pitch_feats)

# TODO I'm operating at the directory level for pitch feats, but at the file
# level for other things. Change this? Currently wav normalization occurs
# elsewhere too (in say, datasets.chatin.prepare_feats()). Kaldi expects
# specific wav format, so that should be coupled together with pitch extraction
# here.
def from_dir(dirpath: Path, feat_type: str) -> None:
    """ Performs feature extraction from the WAV files in a directory. """

    dirname = str(dirpath)

    def all_wavs_processed():
        """
        True if all wavs in the directory have corresponding numpy feature
        file; False otherwise.
        """

        for fn in os.listdir(dirname):
            prefix, ext = os.path.splitext(fn)
            if ext == ".wav":
                if not os.path.exists(
                        os.path.join(dirname, "%s.%s.npy" % (prefix, feat_type))):
                    return False
        return True

    if all_wavs_processed():
        # Then nothing needs to be done here
        return
    # Otherwise, go on and process everything...

    # If pitch features are needed as part of this, extract them
    if feat_type == "pitch" or feat_type == "fbank_and_pitch":
        kaldi_pitch(dirname, dirname)

    # Then apply file-wise feature extraction
    for filename in os.listdir(dirname):
        print("Preparing %s features for %s" % (feat_type, filename))
        path = os.path.join(dirname, filename)
        if path.endswith(".wav"):
            if feat_type == "fbank":
                fbank(path)
            elif feat_type == "fbank_and_pitch":
                fbank(path)
                prefix = os.path.splitext(filename)[0]
                combine_fbank_and_pitch(dirname, prefix)
            elif feat_type == "pitch":
                # Already extracted pitch at the start of this function.
                pass
            elif feat_type == "mfcc13_d":
                mfcc(path)
            else:
                raise PersephoneException("Feature type not found: %s" % feat_type)

def convert_wav(org_wav_fn: Path, tgt_wav_fn: Path) -> None:
    """ Converts the wav into a 16bit mono 16000Hz wav."""
    args = [config.FFMPEG_PATH,
            "-i", str(org_wav_fn), "-ac", "1", "-ar", "16000", str(tgt_wav_fn)]
    subprocess.run(args)

def kaldi_pitch(wav_dir, feat_dir):
    """ Extract Kaldi pitch features. Assumes 16k mono wav files."""

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

    # Convert the Kaldi pitch *.txt files to numpy arrays.
    for fn in os.listdir(feat_dir):
        if fn.endswith(".pitch.txt"):
            pitch_feats = []
            with open(os.path.join(feat_dir, fn)) as f:
                for line in f:
                    sp = line.split()
                    if len(sp) > 1:
                        pitch_feats.append([float(sp[0]), float(sp[1])])
            prefix, _ = os.path.splitext(fn)
            out_fn = prefix + ".npy"
            a = np.array(pitch_feats)
            np.save(os.path.join(feat_dir, out_fn), a)
