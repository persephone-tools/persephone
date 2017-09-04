import sys

import numpy as np

import utils

fbank_fn, pitch_fn, out_fn = sys.argv[1], sys.argv[2], sys.argv[3]

fbanks = np.load(fbank_fn)
pitches = np.load(pitch_fn)

assert len(fbanks.shape) == 3
diff = len(fbanks) - len(pitches)
if abs(diff > 1):
    print(fbank_fn)
    print(len(fbanks), len(pitches))

def flatten(feats_3d):
    swapped = np.swapaxes(feats_3d, 0, 1)
    concatenated = np.concatenate(swapped, axis=1)
    return concatenated

if diff > 0:
    pitches = np.concatenate((np.array([[0,0]]*(len(fbanks) - len(pitches))), pitches))
    print(np.array([[0,0]]*(len(fbanks) - len(pitches))).shape)
    print(pitches.shape)
flat_fbank = flatten(fbanks)
fbank_pitch_feats = np.concatenate((flat_fbank, pitches), axis=1)

np.save(out_fn, fbank_pitch_feats)
