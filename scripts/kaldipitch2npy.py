import os
import sys
import numpy as np

fn = sys.argv[1]

assert fn.endswith(".pitch.txt")


pitch_feats = []
with open(fn) as f:
    for line in f:
        sp = line.split()
        if len(sp) > 1:
            pitch_feats.append([float(sp[0]), float(sp[1])])

out_fn = os.path.splitext(fn)[0] + ".npy"
a = np.array(pitch_feats)
print(out_fn)
np.save(out_fn, a)
