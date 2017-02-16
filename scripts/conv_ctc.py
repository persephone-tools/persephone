import os
import random
import sys

import numpy
import tensorflow as tf

random.seed(0)

def load_timit(path="/home/oadams/mam/data/timit", rand=True, batch_size=100):
    """ Load the already preprocessed TIMIT data. """

    train_paths = []

    for root, dirnames, filenames in os.walk(path):
        for fn in filenames:
            prefix = fn.split(".")[0] # Get the file prefix
            if fn.endswith(".lmfb_d_dd.feat"):
                # Then load into x.
                #with open(os.path.join(root, fn)) as feat_f:
                #    x = numpy.loadtxt(feat_f)

                # And grab the labels too
                #with open(os.path.join(root, prefix+".phn")) as phn_f:
                #    y = phn_f.readline().split()
                #    print(y)
                #    sys.exit()

                # Add to filename list.
                path = os.path.join(root,fn)
                train_paths.append(os.path.join(root,fn))

    if rand:
        # Load a random subset of the training set of batch_size 
        path_batch = random.sample(train_paths, batch_size)
        y = [path.split("/")[-3] for path in path_batch]
        feats = [numpy.loadtxt(path) for path in path_batch]
        #numpy.array(
    else:
        raise Exception("Not implemented.")

    print(len(feats))
    print(type(feats[0]))
    print(feats[0].shape)
    print(feats[1].shape)
    print(feats[2].shape)
    batch_x = numpy.array(feats)
    print(batch_x.shape)
    print(len(train_paths))
    print(y)

load_timit()
