import os
import random
import sys

import numpy
import tensorflow as tf

random.seed(0)

def len_longest_timit_utterance(path="/home/oadams/mam/data/timit"):
    """ Finds the number of frames in the longest utterance in TIMIT, so that
    we can zero-pad the other utterances appropriately."""

    if os.path.exists("max_len.txt"):
        with open("max_len.txt") as f:
            return f.readline()

    max_len = 0

    for root, dirnames, filenames in os.walk(path):
        for fn in filenames:
            prefix = fn.split(".")[0] # Get the file prefix
            print(fn)
            if fn.endswith(".lmfb_d_dd.feat"):
                path = os.path.join(root,fn)
                x = numpy.loadtxt(path)
                if x.shape[0] > max_len:
                    max_len = x.shape[0]
    print(max_len)
    with open("max_len.txt", "w") as f:
        f.write(max_len)


def load_timit(path="/home/oadams/mam/data/timit", rand=True, batch_size=100):
    """ Load the already preprocessed TIMIT data. """

    train_paths = []

    for root, dirnames, filenames in os.walk(path):
        for fn in filenames:
            prefix = fn.split(".")[0] # Get the file prefix
            print(fn)
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

#load_timit()
len_longest_timit_utterance()
