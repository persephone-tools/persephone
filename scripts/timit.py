""" Serves as an interface to the TIMIT data. """

import os
import numpy as np
import random

# Hardcoded numbers
num_timit_phones = 61
eval_phones = 39

def phone_classes(path="/home/oadams/mam/data/timit/train"):
    """ Returns a sorted list of phone classes observed in the TIMIT corpus."""

    train_paths = []
    for root, dirnames, filenames in os.walk(path):
        for fn in filenames:
            prefix = fn.split(".")[0] # Get the file prefix
            if fn.endswith(".log_mel_filterbank.zero_pad.npy"):
                # Add to filename list.
                path = os.path.join(root,fn)
                train_paths.append(os.path.join(root,fn))
    phn_paths = ["".join(path.split(".")[:-3])+".phn" for path in train_paths]
    phone_set = set()
    for phn_path in phn_paths:
        with open(phn_path) as phn_f:
            for phone in phn_f.readline().split():
                phone_set.add(phone)

    assert len(phone_set) == num_timit_phones
    return sorted(list(phone_set))

def batch_gen(path="/home/oadams/mam/data/timit/train", rand=True,
        batch_size=100, labels="phonemes"):
    """ Load the already preprocessed TIMIT data. """

    def create_one_hot(i, num_classes):
        """ Takes a list or array with numbers representing classes, and
        creates a corresponding 2-dimensional array of one-hot vectors."""

        one_hot = np.zeros((1,num_classes))
        one_hot[0,i] = 1
        return one_hot

    train_paths = []
    dialect_classes = set()

    for root, dirnames, filenames in os.walk(path):
        for fn in filenames:
            prefix = fn.split(".")[0] # Get the file prefix
            if fn.endswith(".log_mel_filterbank.zero_pad.npy"):
                # Add to filename list.
                path = os.path.join(root,fn)
                train_paths.append(os.path.join(root,fn))
                dialect_classes.add(path.split("/")[-3])

    dialect_classes = sorted(list(dialect_classes))
    dr_class_map = dict(zip(dialect_classes, range(0,len(dialect_classes))))

    path_batches = [train_paths[i:i+batch_size] for i in range(
            0,len(train_paths),batch_size)]

    if rand:
        random.shuffle(path_batches)

    for path_batch in path_batches:
        feats = [np.load(path) for path in path_batch]
        x = np.array(feats)
        if labels == "dialects":
            y_labels = np.array([dr_class_map[path.split("/")[-3]] for path in path_batch])
            # Make into one-hot vectors
            batch_y = np.concatenate([create_one_hot(label,len(dialect_classes)) for label in y_labels])
        elif labels == "phonemes":
            phn_paths = ["".join(path.split(".")[:-3])+".phn" for path in path_batch]
            batch_y = []
            for phn_path in phn_paths:
                with open(phn_path) as phn_f:
                    batch_y.append(phn_f.readline().split())

        batch_x = np.array(feats)
        yield batch_x, batch_y
