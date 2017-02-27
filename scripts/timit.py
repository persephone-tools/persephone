""" Serves as an interface to the TIMIT data. """

import os
import numpy as np
import random

import utils

from nltk.metrics import distance

random.seed(0)

# Hardcoded numbers
num_phones = 61

def phone_classes(path="/home/oadams/code/mam/data/timit/train"):
    """ Returns a sorted list of phone classes observed in the TIMIT corpus."""

    train_paths = []
    for root, dirnames, filenames in os.walk(path):
        for fn in filenames:
            prefix = fn.split(".")[0] # Get the file prefix
            if fn.endswith(".log_mel_filterbank.npy"):
                # Add to filename list.
                path = os.path.join(root,fn)
                train_paths.append(os.path.join(root,fn))
    phn_paths = [path.split(".")[0] + ".phn" for path in train_paths]
    phone_set = set()
    for phn_path in phn_paths:
        with open(phn_path) as phn_f:
            for phone in phn_f.readline().split():
                phone_set.add(phone)

    assert len(phone_set) == num_phones
    return sorted(list(phone_set))

phone_set = phone_classes()
phone_map = {index: phone for index, phone in enumerate(phone_set)}

def collapse_phones(utterance):
    """ Converts an utterance with labels of 61 possible phones to 39. This is
    done as per Kai-fu Lee & Hsiao-Wuen Hon 1989."""

    # Define some groupings of similar phones
    allophone_map = {"ux":"uw",
                 "axr":"er", "ax-h":"ah", "em":"m", "nx":"n", "hv": "hh", "eng": "ng",
                 "q":"t", # Glottal stop -> t
                 "pau":"sil","h#": "sil", "#h": "sil", # Silences
                 "bcl":"vcl", "dcl":"vcl", "gcl":"vcl", # Voiced closures
                 "pcl":"cl", "tcl":"cl", "kcl":"cl", "qcl":"cl", # Unvoiced closures
                } 

    class_map = {"el": "l", "en": "n", "zh": "sh", "ao": "aa", "ix":"ih", "ax":"ah",
            "sil":"sil", "cl":"sil", "vcl":"sil", "epi":"sil"}

    allo_collapse = [(allophone_map[phn] if phn in allophone_map else phn) for phn in utterance]
    class_collapse = [(class_map[phn] if phn in class_map else phn) for phn in allo_collapse]

    return class_collapse
    #return allo_collapse

def per(hypo, ref):
    """ Calculates the phoneme error rate between decoder output and the gold reference by first collapsing the TIMIT labels into the standard 39 phonemes."""

    hypo = collapse_phonemes(hypo)
    ref = collapse_phonemes(ref)
    return distance.edit_distance.eval(hypo, ref)

def indices_to_phones(indices):
    """ Converts integer representations of phones to human-readable characters. """

    return [phone_map[index] for index in indices]

def batch_gen(path="/home/oadams/code/mam/data/timit/train", rand=True,
        batch_size=100, labels="phonemes", total_size=4620, flatten=True):
    """ Load the already preprocessed TIMIT data.  Flatten=True will make the
    2-dimensional freq x time a 1 dimensional vector of feats."""

    def create_one_hot(i, num_classes):
        """ Takes a list or array with numbers representing classes, and
        creates a corresponding 2-dimensional array of one-hot vectors."""

        one_hot = np.zeros((1,num_classes))
        one_hot[0,i] = 1
        return one_hot

    def collapse(batch_x):
        """ Converts timit into an array of format (batch_size, freq, time). Except
        where Freq is Freqxnum_deltas, so usually freq*3. Essentially multiple
        channels are collapsed to one"""

        train_feats = batch_x
        new_train = []
        for utterance in train_feats:
            swapped = np.swapaxes(utterance,0,1)
            concatenated = np.concatenate(swapped,axis=1)
            reswapped = np.swapaxes(concatenated,0,1)
            new_train.append(reswapped)
        train_feats = np.array(new_train)
        return train_feats


    def load_batch_x(path_batch):
        """ Loads a batch given a list of filenames to numpy arrays in that batch."""

        utterances = [np.load(path) for path in path_batch]
        # The maximum length of an utterance in the batch
        max_len = max([utterance.shape[0] for utterance in utterances])
        shape = (batch_size, max_len) + tuple(utterances[0].shape[1:])
        batch = np.zeros(shape)
        print(batch.shape)
        for i, utt in enumerate(utterances):
            batch[i] = utils.zero_pad(utt, max_len)
        print(batch.shape)
        if flatten:
            batch = collapse(batch)
        print(batch.shape)
        return batch

    train_paths = []
    dialect_classes = set()

    for root, dirnames, filenames in os.walk(path):
        for fn in filenames:
            prefix = fn.split(".")[0] # Get the file prefix
            if fn.endswith(".log_mel_filterbank.npy"):
                # Add to filename list.
                path = os.path.join(root,fn)
                train_paths.append(os.path.join(root,fn))
                dialect_classes.add(path.split("/")[-3])

    dialect_classes = sorted(list(dialect_classes))
    dr_class_map = dict(zip(dialect_classes, range(0,len(dialect_classes))))

    # Adjust the effective size of the TIMIT corpus so I can debug models more easily.
    train_paths = train_paths[:total_size]

    path_batches = [train_paths[i:i+batch_size] for i in range(
            0,len(train_paths),batch_size)]

    if rand:
        random.shuffle(path_batches)

    # A map from phones to ints.
    phone_map = {phone:index for index, phone in enumerate(phone_classes())}

    for path_batch in path_batches:
        batch_x = load_batch_x(path_batch)
        if labels == "dialects":
            y_labels = np.array([dr_class_map[path.split("/")[-3]] for path in path_batch])
            # Make into one-hot vectors
            batch_y = np.concatenate([create_one_hot(label,len(dialect_classes)) for label in y_labels])
        elif labels == "phonemes":
            phn_paths = [path.split(".")[0]+".phn" for path in path_batch]
            batch_y = []
            for phn_path in phn_paths:
                with open(phn_path) as phn_f:
                    phone_indices = [phone_map[phn] for phn in phn_f.readline().split()]
                    batch_y.append(phone_indices)

        yield batch_x, batch_y


