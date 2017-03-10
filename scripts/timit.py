""" Serves as an interface to the TIMIT data. """

import os
import numpy as np
import random

import utils

import config

from nltk.metrics import distance

random.seed(0)

# Hardcoded numbers
num_phones = 61
# The number of training sentences with SA utterances removed.
total_size=3696

def phone_classes(path=os.path.join(config.TGT_DIR, "train"),
        feat_type="mfcc13_d"):
    """ Returns a sorted list of phone classes observed in the TIMIT corpus."""

    train_paths = []
    for root, dirnames, filenames in os.walk(path):
        for fn in filenames:
            prefix = fn.split(".")[0] # Get the file prefix
            if fn.endswith(feat_type + ".npy"):
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
index2phone_map = {index: phone for index, phone in enumerate(phone_set)}
phone2index_map = {phone: index for index, phone in enumerate(phone_set)}

def phones2indices(phones):
    """ Converts a list of phones to a list of indices. Increments the index by
    1 to avoid issues to do with dynamic padding in Tensorflow. """
    return [phone2index_map[phone]+1 for phone in phones]

def indices2phones(indices):
    """ Converts integer representations of phones to human-readable characters. """

    return [(index2phone_map[index-1] if index > 0 else "pad") for index in indices]

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

core_speakers = ["dr1/mdab0", "dr1/mwbt0", "dr1/felc0",
                 "dr2/mtas1", "dr2/mwew0", "dr2/fpas0",
                 "dr3/mjmp0", "dr3/mlnt0", "dr3/fpkt0",
                 "dr4/mlll0", "dr4/mtls0", "dr4/fjlm0",
                 "dr5/mbpm0", "dr5/mklt0", "dr5/fnlp0",
                 "dr6/mcmj0", "dr6/mjdh0", "dr6/fmgd0",
                 "dr7/mgrt0", "dr7/mnjm0", "dr7/fdhc0",
                 "dr8/mjln0", "dr8/mpam0", "dr8/fmld0"]

def collapse(batch_x, time_major=False):
    """ Converts timit into an array of format (batch_size, freqxnum_deltas,
    time). Except where Freq is Freqxnum_deltas, so usually freq*3.
    Essentially, multiple channels are collapsed to one. """

    new_batch_x = []
    for utterance in batch_x:
        swapped = np.swapaxes(utterance,0,1)
        concatenated = np.concatenate(swapped,axis=1)
        new_batch_x.append(concatenated)
    new_batch_x = np.array(new_batch_x)
    if time_major:
        new_batch_x = np.transpose(new_batch_x, (1,0,2))
    return new_batch_x


def load_batch_x(path_batch, flatten, time_major=False):
    """ Loads a batch given a list of filenames to numpy arrays in that batch."""

    utterances = [np.load(path) for path in path_batch]
    # The maximum length of an utterance in the batch
    utter_lens = [utterance.shape[0] for utterance in utterances]
    max_len = max(utter_lens)
    batch_size = len(path_batch)
    shape = (batch_size, max_len) + tuple(utterances[0].shape[1:])
    batch = np.zeros(shape)
    for i, utt in enumerate(utterances):
        batch[i] = utils.zero_pad(utt, max_len)
    if flatten:
        batch = collapse(batch, time_major=time_major)
    return batch, np.array(utter_lens)

def load_batch_y(path_batch):
    batch_y = []
    phn_paths = [path.split(".")[0]+".phn" for path in path_batch]
    for phn_path in phn_paths:
        with open(phn_path) as phn_f:
            phone_indices = phones2indices(phn_f.readline().split())
            batch_y.append(phone_indices)
    return batch_y

def test_set(feat_type, path=os.path.join(config.TGT_DIR, "test"),
        feat_type="mfcc13_d", flatten=True):
    """ Retrieves the core test set of 24 speakers. """

    test_paths = []
    for speaker in core_speakers:
        speaker_path = os.path.join(path, speaker)
        fns = os.listdir(speaker_path)
        for fn in fns:
            if fn.endswith(feat_type + ".npy") and not fn.startswith("sa"):
                test_paths.append(os.path.join(speaker_path, fn))
    batch_x, utter_lens = load_batch_x(test_paths, flatten=flatten)
    batch_y = load_batch_y(test_paths)

    return batch_x, utter_lens, batch_y

def valid_set(feat_type, path=os.path.join(config.TGT_DIR, "test"),
        flatten=True, seed=0):
    """ Retrieves the 50 speaker validation set. """

    random.seed(seed)

    chosen_paths = []
    for dialect in ["dr1", "dr2", "dr3", "dr4", "dr5", "dr6", "dr7", "dr8"]:
        dr_path = os.path.join(path, dialect)
        all_test_speakers = [os.path.join(dr_path, speaker) for speaker in os.listdir(dr_path)]
        valid_speakers = [path for path in all_test_speakers if not
            path.split("test/")[-1] in core_speakers]
        male_valid_speakers = [path for path in valid_speakers if path.split("/")[-1].startswith("m")]
        female_valid_speakers = [path for path in valid_speakers if path.split("/")[-1].startswith("f")]

        # Randomly select two male speakers.
        chosen_paths.extend(random.sample(male_valid_speakers, 2))
        # Randomly select one female speakers.
        chosen_paths.extend(random.sample(female_valid_speakers, 1))

    valid_paths = []
    for speaker_path in chosen_paths:
        fns = os.listdir(speaker_path)
        for fn in fns:
            if fn.endswith(feat_type + ".npy") and not fn.startswith("sa"):
                valid_paths.append(os.path.join(speaker_path, fn))
    batch_x, utter_lens = load_batch_x(valid_paths, flatten=flatten)
    batch_y = load_batch_y(valid_paths)
    batch_y = utils.target_list_to_sparse_tensor(batch_y)

    return batch_x, utter_lens, batch_y

def batch_gen(feat_type, path=os.path.join(config.TGT_DIR, "train"), rand=True,
        batch_size=16, labels="phonemes", total_size=3696, flatten=True,
        time_major=False):
    """ Load the already preprocessed TIMIT data.  Flatten=True will make the
    2-dimensional freq x time a 1 dimensional vector of feats."""

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
            if fn.endswith(feat_type + ".npy") and not fn.startswith("sa"):
                # Add to filename list.
                path = os.path.join(root,fn)
                train_paths.append(os.path.join(root,fn))
                dialect_classes.add(path.split("/")[-3])

    dialect_classes = sorted(list(dialect_classes))
    dr_class_map = dict(zip(dialect_classes, range(0,len(dialect_classes))))

    if rand:
        random.shuffle(train_paths)

    # Adjust the effective size of the TIMIT corpus so I can debug models more easily.
    mod = total_size % batch_size
    if mod != 0:
        print("WARNING Total train size %d not divisible by batch_size %d." % (
                total_size, batch_size))
    train_paths = train_paths[:total_size-mod]

    path_batches = [train_paths[i:i+batch_size] for i in range(
            0,len(train_paths),batch_size)]

    if rand:
        random.shuffle(path_batches)

    for path_batch in path_batches:
        batch_x, batch_x_lens = load_batch_x(path_batch, flatten=flatten)
        if labels == "dialects":
            y_labels = np.array([dr_class_map[path.split("/")[-3]] for path in path_batch])
            # Make into one-hot vectors
            batch_y = np.concatenate([create_one_hot(label,len(dialect_classes)) for label in y_labels])
        elif labels == "phonemes":
            phn_paths = [path.split(".")[0]+".phn" for path in path_batch]
            batch_y = []
            for phn_path in phn_paths:
                with open(phn_path) as phn_f:
                    phone_indices = phones2indices(phn_f.readline().split())
                    batch_y.append(phone_indices)
            batch_y = utils.target_list_to_sparse_tensor(batch_y)

        yield batch_x, batch_x_lens, batch_y

def num_feats(feat_type):
    """ Returns the number of feats for a given feat type. """

    bg = batch_gen(feat_type, rand=False)
    batch = next(bg)
    return batch[0].shape[-1]

def phoneme_error_rate(batch_y, decoded):
    """ Calculates the phoneme error rate between decoder output and the gold
    reference by first collapsing the TIMIT labels into the standard 39
    phonemes."""

    # Use an intermediate human-readable form for debugging. Perhaps can be
    # moved into a separate function down the road.
    y = batch_y[1]
    phn_y = collapse_phones(indices2phones(y))
    phn_pred = collapse_phones(indices2phones(decoded[0].values))
    return distance.edit_distance(phn_y, phn_pred)/len(phn_y)

def batch_per(dense_y, dense_decoded):
    total_per = 0
    for i in range(len(dense_decoded)):
        ref = [phn_i for phn_i in dense_y[i] if phn_i != 0]
        hypo = [phn_i for phn_i in dense_decoded[i] if phn_i != 0]
        ref = collapse_phones(indices2phones(ref))
        hypo = collapse_phones(indices2phones(hypo))
        total_per += distance.edit_distance(ref, hypo)/len(ref)
    return total_per/len(dense_decoded)

