""" Miscellaneous utility functions. """

import os

import numpy as np

from nltk.metrics import distance

def target_list_to_sparse_tensor(target_list):
    """ Make tensorflow SparseTensor from list of targets, with each element in
    the list being a list or array with the values of the target sequence
    (e.g., the integer values of a character map for an ASR target string) See
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/
    contrib/ctc/ctc_loss_op_test.py for example of SparseTensor format
    """
    indices = []
    vals = []
    for t_i, target in enumerate(target_list):
        for seq_i, val in enumerate(target):
            indices.append([t_i, seq_i])
            vals.append(val)
    shape = [len(target_list), np.asarray(indices).max(0)[1]+1]
    return (np.array(indices), np.array(vals), np.array(shape))

def zero_pad(matrix, to_length):
    """ Zero pads along the 0th dimension to make sure the utterance array
    x is of length to_length."""

    assert matrix.shape[0] <= to_length
    result = np.zeros((to_length,) + matrix.shape[1:])
    result[:matrix.shape[0]] = matrix
    return result

def collapse(batch_x, time_major=False):
    """ Converts timit into an array of format (batch_size, freq x num_deltas,
    time). Essentially, multiple channels are collapsed to one. """

    new_batch_x = []
    for utterance in batch_x:
        swapped = np.swapaxes(utterance, 0, 1)
        concatenated = np.concatenate(swapped, axis=1)
        new_batch_x.append(concatenated)
    new_batch_x = np.array(new_batch_x)
    if time_major:
        new_batch_x = np.transpose(new_batch_x, (1, 0, 2))
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
        batch[i] = zero_pad(utt, max_len)
    if flatten:
        batch = collapse(batch, time_major=time_major)
    return batch, np.array(utter_lens)

def batch_per(dense_y, dense_decoded):
    """ Calculates the phoneme error rate of a batch."""

    total_per = 0
    for i in range(len(dense_decoded)):
        ref = [phn_i for phn_i in dense_y[i] if phn_i != 0]
        hypo = [phn_i for phn_i in dense_decoded[i] if phn_i != 0]
        total_per += distance.edit_distance(ref, hypo)/len(ref)
    return total_per/len(dense_decoded)

def get_prefixes(dirname):
    """ Returns a list of prefixes to files in the directory (which might be a whole
    corpus, or a train/valid/test subset. The prefixes include the path leading
    up to it, but only the filename up until the first observed period '.'
    """

    prefixes = []
    for root, _, filenames in os.walk(dirname):
        for filename in filenames:
            if filename.endswith(".npy"):
                # Then it's an input feature file and its prefix will
                # correspond to a training example
                prefixes.append(os.path.join(root, filename.split(".")[0]))
    return sorted(prefixes)
