""" An CorpusReader class that interfaces with preprocessed corpora."""

import os
import random

import utils

def get_prefixes(set_dir):
    """ Returns a list of prefixes to files in the set (which might be a whole
    corpus, or a train/valid/test subset. The prefixes include the path leading
    up to it, but remove only the file extension.
    """

    prefixes = []
    for root, _, filenames in os.walk(set_dir):
        for filename in filenames:
            if filename.endswith(".npy"):
                # Then it's an input feature file and its prefix will
                # correspond to a training example
                prefixes.append(os.path.join(root, filename))
    return prefixes

def load_batch(prefix_batch):
    """ Loads a batch with the given prefix. The prefix is the full path to the
    training example minus the extension.
    """

    batch_inputs_paths = ["%s.npy" % prefix for prefix in prefix_batch]
    batch_inputs, batch_inputs_lens = utils.load_batch_x(batch_inputs_paths,
                                                         flatten=True)

    batch_targets_paths = ["%s.tgt" % prefix for prefix in prefix_batch]
    batch_targets_list = []
    for targets_path in batch_targets_paths:
        with open(targets_path) as targets_f:
            batch_targets_list.append(targets_f.readline().split())
    batch_targets = utils.target_list_to_sparse_tensor(batch_targets_list)

    return batch_inputs, batch_inputs_lens, batch_targets

class CorpusReader:
    """ Interfaces to the corpora to read in train, valid, and test set
    features and transcriptions. This interface is common to all corpora. It is
    the responsibility of <corpora-name>.py to preprocess the data into a valid
    structure of <corpus-name>/[mam-train|mam-valid<seed>|mam-test].
    """

    def __init__(self, corpus_dir, num_train, batch_size=None, rand_seed=0):
        """ corpus_dir: The directory where the preprocessed corpus is found.
            num_train: The number of training instances from the corpus used.
            batch_size: The size of the batches to yield. If None, then it is
                        num_train / 32.0.
            rand_seed: The seed for the random number generator. If None, then
                       no randomization is used.
        """

        if num_train % batch_size != 0:
            raise Exception("""Number of training examples %d not divisible
                               by batch size %d.""" % (num_train, batch_size))

        if batch_size:
            self.batch_size = batch_size
        else:
            # Dynamically change batch size based on number of training
            # examples.
            self.batch_size = num_train / 32.0

        if rand_seed:
            random.seed(rand_seed)
            self.rand = True
        else:
            self.rand = False

        # Get the training prefixes, randomize their order, and take a subset.
        train_prefixes = get_prefixes(
            os.path.join(corpus_dir, "mam-train"))
        if self.rand:
            random.shuffle(train_prefixes)
        self.train_prefixes = train_prefixes[:num_train]

        # Always get the validation and test sets deterministically.
        self.valid_prefixes = get_prefixes(
            os.path.join(corpus_dir, "mam-valid%d" % rand_seed))
        self.test_prefixes = get_prefixes(
            os.path.join(corpus_dir, "mam-test"))

    def train_batch_gen(self):
        """ Returns a generator that outputs batches in the training data."""

        # Randomly select some prefixes from all those available.
        if self.rand:
            random.shuffle(self.train_prefixes)
        prefixes = self.train_prefixes

        # Create batches of batch_size and shuffle them.
        prefix_batches = [prefixes[i:i+self.batch_size]
                          for i in range(0, len(prefixes), self.batch_size)]
        if self.rand:
            random.shuffle(prefix_batches)

        for prefix_batch in prefix_batches:
            yield load_batch(prefix_batch)

    def valid_batch(self):
        """ Returns a single batch with all the validation cases."""

        return load_batch(self.valid_prefixes)

    def test_batch(self):
        """ Returns a single batch with all the test cases."""

        return load_batch(self.test_prefixes)
