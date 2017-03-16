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
    return sorted(prefixes)

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
    """ Interfaces to the preprocessed corpora to read in train, valid, and
    test set features and transcriptions. This interface is common to all
    corpora. It is the responsibility of <corpora-name>.py to preprocess the
    data into a valid structure of
    <corpus-name>/[mam-train|mam-valid<seed>|mam-test].  """

    _train_prefixes = None

    def __init__(self, num_train, batch_size=None, max_samples=None, rand_seed=0):
        """ corpus_dir: The directory where the preprocessed corpus is found.
            num_train: The number of training instances from the corpus used.
            batch_size: The size of the batches to yield. If None, then it is
                        num_train / 32.0.
            max_samples: The maximum length of utterances measured in samples.
                         Longer utterances are filtered out.
            rand_seed: The seed for the random number generator. If None, then
                       no randomization is used.
        """

        if max_samples:
            raise Exception("Not yet implemented.")

        if num_train % batch_size != 0:
            raise Exception("""Number of training examples %d not divisible
                               by batch size %d.""" % (num_train, batch_size))
        self.num_train = num_train

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

    def get_train_prefixes(self, corpus_dir):
        """ A getter for train_prefixes. Ensures that the training examples
        used is consistent between instantiations of train_batch_gen by
        initializing on the very first call and then not changing
        thereafter.
        """
        if not self._train_prefixes:
            # Get the training prefixes, randomize their order, and take a subset.
            train_prefixes = get_prefixes(
                os.path.join(corpus_dir, "mam-train"))
            if self.rand:
                random.shuffle(train_prefixes)
            self._train_prefixes = train_prefixes[:self.num_train]

        return self._train_prefixes

    def train_batch_gen(self, corpus_dir):
        """ Returns a generator that outputs batches in the training data."""

        # Randomly select some prefixes from all those available.
        if self.rand:
            random.shuffle(self.get_train_prefixes(corpus_dir))
        prefixes = self.get_train_prefixes(corpus_dir)

        # Create batches of batch_size and shuffle them.
        prefix_batches = [prefixes[i:i+self.batch_size]
                          for i in range(0, len(prefixes), self.batch_size)]
        if self.rand:
            random.shuffle(prefix_batches)

        for prefix_batch in prefix_batches:
            yield load_batch(prefix_batch)

    @staticmethod
    def valid_batch(corpus_dir):
        """ Returns a single batch with all the validation cases."""

        # Always get the validation and test sets deterministically.
        valid_prefixes = get_prefixes(
            os.path.join(corpus_dir, "mam-valid"))
        return load_batch(valid_prefixes)

    @staticmethod
    def test_batch(corpus_dir):
        """ Returns a single batch with all the test cases."""

        test_prefixes = get_prefixes(
            os.path.join(corpus_dir, "mam-test"))

        return load_batch(test_prefixes)
