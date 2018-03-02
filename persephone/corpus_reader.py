""" An CorpusReader class that interfaces with preprocessed corpora."""

import logging
import logging.config
import pprint
import random

from nltk.metrics import distance
import numpy as np

from . import config
from . import utils
from .exceptions import PersephoneException

logging.config.fileConfig(config.LOGGING_INI_PATH)

class CorpusReader:
    """ Interfaces to the preprocessed corpora to read in train, valid, and
    test set features and transcriptions. This interface is common to all
    corpora. It is the responsibility of <corpora-name>.py to preprocess the
    data into a valid structure of
    <corpus-name>/[mam-train|mam-valid<seed>|mam-test].  """

    rand = True

    def __init__(self, corpus, num_train=None, batch_size=None, max_samples=None, rand_seed=0):
        """ corpus: The Corpus object that interfaces with a given corpus.
            num_train: The number of training instances from the corpus used.
            batch_size: The size of the batches to yield. If None, then it is
                        num_train / 32.0.
            max_samples: The maximum length of utterances measured in samples.
                         Longer utterances are filtered out.
            rand_seed: The seed for the random number generator. If None, then
                       no randomization is used.
        """

        self.corpus = corpus

        if max_samples:
            raise NotImplementedError("Not yet implemented.")

        if not num_train:
            if not batch_size:
                batch_size = 64
            num_train = len(corpus.get_train_fns()[0])
            num_batches = int(num_train / batch_size)
            num_train = num_batches * batch_size
        self.num_train = num_train
        print("Number of training utterances: {}".format(num_train))
        print("Batch size: {}".format(batch_size))
        print("Batches per epoch: {}".format(int(num_train/batch_size)))

        if batch_size:
            self.batch_size = batch_size
            if num_train % batch_size != 0:
                raise PersephoneException("""Number of training examples %d not divisible
                                   by batch size %d.""" % (num_train, batch_size))
        else:
            # Dynamically change batch size based on number of training
            # examples.
            self.batch_size = int(num_train / 32.0)
            if self.batch_size > 64:
                # I was getting OOM errors when training with 4096 sents, as
                # the batch size jumped to 128
                self.batch_size = 64
            # For now we hope that training numbers are powers of two or
            # something... If not, crash before anything else happens.
            assert num_train % self.batch_size == 0

        random.seed(rand_seed)

        # Make a copy of the training prefixes, randomize their order, and take
        # a subset. Doing random slection of a subset of training now ensures
        # the selection of of training sentences is invariant between calls to
        # train_batch_gen()
        self.train_fns = list(zip(*corpus.get_train_fns()))
        if self.rand:
            random.shuffle(self.train_fns)
        self.train_fns = self.train_fns[:self.num_train]

    def load_batch(self, fn_batch):
        """ Loads a batch with the given prefixes. The prefixes is the full path to the
        training example minus the extension.
        """

        # TODO Assumes targets are available, which is how its distinct from
        # utils.load_batch_x(). These functions need to change names to be
        # clearer.

        inverse = list(zip(*fn_batch))
        feat_fn_batch = inverse[0]
        target_fn_batch = inverse[1]

        batch_inputs, batch_inputs_lens = utils.load_batch_x(feat_fn_batch,
                                                             flatten=False)
        batch_targets_list = []
        for targets_path in target_fn_batch:
            with open(targets_path) as targets_f:
                target_indices = self.corpus.labels_to_indices(targets_f.readline().split())
                batch_targets_list.append(target_indices)
        batch_targets = utils.target_list_to_sparse_tensor(batch_targets_list)

        return batch_inputs, batch_inputs_lens, batch_targets

    def make_batches(self, utterance_fns):
        """ Group utterances into batches for decoding.  """

        # Create batches of batch_size and shuffle them.
        fn_batches = [utterance_fns[i:i+self.batch_size]
                          for i in range(0, len(utterance_fns),
                                         self.batch_size)]

        return fn_batches

    def train_batch_gen(self):
        """ Returns a generator that outputs batches in the training data."""

        # Create batches of batch_size and shuffle them.
        fn_batches = self.make_batches(self.train_fns)

        if self.rand:
            random.shuffle(fn_batches)

        for fn_batch in fn_batches:
            logging.debug("Batch of training filenames: " +
                          pprint.pformat(fn_batch))
            yield self.load_batch(fn_batch)

    def valid_batch(self):
        """ Returns a single batch with all the validation cases."""

        valid_fns = list(zip(*self.corpus.get_valid_fns()))
        return self.load_batch(valid_fns)

    def test_batch(self):
        """ Returns a single batch with all the test cases."""

        test_fns = list(zip(*self.corpus.get_test_fns()))
        return self.load_batch(test_fns)

    def untranscribed_batch_gen(self):
        """ A batch generator for all the untranscribed data. """

        feat_fns = self.corpus.get_untranscribed_fns()
        fn_batches = self.make_batches(feat_fns)

        for fn_batch in fn_batches:
            batch_inputs, batch_inputs_lens = utils.load_batch_x(fn_batch,
                                                             flatten=False)
            yield batch_inputs, batch_inputs_lens, fn_batch

    def human_readable_hyp_ref(self, dense_decoded, dense_y):
        """ Returns a human readable version of the hypothesis for manual
        inspection, along with the reference.
        """

        hyps = []
        refs = []
        for i in range(len(dense_decoded)):
            ref = [phn_i for phn_i in dense_y[i] if phn_i != 0]
            hyp = [phn_i for phn_i in dense_decoded[i] if phn_i != 0]
            ref = self.corpus.indices_to_labels(ref)
            hyp = self.corpus.indices_to_labels(hyp)
            refs.append(ref)
            hyps.append(hyp)

        return hyps, refs

    def human_readable(self, dense_repr):
        """ Returns a human readable version of a dense representation of
        either or reference to facilitate simple manual inspection.
        """

        transcripts = []
        for i in range(len(dense_repr)):
            transcript = [phn_i for phn_i in dense_repr[i] if phn_i != 0]
            transcript = self.corpus.indices_to_labels(transcript)
            transcripts.append(transcript)

        return transcripts

    #def __init__(self, corpus, num_train=None, batch_size=None, max_samples=None, rand_seed=0):

    def __repr__(self):
        return ("%s(" % self.__class__.__name__ +
                "num_train=%s,\n" % repr(self.num_train) +
                "\tbatch_size=%s,\n" % repr(self.batch_size) +
                "\tcorpus=\n%s)" % repr(self.corpus))

    def calc_time(self):
        """
        Prints statistics about the the total duration of recordings in the
        corpus.
        """

        def get_number_of_frames(feat_fns):
            """ fns: A list of numpy files which contain a number of feature
            frames. """

            total = 0
            for feat_fn in feat_fns:
                num_frames = len(np.load(feat_fn))
                total += num_frames

            return total

        def numframes_to_minutes(num_frames):
            # TODO Assumes 10ms strides for the frames. This should generalize to
            # different frame stride widths, as should feature preparation.
            minutes = ((num_frames*10)/1000)/60
            return minutes

        total_frames = 0

        train_fns = [train_fn[0] for train_fn in self.train_fns]
        num_train_frames = get_number_of_frames(train_fns)
        total_frames += num_train_frames
        num_valid_frames = get_number_of_frames(self.corpus.get_valid_fns()[0])
        total_frames += num_valid_frames
        num_test_frames = get_number_of_frames(self.corpus.get_test_fns()[0])
        total_frames += num_test_frames

        print("Train duration: %0.3f" % numframes_to_minutes(num_train_frames))
        print("Validation duration: %0.3f" % numframes_to_minutes(num_valid_frames))
        print("Test duration: %0.3f" % numframes_to_minutes(num_test_frames))
        print("Total duration: %0.3f" % numframes_to_minutes(total_frames))
