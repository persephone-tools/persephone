""" Generic model for automatic speech recognition. """

import inspect
import itertools
import logging
import os
from pathlib import Path
import sys
from typing import Union, Sequence, Set, List

import tensorflow as tf

from .preprocess import labels
from . import utils
from . import config
from .exceptions import PersephoneException

OPENFST_PATH = config.OPENFST_BIN_PATH

allow_growth_config = tf.ConfigProto(log_device_placement=False)
allow_growth_config.gpu_options.allow_growth = True #pylint: disable=no-member

logger = logging.getLogger(__name__) # type: ignore

def load_metagraph(model_path_prefix: Union[str, Path]) -> tf.train.Saver:
    """ Given the path to a model on disk (these will typically be found in
    directories such as exp/<exp_num>/model/model_best.*) creates a Saver
    object that can then be used to restore the graph inside a tf.Session.
    """

    model_path_prefix = str(model_path_prefix)
    metagraph = tf.train.import_meta_graph(model_path_prefix + ".meta")
    return metagraph

def dense_to_human_readable(dense_repr, index_to_label):
    """ Converts a dense representation of model decoded output into human
    readable, using a mapping from indices to labels. """

    transcripts = []
    for i in range(len(dense_repr)):
        transcript = [phn_i for phn_i in dense_repr[i] if phn_i != 0]
        transcript = [index_to_label[index] for index in transcript]
        transcripts.append(transcript)

    return transcripts

def decode(model_path_prefix: Union[str, Path],
           input_paths: Sequence[Path],
           label_set: Set[str]) -> List[List[str]]:

    model_path_prefix = str(model_path_prefix)

    # TODO Confirm that that WAVs exist.

    # TODO Confirm that the feature files exist. Create them if they don't.

    # TODO Change the second argument to have some upper bound. If the caller
    # requests 1000 WAVs be transcribed, they shouldn't all go in one batch.
    fn_batches = utils.make_batches(input_paths, len(input_paths))
    # Load the model and perform decoding.
    metagraph = load_metagraph(model_path_prefix)
    with tf.Session() as sess:
        metagraph.restore(sess, model_path_prefix)

        for fn_batch in fn_batches:
            batch_x, batch_x_lens = utils.load_batch_x(fn_batch)

        # TODO These placeholder names should be a backup if names from a newer
        # naming scheme aren't present. Otherwise this won't generalize to
        # different architectures.
        feed_dict = {"Placeholder:0": batch_x,
                     "Placeholder_1:0": batch_x_lens}

        dense_decoded = sess.run("SparseToDense:0", feed_dict=feed_dict)

    # Create a human-readable representation of the decoded.
    indices_to_labels = labels.make_indices_to_labels(label_set)
    human_readable = dense_to_human_readable(dense_decoded, indices_to_labels)

    return human_readable

class Model:
    """ Generic model for our ASR tasks.

    Attributes:
        exp_dir: Path that the experiment directory is located
        corpus_reader: `CorpusReader` object that provides access to the corpus
                       this model is being trained on.
        log_softmax: log softmax function
        batch_x: A batch of input features. ("x" is the typical notation in ML
                 papers on this topic denoting model input)
        batch_x_lens: The lengths of each utterance. This is used by Tensorflow
                      to know how much to pad utterances that are shorter than
                      this length.
        batch_y: Reference labels for a batch ("y" is the typical notation in ML
                 papers on this topic denoting training labels)
        optimizer: The gradient descent method being used. (Typically we use Adam
                   because it has provided good results but any stochastic gradient
                   descent method could be substituted here)
        ler: Label error rate.
        dense_decoded: Dense representation of the model transcription output.
        dense_ref: Dense representation of the reference transcription.
        saved_model_path: Path to where the Tensorflow model is being saved on disk.
    """

    def __init__(self, exp_dir, corpus_reader) -> None:
        self.exp_dir = exp_dir
        self.corpus_reader = corpus_reader
        self.log_softmax = None
        self.batch_x = None
        self.batch_x_lens = None
        self.batch_y = None
        self.optimizer = None
        self.ler = None
        self.dense_decoded = None
        self.dense_ref = None
        self.saved_model_path = None

    def transcribe(self, restore_model_path=None) -> None:
        """ Transcribes an untranscribed dataset. Similar to eval() except
        no reference translation is assumed, thus no LER is calculated.
        """

        saver = tf.train.Saver()
        with tf.Session(config=allow_growth_config) as sess:
            if restore_model_path:
                saver.restore(sess, restore_model_path)
            else:
                if self.saved_model_path:
                    saver.restore(sess, self.saved_model_path)
                else:
                    raise PersephoneException("No model to use for transcription.")

            batch_gen = self.corpus_reader.untranscribed_batch_gen()

            hyp_batches = []
            for batch_i, batch in enumerate(batch_gen):

                batch_x, batch_x_lens, feat_fn_batch = batch

                feed_dict = {self.batch_x: batch_x,
                             self.batch_x_lens: batch_x_lens}

                [dense_decoded] = sess.run([self.dense_decoded], feed_dict=feed_dict)
                hyps = self.corpus_reader.human_readable(dense_decoded)

                # Prepare dir for transcription
                hyps_dir = os.path.join(self.exp_dir, "transcriptions")
                if not os.path.isdir(hyps_dir):
                    os.mkdir(hyps_dir)

                hyp_batches.append((hyps,feat_fn_batch))

            with open(os.path.join(hyps_dir, "hyps.txt"), "w") as hyps_f:
                for hyp_batch, fn_batch in hyp_batches:
                    for hyp, fn in zip(hyp_batch, fn_batch):
                        print(fn, file=hyps_f)
                        print(" ".join(hyp), file=hyps_f)
                        print("", file=hyps_f)

    def eval(self, restore_model_path=None) -> None:
        """ Evaluates the model on a test set."""

        saver = tf.train.Saver()
        with tf.Session(config=allow_growth_config) as sess:
            if restore_model_path:
                logger.info("restoring model from %s", restore_model_path)
                saver.restore(sess, restore_model_path)
            else:
                assert self.saved_model_path
                logger.info("restoring model from %s", self.saved_model_path)
                saver.restore(sess, self.saved_model_path)

            test_x, test_x_lens, test_y = self.corpus_reader.test_batch()

            feed_dict = {self.batch_x: test_x,
                         self.batch_x_lens: test_x_lens,
                         self.batch_y: test_y}

            test_ler, dense_decoded, dense_ref = sess.run(
                [self.ler, self.dense_decoded, self.dense_ref],
                feed_dict=feed_dict)
            hyps, refs = self.corpus_reader.human_readable_hyp_ref(
                dense_decoded, dense_ref)
            # Log hypotheses
            hyps_dir = os.path.join(self.exp_dir, "test")
            if not os.path.isdir(hyps_dir):
                os.mkdir(hyps_dir)
            with open(os.path.join(hyps_dir, "hyps"), "w") as hyps_f:
                for hyp in hyps:
                    print(" ".join(hyp), file=hyps_f)
            with open(os.path.join(hyps_dir, "refs"), "w") as refs_f:
                for ref in refs:
                    print(" ".join(ref), file=refs_f)

            test_per = utils.batch_per(hyps, refs)
            with open(os.path.join(hyps_dir, "test_per"), "w") as per_f:
                print("Test PER: %f, tf LER: %f" % (test_per, test_ler), file=per_f)

    def output_best_scores(self, best_epoch_str):
        """Output best scores to the filesystem"""
        BEST_SCORES_FILENAME = "best_scores.txt"
        with open(os.path.join(self.exp_dir, BEST_SCORES_FILENAME), "w") as best_f:
            print(best_epoch_str, file=best_f, flush=True)

    def train(self, early_stopping_steps: int = 10, min_epochs: int = 30,
              max_valid_ler: float = 1.0, max_train_ler: float = 0.3,
              max_epochs: int = 100, restore_model_path=None) -> None:
        """ Train the model.

            min_epochs: minimum number of epochs to run training for.
            max_epochs: maximum number of epochs to run training for.
            early_stopping_steps: Stop training after this number of steps
                                  if no LER improvement has been made.
            max_valid_ler: Maximum LER for the validation set.
                           Training will continue until this is met or another
                           stopping condition occurs.
            max_train_ler: Maximum LER for the training set.
                           Training will continue until this is met or another
                           stopping condition occurs.
            restore_model_path: The path to restore a model from.
        """
        logger.info("Training model")
        best_valid_ler = 2.0
        steps_since_last_record = 0

        #Get information about training for the names of output files.
        frame = inspect.currentframe()
        # pylint: disable=deprecated-method
        # It was a mistake to deprecate this in Python 3.5
        if frame:
            args, _, _, values = inspect.getargvalues(frame)
            with open(os.path.join(self.exp_dir, "train_description.txt"), "w") as desc_f:
                for arg in args:
                    if type(values[arg]) in [str, int, float] or isinstance(
                            values[arg], type(None)):
                        print("%s=%s" % (arg, values[arg]), file=desc_f)
                    else:
                        print("%s=%s" % (arg, values[arg].__dict__), file=desc_f)
                print("num_train=%s" % (self.corpus_reader.num_train), file=desc_f)
                print("batch_size=%s" % (self.corpus_reader.batch_size), file=desc_f)
        else:
            logger.error("Couldn't find frame information, failed to write train_description.txt")


        # Load the validation set
        valid_x, valid_x_lens, valid_y = self.corpus_reader.valid_batch()

        saver = tf.train.Saver()

        with tf.Session(config=allow_growth_config) as sess:

            if restore_model_path:
                logger.info("Restoring model from path %s", restore_model_path)
                saver.restore(sess, restore_model_path)
            else:
                sess.run(tf.global_variables_initializer())

            # Prepare directory to output hypotheses to
            hyps_dir = os.path.join(self.exp_dir, "decoded")
            if not os.path.isdir(hyps_dir):
                os.mkdir(hyps_dir)

            best_epoch_str = None

            training_log_path = os.path.join(self.exp_dir, "train_log.txt")
            if os.path.exists(training_log_path):
                logger.error("Error, overwriting existing log file at path {}".format(training_log_path))
            with open(training_log_path, "w") as out_file:
                for epoch in itertools.count():
                    print("\nexp_dir %s, epoch %d" % (self.exp_dir, epoch))
                    batch_gen = self.corpus_reader.train_batch_gen()

                    train_ler_total = 0
                    batch_i = None
                    print("\tBatch...", end="")
                    for batch_i, batch in enumerate(batch_gen):
                        print("%d..." % batch_i, end="")
                        sys.stdout.flush()
                        batch_x, batch_x_lens, batch_y = batch

                        feed_dict = {self.batch_x: batch_x,
                                    self.batch_x_lens: batch_x_lens,
                                    self.batch_y: batch_y}

                        _, ler, = sess.run([self.optimizer, self.ler],
                                        feed_dict=feed_dict)

                        train_ler_total += ler

                    feed_dict = {self.batch_x: valid_x,
                                self.batch_x_lens: valid_x_lens,
                                self.batch_y: valid_y}

                    try:
                        valid_ler, dense_decoded, dense_ref = sess.run(
                            [self.ler, self.dense_decoded, self.dense_ref],
                            feed_dict=feed_dict)
                    except tf.errors.ResourceExhaustedError:
                        import pprint
                        print("Ran out of memory allocating a batch:")
                        pprint.pprint(feed_dict)
                        logger.critical("Ran out of memory allocating a batch: %s", pprint.pformat(feed_dict))
                        raise
                    hyps, refs = self.corpus_reader.human_readable_hyp_ref(
                        dense_decoded, dense_ref)
                    # Log hypotheses
                    with open(os.path.join(hyps_dir, "epoch%d_hyps" % epoch), "w") as hyps_f:
                        for hyp in hyps:
                            print(" ".join(hyp), file=hyps_f)
                    if epoch == 0:
                        with open(os.path.join(hyps_dir, "refs"), "w") as refs_f:
                            for ref in refs:
                                print(" ".join(ref), file=refs_f)

                    valid_per = utils.batch_per(hyps, refs)

                    epoch_str = "Epoch %d. Training LER: %f, validation LER: %f, validation PER: %f" % (
                        epoch, (train_ler_total / (batch_i + 1)), valid_ler, valid_per)
                    print(epoch_str, flush=True, file=out_file)
                    if best_epoch_str is None:
                        best_epoch_str = epoch_str

                    # Implement early stopping.
                    if valid_ler < best_valid_ler:
                        print("New best valid_ler", file=out_file)
                        best_valid_ler = valid_ler
                        best_epoch_str = epoch_str
                        steps_since_last_record = 0

                        # Save the model.
                        checkpoint_path = os.path.join(self.exp_dir, "model", "model_best.ckpt")
                        if not os.path.exists(os.path.dirname(checkpoint_path)):
                            os.mkdir(os.path.dirname(checkpoint_path))
                        saver.save(sess, checkpoint_path)
                        self.saved_model_path = checkpoint_path

                        # Output best hyps
                        with open(os.path.join(hyps_dir, "best_hyps"), "w") as hyps_f:
                            for hyp in hyps:
                                print(" ".join(hyp), file=hyps_f)

                    else:
                        print("Steps since last best valid_ler: %d" % (steps_since_last_record), file=out_file)
                        steps_since_last_record += 1
                        if epoch >= max_epochs:
                            self.output_best_scores(best_epoch_str)
                            break
                        if steps_since_last_record >= early_stopping_steps:
                            if epoch >= min_epochs:
                                # Then we've done the minimum number of epochs.
                                if valid_ler <= max_valid_ler and ler <= max_train_ler:
                                    # Then training error has moved sufficiently
                                    # towards convergence.
                                    print("Stopping since best validation score hasn't been"
                                        " beaten in %d epochs and at least %d have been"
                                        " done. The valid ler (%d) is below %d and"
                                        " the train ler (%d) is below %d." %
                                        (early_stopping_steps, min_epochs, valid_ler,
                                        max_valid_ler, ler, max_train_ler),
                                        file=out_file, flush=True)
                                    self.output_best_scores(best_epoch_str)
                                    break
                                else:
                                    # Keep training because we haven't achieved
                                    # convergence.
                                    continue
                            else:
                                # Keep training because we haven't done the minimum
                                # numper of epochs.
                                continue

                # Finally, run evaluation on the test set.
                self.eval(restore_model_path=self.saved_model_path)
