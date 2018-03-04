""" Generic model for automatic speech recognition. """

import inspect
import itertools
import logging
import logging.config
import os
import subprocess
import sys

import numpy as np
import tensorflow as tf

from . import utils
from . import lattice
from . import config
from .exceptions import PersephoneException

OPENFST_PATH = config.OPENFST_BIN_PATH

allow_growth_config = tf.ConfigProto(log_device_placement=False)
allow_growth_config.gpu_options.allow_growth=True #pylint: disable=no-member

logging.config.fileConfig(config.LOGGING_INI_PATH)

class Model:
    """ Generic model for our ASR tasks. """

    # Subclasses should instantiate these variables:
    exp_dir = None
    batch_x = None
    batch_x_lens = None
    batch_y = None
    optimizer = None
    ler = None
    dense_decoded = None
    dense_ref = None
    corpus_reader = None
    saved_model_path = None

    def __init__(self, exp_dir, corpus_reader):
        self.exp_dir = exp_dir
        self.corpus_reader = corpus_reader
        self.log_softmax = None

    def transcribe(self, restore_model_path=None):
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

            """
            # TODO This sorting is Na-corpus centric and won't generalize. It
            # is to sort by recording name (ie. Benevolence) then by utterance
            # id within that (Benevolence.0, benevolence.1, ...)
            utters = [(hyps, feat_fn, feat_fn.split(".")[0], int(feat_fn.split(".")[1]))
                      for hyps, feat_fn in zip(hyps, feat_fn_batch)]
            print(utters)
            utters.sort(key=itemgetter(2,3))
            print(utters)
            with open(os.path.join(hyps_dir, "hyps"), "w") as hyps_f:
                for hyp, fn, _, _ in utters:
                    fn = "_".join(os.path.basename(fn).split(".")[:2])
                    print(fn + ": ", file=hyps_f)
                    print(" ".join(hyp), file=hyps_f)
            """

    def output_lattices(self, batch, restore_model_path=None):
        """ Outputs the logits from the model, given an input batch, so that
            lattices can ultimately be extracted."""

        saver = tf.train.Saver()
        with tf.Session(config=allow_growth_config) as sess:
            if restore_model_path:
                saver.restore(sess, restore_model_path)
            else:
                assert self.saved_model_path
                saver.restore(sess, self.saved_model_path)

            batch_x, batch_x_lens, batch_y = batch

            feed_dict = {self.batch_x: batch_x,
                         self.batch_x_lens: batch_x_lens,
                         self.batch_y: batch_y}

            # Get the log_softmax matrices
            log_softmax = sess.run([self.log_softmax], feed_dict=feed_dict)
            log_softmax = log_softmax[0]
            log_softmax = np.swapaxes(log_softmax, 0, 1)
            out_dir = os.path.join(self.exp_dir, "lattice")
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            for i, example in enumerate(log_softmax):
                length = batch_x_lens[i]
                np.save(os.path.join(out_dir, "utterance_%d_log_softmax" % i),
                        example[:length])

        ### Create the lattices.###
        index_to_token = self.corpus_reader.corpus.INDEX_TO_PHONEME

        # Create symbol table.
        syms_fn = os.path.join(out_dir, "symbols.txt")
        lattice.create_symbol_table(index_to_token, syms_fn)

        # Create the FST that removes blanks and repeated tokens.
        lattice.create_collapse_fst(index_to_token,
                                    os.path.join(out_dir, "collapse_fst.txt"))
        lattice.compile_fst(os.path.join(out_dir, "collapse_fst"), syms_fn)

        for i, log_softmax_example in enumerate(log_softmax):
            # Create a confusion network
            length = batch_x_lens[i]
            lattice.logsoftmax2confusion(log_softmax_example[:length],
                                         index_to_token,
                                         os.path.join(out_dir, "utterance_%d" % i),
                                         beam_size=4)
            lattice.compile_fst(os.path.join(out_dir, "utterance_%d.confusion" % i),
                                syms_fn)

            prefix = os.path.join(out_dir, "utterance_%d" % i)

            try:
                run_args = [os.path.join(OPENFST_PATH, "fstarcsort"),
                            prefix + ".confusion.bin",
                            prefix + ".confusion.sort.bin"]
                subprocess.run(run_args)

                # Compose the confusion network with the FST that removes blanks
                # and repetitions, expanding to a larger lattice-like FST.
                run_args = [os.path.join(OPENFST_PATH, "fstcompose"),
                            prefix + ".confusion.sort.bin",
                            os.path.join(out_dir, "collapse_fst.bin"),
                            prefix + ".collapsed.bin"]
                subprocess.run(run_args)

                # Take the output projection of the FST.
                run_args = [os.path.join(OPENFST_PATH, "fstproject"),
                            "--project_output",
                            prefix + ".collapsed.bin", prefix + ".projection.bin"]
                subprocess.run(run_args)

                # Push weights
#               run_args = [os.path.join(OPENFST_PATH, "fstpush"),
#                           "--push_weights",
#                           prefix + ".projection.bin", prefix + ".pushed.bin"]
#               subprocess.run(run_args)

                # Remove epsilons
                run_args = [os.path.join(OPENFST_PATH, "fstrmepsilon"),
                            "--reverse=true",
                            prefix + ".projection.bin", prefix + ".rmepsilon.bin"]
                subprocess.run(run_args)
            except FileNotFoundError:
                print("Make sure you have OpenFST binaries installed and "
                      "available on the path")
                raise

    def eval(self, restore_model_path=None):
        """ Evaluates the model on a test set."""

        saver = tf.train.Saver()
        with tf.Session(config=allow_growth_config) as sess:
            if restore_model_path:
                saver.restore(sess, restore_model_path)
            else:
                assert self.saved_model_path
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

    def train(self, early_stopping_steps=10, min_epochs=30, max_valid_ler=1.0,
              max_train_ler=0.3, max_epochs=100, restore_model_path=None):
        """ Train the model.

            batch_size: The number of utterances in each batch.
            total_size: The number of TIMIT training examples to use.
            num_epochs: The number of times to iterate over all the training
                        examples.
            feat_type:  Is the identifier for the type of features we're using.
                        'mfcc13_d' means MFCCs of 13 dimensions with their first
                        derivatives.
            save_n: Whether to save the model at every n epochs.
            restore_model_path: The path to restore a model from.
        """

        best_valid_ler = 2.0
        steps_since_last_record = 0

        #Get information about training for the names of output files.
        frame = inspect.currentframe()
        # pylint: disable=deprecated-method
        # It was a mistake to deprecate this in Python 3.5
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

        out_file = open(os.path.join(self.exp_dir, "train_log.txt"), "w")

        # Load the validation set
        valid_x, valid_x_lens, valid_y = self.corpus_reader.valid_batch()

        saver = tf.train.Saver()

        sess = tf.Session(config=allow_growth_config)

        if restore_model_path:
            saver.restore(sess, restore_model_path)
        else:
            sess.run(tf.global_variables_initializer())

        # Prepare directory to output hypotheses to
        hyps_dir = os.path.join(self.exp_dir, "decoded")
        if not os.path.isdir(hyps_dir):
            os.mkdir(hyps_dir)

        best_epoch_str = None
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
                print("Ran out of memory allocating a batch:")
                import pprint
                pprint.pprint(feed_dict)
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
            if best_epoch_str == None:
                best_epoch_str = epoch_str

            # Implement early stopping.
            if valid_ler < best_valid_ler:
                print("New best valid_ler", file=out_file)
                best_valid_ler = valid_ler
                best_epoch_str = epoch_str
                steps_since_last_record = 0

                # Save the model.
                path = os.path.join(self.exp_dir, "model", "model_best.ckpt")
                if not os.path.exists(os.path.dirname(path)):
                    os.mkdir(os.path.dirname(path))
                saver.save(sess, path)
                self.saved_model_path = path

                # Output best hyps
                with open(os.path.join(hyps_dir, "best_hyps"), "w") as hyps_f:
                    for hyp in hyps:
                        print(" ".join(hyp), file=hyps_f)

            else:
                print("Steps since last best valid_ler: %d" % (steps_since_last_record), file=out_file)
                steps_since_last_record += 1
                if epoch >= max_epochs:
                    with open(os.path.join(self.exp_dir, "best_scores.txt"), "w") as best_f:
                        print(best_epoch_str, file=best_f, flush=True)
                        sess.close()
                        out_file.close()
                        break
                if steps_since_last_record >= early_stopping_steps:
                    if epoch >= min_epochs:
                        # Then we've done the minimum number of epochs.
                        if valid_ler <= max_valid_ler and ler <= max_train_ler:
                            # Then training error has moved sufficiently
                            # towards convergence.
                            print("""Stopping since best validation score hasn't been
                                  beaten in %d epochs and at least %d have been
                                  done. The valid ler (%d) is below %d and 
                                  the train ler (%d) is below %d.""" %
                                  (early_stopping_steps, min_epochs, valid_ler,
                                  max_valid_ler, ler, max_train_ler),
                                  file=out_file, flush=True)
                            with open(os.path.join(self.exp_dir, "best_scores.txt"), "w") as best_f:
                                print(best_epoch_str, file=best_f, flush=True)
                                sess.close()
                                out_file.close()
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

        sess.close()
        out_file.close()
