""" Generic model for automatic speech recognition. """

import inspect
import itertools
import logging
import math
import os
from pathlib import Path
import sys
from typing import Callable, Optional, Union, Sequence, Set, List, Dict

import tensorflow as tf

from .preprocess import labels, feat_extract
from . import utils
from . import config
from .config import ENCODING
from .corpus import Corpus
from .exceptions import PersephoneException
from .corpus_reader import CorpusReader

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

def dense_to_human_readable(dense_repr: Sequence[Sequence[int]], index_to_label: Dict[int, str]) -> List[List[str]]:
    """ Converts a dense representation of model decoded output into human
    readable, using a mapping from indices to labels. """

    transcripts = []
    for dense_r in dense_repr:
        non_empty_phonemes = [phn_i for phn_i in dense_r if phn_i != 0]
        transcript = [index_to_label[index] for index in non_empty_phonemes]
        transcripts.append(transcript)

    return transcripts

def decode_corpus(model_path_prefix: Union[str, Path],
                  corpus: Corpus,
                  *,
                  batch_size: int = 64,
                  feat_dir: Optional[Path]=None,
                  batch_x_name: str="batch_x:0",
                  batch_x_lens_name: str="batch_x_lens:0",
                  output_name: str="hyp_dense_decoded:0") -> List[List[str]]:
    input_paths = [Path(corpus.tgt_dir) / "wav" / Path(prefix + ".wav")
                   for prefix in corpus.untranscribed_prefixes]
    return decode(model_path_prefix,
           input_paths,
           label_set=corpus.labels,
           feature_type=corpus.feat_type,
           batch_size=batch_size,
           feat_dir=feat_dir,
           batch_x_name=batch_x_name,
           batch_x_lens_name=batch_x_lens_name,
           output_name=output_name)

def decode(model_path_prefix: Union[str, Path],
           input_paths: Sequence[Path],
           label_set: Set[str],
           *,
           feature_type: str = "fbank", #TODO Make this None and infer feature_type from dimension of NN input layer.
           batch_size: int = 64,
           feat_dir: Optional[Path]=None,
           batch_x_name: str="batch_x:0",
           batch_x_lens_name: str="batch_x_lens:0",
           output_name: str="hyp_dense_decoded:0") -> List[List[str]]:
    """Use an existing tensorflow model that exists on disk to decode
    WAV files.

    Args:
        model_path_prefix: The path to the saved tensorflow model.
                           This is the full prefix to the ".ckpt" file.
        input_paths: A sequence of `pathlib.Path`s to WAV files to put through
                     the model provided.
        label_set: The set of all the labels this model uses.
        feature_type: The type of features this model uses.
                      Note that this MUST match the type of features that the
                      model was trained on initially.
        feat_dir: Any files that require preprocessing will be
                                  saved to the path specified by this.
        batch_x_name: The name of the tensorflow input for batch_x
        batch_x_lens_name: The name of the tensorflow input for batch_x_lens
        output_name: The name of the tensorflow output
    """

    if not input_paths:
        raise PersephoneException("No untranscribed WAVs to transcribe.")

    model_path_prefix = str(model_path_prefix)

    for p in input_paths:
        if not p.exists():
            raise PersephoneException(
                "The WAV file path {} does not exist".format(p)
            )

    preprocessed_file_paths = []
    for p in input_paths:
        prefix = p.stem
        # Check the "feat" directory as per the filesystem conventions of a Corpus
        feature_file_ext = ".{}.npy".format(feature_type)
        conventional_npy_location =  p.parent.parent / "feat" / (Path(prefix + feature_file_ext))
        if conventional_npy_location.exists():
            # don't need to preprocess it
            preprocessed_file_paths.append(conventional_npy_location)
        else:
            if not feat_dir:
                feat_dir = p.parent.parent / "feat"
            if not feat_dir.is_dir():
                os.makedirs(str(feat_dir))

            mono16k_wav_path = feat_dir / "{}.wav".format(prefix)
            feat_path = feat_dir / "{}.{}.npy".format(prefix, feature_type)
            feat_extract.convert_wav(p, mono16k_wav_path)
            preprocessed_file_paths.append(feat_path)
    # preprocess the file that weren't found in the features directory
    # as per the filesystem conventions
    if feat_dir:
        feat_extract.from_dir(feat_dir, feature_type)

    fn_batches = utils.make_batches(preprocessed_file_paths, batch_size)
    # Load the model and perform decoding.
    metagraph = load_metagraph(model_path_prefix)
    with tf.Session() as sess:
        metagraph.restore(sess, model_path_prefix)

        for fn_batch in fn_batches:
            batch_x, batch_x_lens = utils.load_batch_x(fn_batch)

        # TODO These placeholder names should be a backup if names from a newer
        # naming scheme aren't present. Otherwise this won't generalize to
        # different architectures.
        feed_dict = {batch_x_name: batch_x,
                     batch_x_lens_name: batch_x_lens}

        dense_decoded = sess.run(output_name, feed_dict=feed_dict)

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

    def __init__(self, exp_dir: Union[Path, str], corpus_reader: CorpusReader) -> None:
        self.exp_dir = str(exp_dir) if isinstance(exp_dir, Path) else exp_dir # type: str
        self.corpus_reader = corpus_reader
        self.log_softmax = None
        self.batch_x = None
        self.batch_x_lens = None
        self.batch_y = None
        self.optimizer = None
        self.ler = None
        self.dense_decoded = None
        self.dense_ref = None
        self.saved_model_path = "" # type: str

    def transcribe(self, restore_model_path: Optional[str]=None) -> None:
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

            with open(os.path.join(hyps_dir, "hyps.txt"), "w",
                      encoding=ENCODING) as hyps_f:
                for hyp_batch, fn_batch in hyp_batches:
                    for hyp, fn in zip(hyp_batch, fn_batch):
                        print(fn, file=hyps_f)
                        print(" ".join(hyp), file=hyps_f)
                        print("", file=hyps_f)

    def decode(self):
        model_path_prefix = Path(self.exp_dir) / "model" / "model_best.ckpt"
        prefixes = self.corpus_reader.corpus.untranscribed_prefixes
        input_paths = [self.corpus_reader.corpus.tgt_dir / "feat" / Path(p + ".wav")
                       for p in prefixes]
        label_set = self.corpus_reader.corpus.labels
        feature_type = self.corpus_reader.corpus.feat_type
        batch_size = self.corpus_reader.batch_size
        batch_x_name = self.batch_x.name
        batch_x_lens_name = self.batch_x_lens.name
        output_name = self.dense_decoded.name
        return decode(model_path_prefix,
               input_paths,
               label_set,
               feature_type=feature_type,
               batch_size=batch_size,
               batch_x_name=batch_x_name,
               batch_x_lens_name=batch_x_lens_name,
               output_name=output_name)
 
    def eval(self, restore_model_path: Optional[str]=None) -> None:
        """ Evaluates the model on a test set."""

        saver = tf.train.Saver()
        with tf.Session(config=allow_growth_config) as sess:
            if restore_model_path:
                logger.info("restoring model from %s", restore_model_path)
                saver.restore(sess, restore_model_path)
            else:
                assert self.saved_model_path, "{}".format(self.saved_model_path)
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
            with open(os.path.join(hyps_dir, "hyps"), "w",
                      encoding=ENCODING) as hyps_f:
                for hyp in hyps:
                    print(" ".join(hyp), file=hyps_f)
            with open(os.path.join(hyps_dir, "refs"), "w",
                      encoding=ENCODING) as refs_f:
                for ref in refs:
                    print(" ".join(ref), file=refs_f)

            test_per = utils.batch_per(hyps, refs)
            if not math.isclose(test_per, test_ler, rel_tol=1e-07):
                logger.warning("The label error rate from Tensorflow doesn't exactly"
                                "match the phoneme error rate calculated in persephone"
                                "Tensorflow %f, Persephone %f", test_ler, test_per)
            with open(os.path.join(hyps_dir, "test_per"), "w",
                      encoding=ENCODING) as per_f:
                print("LER: %f" % (test_ler), file=per_f)

    def output_best_scores(self, best_epoch_str: str) -> None:
        """Output best scores to the filesystem"""
        BEST_SCORES_FILENAME = "best_scores.txt"
        with open(os.path.join(self.exp_dir, BEST_SCORES_FILENAME),
                  "w", encoding=ENCODING) as best_f:
            print(best_epoch_str, file=best_f, flush=True)

    def train(self, *, early_stopping_steps: int = 10, min_epochs: int = 30,
              max_valid_ler: float = 1.0, max_train_ler: float = 0.3,
              max_epochs: int = 100, restore_model_path: Optional[str]=None,
              epoch_callback: Optional[Callable[[Dict], None]]=None) -> None:
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
            epoch_callback: A callback that is called at the end of each training epoch.
                            The parameters passed to the callable will be the epoch number,
                            the current training LER and the current validation LER.
                            This can be useful for progress reporting.
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
            with open(os.path.join(self.exp_dir, "train_description.txt"), 
                      "w", encoding=ENCODING) as desc_f:
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
            with open(training_log_path, "w",
                      encoding=ENCODING) as out_file:
                for epoch in itertools.count(start=1):
                    print("\nexp_dir %s, epoch %d" % (self.exp_dir, epoch))
                    batch_gen = self.corpus_reader.train_batch_gen()

                    train_ler_total = 0
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
                    #else:
                    #    raise PersephoneException("No training data was provided."
                    #                              " Check your batch generation.")

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
                    with open(os.path.join(hyps_dir, "epoch%d_hyps" % epoch),
                              "w", encoding=ENCODING) as hyps_f:
                        for hyp in hyps:
                            print(" ".join(hyp), file=hyps_f)
                    if epoch == 1:
                        with open(os.path.join(hyps_dir, "refs"), "w", 
                                  encoding=ENCODING) as refs_f:
                            for ref in refs:
                                print(" ".join(ref), file=refs_f)

                    valid_per = utils.batch_per(hyps, refs)

                    epoch_str = "Epoch %d. Training LER: %f, validation LER: %f" % (
                        epoch, (train_ler_total / (batch_i + 1)), valid_ler)
                    print(epoch_str, flush=True, file=out_file)
                    if best_epoch_str is None:
                        best_epoch_str = epoch_str

                    # Call the callback here if it was defined
                    if epoch_callback:
                        epoch_callback({
                            "epoch": epoch,
                            "training_ler": (train_ler_total / (batch_i + 1)), # current training LER
                            "valid_ler": valid_ler, # Current validation LER
                        })

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
                        with open(os.path.join(hyps_dir, "best_hyps"),
                                  "w", encoding=ENCODING) as hyps_f:
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

                # Check we actually saved a checkpoint
                if not self.saved_model_path:
                    raise PersephoneException(
                        "No checkpoint was saved so model evaluation cannot be performed. "
                        "This can happen if the validaion LER never converges.")
                # Finally, run evaluation on the test set.
                self.eval(restore_model_path=self.saved_model_path)
