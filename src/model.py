""" Generic model for automatic speech recognition. """

import inspect
import itertools
import os
import tensorflow as tf

import utils

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

    def train(self, corpus_batches, early_stopping_steps=10, min_epochs=30,
              restore_model_path=None):
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

        # Not technically the upper bound on a LER but we don't want to save if
        # it's not below this.
        best_valid_ler = 1.0
        steps_since_last_record = 0
        best_epoch = -1

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

        out_file = open(os.path.join(self.exp_dir, "train_log.txt"), "w")

        # Load the validation set
        valid_x, valid_x_lens, valid_y = corpus_batches.valid_set(seed=0)

        saver = tf.train.Saver()

        sess = tf.Session()

        if restore_model_path:
            saver.restore(sess, restore_model_path)
        else:
            sess.run(tf.global_variables_initializer())

        for epoch in itertools.count():
            batch_gen = corpus_batches.train_batch_gen()

            train_ler_total = 0
            batch_i = None
            for batch_i, batch in enumerate(batch_gen):
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

            valid_ler, dense_decoded, dense_ref = sess.run(
                    [self.ler, self.dense_decoded, self.dense_ref],
                    feed_dict=feed_dict)
            valid_per = corpus_batches.batch_per(dense_ref, dense_decoded)

            epoch_str = "Epoch %d. Training LER: %f, validation LER: %f, validation PER: %f" % (
                    epoch, (train_ler_total / (batch_i + 1)), valid_ler, valid_per)
            print(epoch_str, flush=True, file=out_file)

            # Implement early stopping.
            if valid_ler < best_valid_ler:
                print("New best valid_ler", file=out_file)
                best_valid_ler = valid_ler
                best_epoch_str = epoch_str
                steps_since_last_record = 0
                best_epoch = epoch

                # Save the model.
                path = os.path.join(self.exp_dir, "model", "model_best.ckpt")
                if not os.path.exists(os.path.dirname(path)):
                    os.mkdir(os.path.dirname(path))
                saver.save(sess, path)
            else:
                print("Steps since last best valid_ler: %d" % (
                        steps_since_last_record), file=out_file)
                steps_since_last_record += 1
                if steps_since_last_record >= early_stopping_steps:
                    if epoch >= min_epochs:
                        # Then we've done the minimum number of epochs and can
                        # stop training.
                        print("""Stopping since best validation score hasn't been
                                beaten in %d epochs and at least %d have been
                                done""" % (early_stopping_steps, min_epochs),
                                file=out_file, flush=True)
                        with open(os.path.join(
                                self.exp_dir, "best_scores.txt"), "w") as best_f:
                            print(best_epoch_str, file=best_f, flush=True)
                            sess.close()
                            out_file.close()
                            return
                    else:
                        # Keep training because we haven't done the minimum
                        # numper of epochs.
                        continue


        sess.close()
        out_file.close()
