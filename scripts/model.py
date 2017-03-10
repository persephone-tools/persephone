import inspect
import os
import tensorflow as tf

import timit
import config

class Model:
    """ Generic model for our ASR tasks. """


    def train(self, batch_size, total_size, num_epochs, feat_type, save_n,
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

        #Get information about training for the names of output files.
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        with open(os.path.join(self.exp_dir, "train_description.txt"), "w") as f:
            for arg in args:
                print("%s=%s" % (arg, values[arg]), file=f)

        out_file = open(os.path.join(self.exp_dir, "train.log"), "w")

        # Load the validation set
        valid_x, valid_x_lens, valid_y = timit.valid_set(seed=0)

        if save_n:
            saver = tf.train.Saver()

        sess = tf.Session()

        if restore_model_path:
            saver.restore(sess, restore_model_path)
        else:
            sess.run(tf.global_variables_initializer())

        for epoch in range(1,num_epochs+1):
            batch_gen = timit.batch_gen(batch_size=batch_size,
                                        total_size=total_size)

            train_ler_total = 0
            for batch_i, batch in enumerate(batch_gen):
                batch_x, batch_x_lens, batch_y = batch

                feed_dict={self.batch_x: batch_x,
                           self.batch_x_lens: batch_x_lens,
                           self.batch_y: batch_y}

                _, ler, decoded = sess.run(
                        [self.optimizer, self.ler, self.decoded],
                        feed_dict=feed_dict)

                #timit.error_rate(batch_y, decoded)
                #if batch_i == 0:
                #    logging.debug("Batch[0] hypothesis: ")
                #    logging.debug(decoded[0].values)
                #    logging.debug("Batch[0] Reference: ")
                #    logging.debug(batch_y[1])

                train_ler_total += ler

            feed_dict={self.batch_x: valid_x,
                       self.batch_x_lens: valid_x_lens,
                       self.batch_y: valid_y}
            valid_ler, dense_decoded, dense_ref = sess.run(
                    [self.ler, self.dense_decoded, self.dense_ref],
                    feed_dict=feed_dict)
            valid_per = timit.batch_per(dense_ref, dense_decoded)
            #total_per += timit.error_rate(utter_y, decoded)

            print("Epoch %d. Training LER: %f, validation LER: %f, validation PER: %f" % (
                    epoch, (train_ler_total / (batch_i + 1)), valid_ler, valid_per),
                    flush=True, file=out_file)

            # Give the model an appropriate number and save it.
            if save_n and epoch % save_n == 0:
                # Save the model
                path = os.path.join(self.exp_dir, "model", "model.epoch%d.ckpt" % epoch)
                save_path = saver.save(sess, path)

                # Get the validation PER. We do this less often because it's
                # compoutationally more expensive. This is because we calculate the
                # PER for each utterance in the validation set independently.
                #total_per = 0
                #for i in range(len(valid_x)):
                #    utter_x = np.array([valid_x[i]])
                #    utter_x_len = np.array([valid_x_lens[i]])
                #    utter_y = [valid_y[i]]

        sess.close()

        out_file.close()
