import inspect
import logging
import os
import shutil
import tensorflow as tf

import rnn_ctc
import timit
from utils import target_list_to_sparse_tensor

EXP_DIR = "../exp"

def train(model, batch_size, total_size, num_epochs,
        feat_type="mfcc13_d", save_n=None, restore_model_path=None):
    """ Run an experiment.

        batch_size: The number of utterances in each batch.
        total_size: The number of TIMIT training examples to use.
        num_epochs: The number of times to iterate over all the training
        examples.
        feat_type: Is the identifier for the type of features we're using.
        'mfcc13_d' means MFCCs of 13 dimensions with their first derivatives.
        save_n: Whether to save the model at every n epochs.
    """

    #Get information about training for the names of output files.
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    strs = ["%s=%s" % (arg, values[arg]) for arg in args if type(values[arg]) in [str, int, float]]
    fn = "train~" + "~".join(strs) + ".out"

    out_file = open(os.path.join(EXP_DIR, str(get_exp_dir_num()), fn), "w")

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

            feed_dict={model.batch_x: batch_x,
                       model.batch_x_lens: batch_x_lens,
                       model.batch_y: batch_y}

            _, ler, decoded = sess.run(
                    [model.optimizer, model.ler, model.decoded],
                    feed_dict=feed_dict)

            #timit.error_rate(batch_y, decoded)
            if batch_i == 0:
                logging.debug("Batch[0] hypothesis: ")
                logging.debug(decoded[0].values)
                logging.debug("Batch[0] Reference: ")
                logging.debug(batch_y[1])

            train_ler_total += ler

        feed_dict={model.batch_x: valid_x,
                   model.batch_x_lens: valid_x_lens,
                   model.batch_y: valid_y}
        valid_ler, dense_decoded, dense_ref = sess.run(
                [model.ler, model.dense_decoded, model.dense_ref],
                feed_dict=feed_dict)
        valid_per = timit.batch_per(dense_ref, dense_decoded)
        #total_per += timit.error_rate(utter_y, decoded)

        print("Epoch %d. Training LER: %f, validation LER: %f, validation PER: %f" % (
                epoch, (train_ler_total / (batch_i + 1)), valid_ler, valid_per),
                flush=True, file=out_file)

        # Give the model an appropriate number and save it in the EXP_DIR
        if save_n and epoch % save_n == 0:
            # Save the model
            path = os.path.join(EXP_DIR, str(get_exp_dir_num()),
                    "model.epoch%d.ckpt" % epoch)
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

def get_exp_dir_num():
    """ Gets the number of the current experiment directory."""
    return max([int(fn.split(".")[0]) for fn in os.listdir(EXP_DIR) if fn.split(".")[0].isdigit()])

def prep_exp_dir():
    """ Prepares an experiment directory by copying the code in this directory
    to it as is, and setting the logger to write to files in that
    directory."""

    n = get_exp_dir_num()
    n = n + 1
    code_dir = os.path.join(EXP_DIR, str(n), "code")
    os.makedirs(code_dir)
    for fn in os.listdir():
        if fn.endswith(".py"):
            shutil.copyfile(fn, os.path.join(code_dir, fn))

    logging.basicConfig(filename=os.path.join(EXP_DIR, str(n), "debug.log"),
                        filemode="w", level=logging.DEBUG)

if __name__ == "__main__":

    # Prepares a new experiment dir for all logging.
    #prep_exp_dir()

    # Vocab size is two more than the number of TIMIT labels. This is because
    # we one extra for a blank label in CTC, and also another extra so that 0
    # can be used for dynamic padding in our minibatches.
    #feat_type="log_mel_filterbank"
    feat_type="log_mel_filterbank"
    num_feats = timit.num_feats(feat_type)
    model = rnn_ctc.Model(vocab_size=timit.num_phones+2, num_layers=3,
                          num_feats=num_feats)

    for i in range(6,13):
        train(model=model, batch_size=64, total_size=2**i, num_epochs=100,
              feat_type=feat_type, save_n=25)
