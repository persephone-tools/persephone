import rnn_ctc
import timit
from utils import target_list_to_sparse_tensor

EXP_DIR = "../exp"

def train(batch_size, total_size, num_epochs, save=True, restore_model_path=None):
    """ Run an experiment. 

        batch_size: The number of utterances in each batch.
        total_size: The number of TIMIT training examples to use.
        num_epochs: The number of times to iterate over all the training
        examples.
    """

    # A generator that pumps out batches
    batch_gen = timit.batch_gen(batch_size=batch_size, labels="phonemes",
            total_size=total_size)
    # Load the data and observe the number of feats in the numpy arrays.
    freq_feats = next(batch_gen)[0].shape[-1]

    valid_x, valid_x_lens, valid_y = timit.valid_set(seed=0)
    valid_y_lens = np.asarray([len(s) for s in valid_y], dtype=np.int32)
    valid_y = target_list_to_sparse_tensor(valid_y)

    inputs = tf.placeholder(tf.float32, [None, None, freq_feats])
    input_lens = tf.placeholder(tf.int32, [None])
    targets = tf.sparse_placeholder(tf.int32)
    # The lengths of the target sequences.
    seq_lens = tf.placeholder(tf.int32, [None])

    model = rnn_ctc.Model(inputs, input_lens, targets, seq_lens,
                   vocab_size=timit.num_phones+2)

    if save:
        saver = tf.train.Saver()

    sess = tf.Session()

    if restore_model_path:
        saver.restore(sess, restore_model_path)
    else:
        sess.run(tf.global_variables_initializer())

    for epoch in range(1,num_epochs+1):
        batch_gen = timit.batch_gen(batch_size=batch_size, labels="phonemes",
                total_size=total_size, rand=True)

        err_total = 0
        for batch_i, batch in enumerate(batch_gen):
            batch_x, x_lens, batch_y = batch
            #batch_seq_lens = np.asarray(
            #        [len(s) for s in batch_y], dtype=np.int32)
            batch_y = target_list_to_sparse_tensor(batch_y)

            feed_dict={inputs: batch_x, input_lens: x_lens, targets: batch_y}
            _, error, decoded = sess.run(
                    [model.optimizer, model.ler, model.decoded],
                    feed_dict=feed_dict)
            #timit.error_rate(batch_y, decoded)
            if batch_i == 0:
                print(decoded[0].values)
                print(batch_y[1])

            err_total += error

        feed_dict={inputs: valid_x, input_lens: valid_x_lens,
                targets: valid_y, seq_lens: valid_y_lens}
        valid_ler, decoded = sess.run([model.ler, model.decoded], feed_dict=feed_dict)
        #total_per += timit.error_rate(utter_y, decoded)

        print("Epoch %d. Training LER: %f, validation LER: %f" % (
                epoch, (err_total / (batch_i + 1)), valid_ler), flush=True)

        # Give the model an appropriate number and save it in the EXP_DIR
        if epoch % 50 == 0 and save:
            n = max([int(fn.split(".")[0]) for fn in os.listdir(EXP_DIR) if fn.split(".")[0].isdigit()])
            path = os.path.join(EXP_DIR, "%d.model.epoch%d.ckpt" % (n, epoch))
            save_path = saver.save(sess, path)

    sess.close()

if __name__ == "__main__":
    train(batch_size=64, total_size=3648, num_epochs=300, save=True)
    #train(batch_size=32, total_size=3696, num_epochs=300, save=True)
            #restore_model_path=os.path.join(EXP_DIR,"20.model.epoch100.ckpt"))
