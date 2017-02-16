import os
import random
import sys

import numpy
import tensorflow as tf

#from preprocess_timit import len_longest_timit_utterance

random.seed(0)

def load_timit(path="/home/oadams/mam/data/timit/train", rand=True, batch_size=100):
    """ Load the already preprocessed TIMIT data. """

    train_paths = []
    dr_classes = set()

    for root, dirnames, filenames in os.walk(path):
        for fn in filenames:
            prefix = fn.split(".")[0] # Get the file prefix
            if fn.endswith(".log_mel_filterbank.zero_pad.npy"):
                # Add to filename list.
                path = os.path.join(root,fn)
                train_paths.append(os.path.join(root,fn))
                dr_classes.add(path.split("/")[-3])

    dr_classes = sorted(list(dr_classes))
    dr_class_map = dict(zip(dr_classes, range(0,len(dr_classes))))

    if rand:
        # Load a random subset of the training set of batch_size 
        path_batch = random.sample(train_paths, batch_size)
        feats = [numpy.load(path) for path in path_batch]
        x = numpy.array(feats)
        batch_y = numpy.array([dr_class_map[path.split("/")[-3]] for path in path_batch])
    else:
        raise Exception("Not implemented.")

    batch_x = numpy.array(feats)
    print(batch_x.shape)
    print(batch_y.shape)
    return batch_x, batch_y

def create_graph():

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        # Shift the window by 1 in each direction
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

    def max_pool_2x2(x):
        # Use max pooling over each 2x2 window.
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1,2,2,1],
                              padding="SAME")

    # There are 26 log Mel filterbank features and the 3 channels correspond to
    # static, delta and double-delta.
    x = tf.placeholder(tf.float32,
                        shape=[None, None, 26, 3])
    # There are 8 dialect classes.
    y = tf.placeholder(tf.float32, shape=[None, 8])

create_graph()
batch = load_timit()
