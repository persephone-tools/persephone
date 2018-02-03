# pylint: disable-all

""" Implement the CNN-based acoustic model of Zhang et al. 2017
    'Towards End-to-End Speech Recognition with Deep Convolutional Neural
    Networks'.
"""


import os
import random

import numpy as np
import tensorflow as tf

#from preprocess_timit import len_longest_timit_utterance
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)

random.seed(0)

def count_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        print(shape)
        print(len(shape))
        variable_parametes = 1
        for dim in shape:
            print(dim)
            variable_parametes *= dim.value
        print(variable_parametes)
        total_parameters += variable_parametes
    print(total_parameters)

def num_phones(path="/home/oadams/mam/data/timit/"):
    phn_set = set()
    for root, dirnames, filenames in os.walk(path):
        for fn in filenames:
            prefix = fn.split(".")[0] # Get the file prefix
            if fn.endswith(".phn"):
                path = os.path.join(root, fn)
                with open(path) as phn_f:
                    phns = phn_f.readline().split()
                for phn in phns:
                    phn_set.add(phn)
    return len(phn_set)

def load_timit(path="/home/oadams/mam/data/timit/train", rand=True,
        batch_size=10, labels="phonemes"):
    """ Load the already preprocessed TIMIT data. """

    def create_one_hot(i, num_classes):
        """ Takes a list or array with numbers representing classes, and
        creates a corresponding 2-dimensional array of one-hot vectors."""

        one_hot = np.zeros((1, num_classes))
        one_hot[0, i] = 1
        return one_hot

    train_paths = []
    dialect_classes = set()

    for root, dirnames, filenames in os.walk(path):
        for fn in filenames:
            prefix = fn.split(".")[0] # Get the file prefix
            if fn.endswith(".log_mel_filterbank.zero_pad.npy"):
                # Add to filename list.
                path = os.path.join(root, fn)
                train_paths.append(os.path.join(root, fn))
                dialect_classes.add(path.split("/")[-3])

    dialect_classes = sorted(list(dialect_classes))
    dr_class_map = dict(zip(dialect_classes, range(0, len(dialect_classes))))

    if rand:
        # Load a random subset of the training set of batch_size
        path_batch = random.sample(train_paths, batch_size)
        feats = [np.load(path) for path in path_batch]
        x = np.array(feats)
        if labels == "dialects":
            y_labels = np.array([dr_class_map[path.split("/")[-3]] for path in path_batch])
            # Make into one-hot vectors
            batch_y = np.concatenate([create_one_hot(label, len(dialect_classes))
                    for label in y_labels])
        elif labels == "phonemes":
            phn_paths = ["".join(path.split(".")[:-3])+".phn" for path in path_batch]
            batch_y = []
            for phn_path in phn_paths:
                with open(phn_path) as phn_f:
                    batch_y.append(phn_f.readline().split())
    else:
        raise Exception("Not implemented.")

    batch_x = np.array(feats)
    return batch_x, batch_y

# Create the graph
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # Shift the window by 1 in each direction
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

def max_pool_1x3(x):
    # Use max pooling over each 1x3 windows, as per Zhang et al.
    return tf.nn.max_pool(x, ksize=[1, 1, 3, 1], strides=[1, 1, 3, 1],
                          padding="SAME")


# There are 26 log Mel filterbank features and the 3 channels correspond to
# static, delta and double-delta.
x = tf.placeholder(tf.float32,
                    shape=[None, None, 26, 3])
# There are 8 dialect classes.
y = tf.placeholder(tf.float32, shape=[None, 8])

keep_prob = tf.placeholder(tf.float32)

# The 1st, 2nd and 4th dimensions I can arbitrarily pick, but the 3rd dimension
# is the input number of channels and must correspond to x. This weight becomes
# the filter argument to tf.nn.conv2d.
W_conv1 = weight_variable([5, 3, 3, 128])
b_conv1 = bias_variable([128])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_conv1_drop = tf.nn.dropout(h_conv1, keep_prob)

h_pool1 = max_pool_1x3(h_conv1_drop)
h_pool1_drop = tf.nn.dropout(h_pool1, keep_prob)

W_conv2 = weight_variable([5, 3, 128, 128])
b_conv2 = bias_variable([128])
h_conv2 = tf.nn.relu(conv2d(h_pool1_drop, W_conv2) + b_conv2)
h_conv2_drop = tf.nn.dropout(h_conv2, keep_prob)

W_conv3 = weight_variable([5, 3, 128, 128])
b_conv3 = bias_variable([128])
h_conv3 = tf.nn.relu(conv2d(h_conv2_drop, W_conv3) + b_conv3)
h_conv3_drop = tf.nn.dropout(h_conv3, keep_prob)

W_conv4 = weight_variable([5, 3, 128, 128])
b_conv4 = bias_variable([128])
h_conv4 = tf.nn.relu(conv2d(h_conv3_drop, W_conv4) + b_conv4)
h_conv4_drop = tf.nn.dropout(h_conv4, keep_prob)

W_conv5 = weight_variable([5, 3, 128, 256])
b_conv5 = bias_variable([256])
h_conv5 = tf.nn.relu(conv2d(h_conv4_drop, W_conv5) + b_conv5)
h_conv5_drop = tf.nn.dropout(h_conv5, keep_prob)

W_conv6 = weight_variable([5, 3, 256, 256])
b_conv6 = bias_variable([256])
h_conv6 = tf.nn.relu(conv2d(h_conv5_drop, W_conv6) + b_conv6)
h_conv6_drop = tf.nn.dropout(h_conv6, keep_prob)

W_conv7 = weight_variable([5, 3, 256, 256])
b_conv7 = bias_variable([256])
h_conv7 = tf.nn.relu(conv2d(h_conv6_drop, W_conv7) + b_conv7)
h_conv7_drop = tf.nn.dropout(h_conv7, keep_prob)

W_conv8 = weight_variable([5, 3, 256, 256])
b_conv8 = bias_variable([256])
h_conv8 = tf.nn.relu(conv2d(h_conv7_drop, W_conv8) + b_conv8)
h_conv8_drop = tf.nn.dropout(h_conv8, keep_prob)

W_conv9 = weight_variable([5, 3, 256, 256])
b_conv9 = bias_variable([256])
h_conv9 = tf.nn.relu(conv2d(h_conv8_drop, W_conv9) + b_conv9)
h_conv9_drop = tf.nn.dropout(h_conv9, keep_prob)

W_conv10 = weight_variable([5, 3, 256, 256])
b_conv10 = bias_variable([256])
h_conv10 = tf.nn.relu(conv2d(h_conv9_drop, W_conv10) + b_conv10)

h_conv10_flat = tf.reshape(h_conv10, [-1, 778*9*256])
h_conv10_flat_drop = tf.nn.dropout(h_conv10_flat, keep_prob)

W_fc1 = weight_variable([778*9*256, 128])
b_fc1 = bias_variable([128])
h_fc1 = tf.nn.relu(tf.matmul(h_conv10_flat_drop, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([128, 128])
b_fc2 = bias_variable([128])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

num_labels = num_phones() + 1

W_fc3 = weight_variable([128, num_labels])
b_fc3 = bias_variable([num_labels])
y_fc = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_fc))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_fc, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.initialize_all_variables().run()

total_acc = 0.0
for i in range(1000):
    batch = load_timit(batch_size=10)
    train_accuracy = accuracy.eval(
            feed_dict={x:batch[0], y: batch[1], keep_prob: 0.7})
    total_acc += train_accuracy
    print("Step %d, training accuracy %g, avg acc: %g" % (i, train_accuracy, total_acc/i))
    train_step.run(
            feed_dict={x:batch[0], y: batch[1], keep_prob: 0.7})
