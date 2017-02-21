import tensorflow as tf
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor

def SimpleSparseTensorFrom(x):
  """Create a very simple SparseTensor with dimensions (batch, time).
  Args:
    x: a list of lists of type int
  Returns:
    x_ix and x_val, the indices and values of the SparseTensor<2>.
  """
  x_ix = []
  x_val = []
  for batch_i, batch in enumerate(x):
    for time, val in enumerate(batch):
      x_ix.append([batch_i, time])
      x_val.append(val)
  x_shape = [len(x), np.asarray(x_ix).max(0)[1] + 1]
  x_ix = constant_op.constant(x_ix, dtypes.int64)
  x_val = constant_op.constant(x_val, dtypes.int32)
  x_shape = constant_op.constant(x_shape, dtypes.int64)

  return tf.SparseTensorValue(x_ix, x_val, x_shape)

def target_list_to_sparse_tensor(targetList):
    '''make tensorflow SparseTensor from list of targets, with each element
       in the list being a list or array with the values of the target sequence
       (e.g., the integer values of a character map for an ASR target string)
       See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/ctc/ctc_loss_op_test.py
       for example of SparseTensor format'''
    indices = []
    vals = []
    for tI, target in enumerate(targetList):
        for seqI, val in enumerate(target):
            indices.append([tI, seqI])
            vals.append(val)
    shape = [len(targetList), np.asarray(indices).max(0)[1]+1]
    return (np.array(indices), np.array(vals), np.array(shape))

def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape

logits = tf.Variable([[[2.,0.,0.],[0.,2.,0.],[0.,2.,0.],[2.,0.,0.]]], tf.float32)
logits = tf.Variable([[[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]], tf.float32)
logits = tf.Variable(tf.random_normal([1,4,3], stddev=0.35))
# Make time major.
logits = tf.transpose(logits, (1,0,2))
seq_len = [3]

#train_targets = target_list_to_sparse_tensor([[0,1,0]])
#print(targets)
#for t in targets:
#    print(t)
#import sys; sys.exit()
#targetIxs = tf.placeholder(tf.int64)
#targetVals = tf.placeholder(tf.int32)
#targetShape = tf.placeholder(tf.int64)
#targetY = tf.SparseTensor(targetIxs, targetVals, targetShape)

train_targets = sparse_tuple_from([[0,1,0]])
print(train_targets)
train_targets = target_list_to_sparse_tensor([[0,1,0]])
print(train_targets)
#batchTargetSparse = train_targets
#batchTargetIxs, batchTargetVals, batchTargetShape = batchTargetSparse

targets = tf.sparse_placeholder(tf.int32)

loss = tf.nn.ctc_loss(targets, logits, seq_len)
cost = tf.reduce_mean(loss)
optimizer = tf.train.MomentumOptimizer(1e-4, 0.9).minimize(cost)

decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)
ler = tf.reduce_mean(tf.edit_distance(
        tf.cast(decoded[0], tf.int32), targets))


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    #sess.run(init, feed_dict={targetIxs: batchTargetIxs, targetVals: batchTargetVals,
    #                       targetShape: batchTargetShape})
    feed_dict = {targets: train_targets}
    sess.run(init, feed_dict=feed_dict)
    #train_targets_val = train_targets.eval(sess)
    for _ in range(1000):
        out_opt, out_ler, out_logits, out_decoded = sess.run([optimizer, ler, logits, decoded],
                feed_dict=feed_dict)
                #feed_dict={targetIxs: batchTargetIxs, targetVals: batchTargetVals,
                #           targetShape: batchTargetShape})
        #sess.run(optimizer)
        print(out_logits)
        print(out_ler)
        print(out_decoded)
