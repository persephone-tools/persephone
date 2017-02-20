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

  return sparse_tensor.SparseTensor(x_ix, x_val, x_shape)

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

logits = tf.Variable([[[2.,0.,0.],[0.,2.,0.],[0.,2.,0.],[2.,0.,0.]]], tf.float32)
logits = tf.Variable([[[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]], tf.float32)
logits = tf.Variable(tf.random_normal([1,4,3], stddev=0.35))
# Make time major.
logits = tf.transpose(logits, (1,0,2))
seq_len = [3]

#targets = target_list_to_sparse_tensor([[0,1,0]])
targets = SimpleSparseTensorFrom([[0,1,0]])

loss = tf.nn.ctc_loss(targets, logits, seq_len)
cost = tf.reduce_mean(loss)
optimizer = tf.train.MomentumOptimizer(1e-4, 0.9).minimize(cost)

decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)
ler = tf.reduce_mean(tf.edit_distance(
        tf.cast(decoded[0], tf.int32), targets))


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for _ in range(1000):
        out_opt, out_ler, out_logits, out_decoded = sess.run([optimizer, ler, logits, decoded])
        sess.run(optimizer)
        print(out_logits)
        print(out_ler)
        print(out_decoded)
