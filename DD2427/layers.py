import tensorflow as tf
import numpy as np

def weights_variable(shape, std):
	initial = tf.truncated_normal(shape, stddev=std, name='weights')
	return tf.Variable(initial)

def _get_fans(shape):
  if len(shape) == 2:  # fully connected layer
    fan_in = shape[0]
    fan_out = shape[1]
  elif len(shape) == 4:  # convolutional layer
    receptive_field_size = np.prod(shape[:2])
    fan_in = shape[-2] * receptive_field_size
    fan_out = shape[-1] * receptive_field_size
  else:
    raise ValueError('Unrecognized shape %s' % str(shape))
  return fan_in, fan_out


def xavier_weights_variable(shape, name='kaiming_variable'):
  fan_in, fan_out = _get_fans(shape)
  scale = np.sqrt(6. / fan_in)
  initial = tf.random_uniform(shape,
                              minval=-scale,
                              maxval=scale,
                              name='kaiming_init')
  return tf.Variable(initial, dtype=tf.float32, name=name)

# bias initialised as 0
def bias_variable(shape):
  initial = tf.constant(0.0, shape=shape, name='biases')
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name="pool")