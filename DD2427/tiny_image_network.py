import tensorflow as tf
import layers

def inference(x):
	#conv layer 1
	with tf.name_scope('conv1') as scope:
		w = layers.xavier_weights_variable([3,3,3,128])
		b = layers.bias_variable([128])
		conv1 = tf.nn.relu(layers.conv2d(x,w) + b, name="activations")

	#conv layer 2
	with tf.name_scope('conv2') as scope:
		w = layers.xavier_weights_variable([3,3,128,128])
		b = layers.bias_variable([128])
		conv2 = tf.nn.relu(layers.conv2d(conv1,w) + b, name="activations")

	#maxpool1, images now 32x32
	with tf.name_scope('pool1') as scope:
		pool1 = layers.max_pool_2x2(conv2)

	#conv layer 3
	with tf.name_scope('conv3') as scope:
		w = layers.xavier_weights_variable([3,3,128,128])
		b = layers.bias_variable([128])
		conv3 = layers.tf.nn.relu(layers.conv2d(pool1,w) + b, name="activations")

	#conv layer 4
	with tf.name_scope('conv4') as scope:
		w = layers.xavier_weights_variable([3,3,128,128])
		b = layers.bias_variable([128])
		conv4 = tf.nn.relu(layers.conv2d(conv3,w) + b, name="activations")

	#maxpool2, images now 16x16
	with tf.name_scope('pool2') as scope:
		pool2 = layers.max_pool_2x2(conv4)

	#conv layer 5
	with tf.name_scope('conv5') as scope:
		w = layers.xavier_weights_variable([3,3,128,128])
		b = layers.bias_variable([128])
		conv5 = tf.nn.relu(layers.conv2d(pool2,w) + b, name="activations")

	#maxpool3, imags now 8x8
	with tf.name_scope('pool3') as scope:
		pool3 = layers.max_pool_2x2(conv5)

	#fully connected layer 1
	with tf.name_scope('fully_connected1') as scope:
		w = layers.xavier_weights_variable([8*8*128,400])
		b = layers.bias_variable([400])
		pool3_flat = tf.reshape(pool3,[-1,8*8*128])
		fully_connected1 = tf.nn.relu(tf.matmul(pool3_flat, w) + b, name="activations")

	#fully connected layer 2
	with tf.name_scope('fully_connected2') as scope:
		w = layers.xavier_weights_variable([400,400])
		b = layers.bias_variable([400])
		fully_connected2 = tf.nn.relu(tf.matmul(fully_connected1, w) + b, name="activations")

	#fully connected layer 3
	with tf.name_scope('fully_connected3') as scope:
		w = layers.xavier_weights_variable([400,200])
		b = layers.bias_variable([200])
		y_pred = tf.matmul(fully_connected2, w) + b

	return y_pred