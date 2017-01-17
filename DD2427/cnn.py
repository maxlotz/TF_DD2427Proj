import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

validation_file = "validation_labels.csv"
training_file = "training_labels.csv"

IMAGE_HEIGHT = 64
IMAGE_WIDTH =  64
CHANNELS =  3
NUM_CLASSES = 200
NUM_ITERATIONS = 100000

BATCH_SIZE = 25
MIN_AFTER_DEQUEUE =	1000
CAPACITY = MIN_AFTER_DEQUEUE + 3 * BATCH_SIZE #As recommended on tf website

def read_labeled_image_list(image_list_file):
    '''
	Reads a .txt file containing paths and labels

    Args:
       image_list_file: a .txt file with one /path/to/image per line eg "fpath/file1.jpg"
       followed by a label corresponding to the class number eg. "156"; seperated with a ','.
       Class number ranges from 0 to NUM_CLASSES - 1
       example of file contents: 
       fpath/file1.jpeg 126
       fpath/file2.jpeg 0
       fpath/file3.jpeg 57
       fpath/file4.jpeg 199

    Returns:
       a list with the filenames eg. ["fpath/file1.jpeg","fpath/file2.jpeg"...]
       and a list with the labels eg. ["126","0"...]
    '''
    with open(image_list_file, 'r') as f:
	    filenames = []
	    labels = []
	    for line in f:
	        filename, label = line[:-1].split(',')
	        filenames.append(filename)
	        labels.append(label)
	    return filenames, labels

def input_pipeline(image_list, label_list):
	'''
	Accepts list of jpeg image filenames and its corresponding list of labels (the class number), returns a batch of image data and a batch of label data ready for tensorflow	processing

    Args:
       see above, read_labeled_image_list return type

    Returns:
       image_batch: a batch of images tensors (BATCH X HEIGHT X WIDTH X CHANNELS) of type tf.float32 
	   label_batch: a batch of label tensors (BATCH) of type tf.int32 [126,0,57,199...]
	Notes:
		I decided to leave the output labels as ints, not as onehot as many tensorflow functions use ints instead of onehot, so onehot can be used when needed.
    '''
	with tf.name_scope('input'):
		image_queue, label_queue = tf.train.slice_input_producer(
			                                [image_list, label_list],
			                                shuffle=True)
		#File reader and decoder for images and labels goes here
		with tf.name_scope('image'): 

			with tf.name_scope('decode'): 
				file_content = tf.read_file(image_queue)
				image_data = tf.image.decode_jpeg(file_content, channels=3)
		
			#Any resizing or processing (eg. normalising) goes here
			with tf.name_scope('normalize'): 
				image_data = tf.cast(image_data, tf.float32)
				image_data = tf.image.per_image_whitening(image_data)
				image_data.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS])

		
		with tf.name_scope('label'): 

			with tf.name_scope('decode'): 
				label_data = tf.string_to_number(label_queue, out_type=tf.int32)

		image_batch, label_batch = tf.train.shuffle_batch(
			                                [image_data, label_data],
			                                batch_size=BATCH_SIZE,
											capacity=CAPACITY,
	                                        min_after_dequeue=MIN_AFTER_DEQUEUE,
			                                #,num_threads=1
			                                )

		# tf.image_summary('image', image_batch, max_images=10)
  #       tf.histogram_summary('image', image_batch)

	return image_batch, label_batch

# use this as the input to weights_variable to get a xavier normal distribution
# see Glorot, Bengio 2010 http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf

# use this for a xavier uniform distribution

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

# returns 1 for each batch image that was in top k predictions, 0 for each one that isn't. output type = tf.float32
def topk(prediction, target, k):
	topkbool = tf.nn.in_top_k(prediction, target, k)
	topkfloat = tf.cast(topkbool, tf.float32)
	return topkfloat

# BATCH FILE AND LABEL READING HERE

#validation_images, validation_labels = read_labeled_image_list(validation_file)
#validation_image_batch, validation_label_batch = input_pipeline(validation_images, validation_labels)

train_images, train_labels = read_labeled_image_list(training_file)
train_image_batch, train_label_batch = input_pipeline(train_images, train_labels)
train_image_batch = train_image_batch

print "input pipeline ready"

x = train_image_batch
y = train_label_batch # use this for accuracy
y_onehot = tf.one_hot(y,NUM_CLASSES) # use in one-hot format for loss

# NEURAL NETWORK BEGINS HERE!
# NETWORK ARCHITECTURE: conv,conv,maxpool,conv,conv,maxpool,conv,maxpool,full,full,full,softmax

#conv layer 1
with tf.name_scope('conv1') as scope:
	w = xavier_weights_variable([3,3,3,128])
	#w = weights_variable([3,3,3,128], 0.02)
	b = bias_variable([128])
	conv1 = tf.nn.relu(conv2d(x,w) + b, name="activations")

#conv layer 2
with tf.name_scope('conv2') as scope:
	w = xavier_weights_variable([3,3,128,128])
	#w = weights_variable([3,3,128,128], 0.02)
	b = bias_variable([128])
	conv2 = tf.nn.relu(conv2d(conv1,w) + b, name="activations")

#maxpool1, images now 32x32
with tf.name_scope('pool1') as scope:
	pool1 = max_pool_2x2(conv2)

#conv layer 3
with tf.name_scope('conv3') as scope:
	w = xavier_weights_variable([3,3,128,128])
	#w = weights_variable([3,3,128,128], 0.02)
	b = bias_variable([128])
	conv3 = tf.nn.relu(conv2d(pool1,w) + b, name="activations")

#conv layer 4
with tf.name_scope('conv4') as scope:
	w = xavier_weights_variable([3,3,128,128])
	#w = weights_variable([3,3,128,128], 0.02)
	b = bias_variable([128])
	conv4 = tf.nn.relu(conv2d(conv3,w) + b, name="activations")

#maxpool2, images now 16x16
with tf.name_scope('pool2') as scope:
	pool2 = max_pool_2x2(conv4)

#conv layer 5
with tf.name_scope('conv5') as scope:
	w = xavier_weights_variable([3,3,128,128])
	#w = weights_variable([3,3,128,128], 0.02)
	b = bias_variable([128])
	conv5 = tf.nn.relu(conv2d(pool2,w) + b, name="activations")

#maxpool3, imags now 8x8
with tf.name_scope('pool3') as scope:
	pool3 = max_pool_2x2(conv5)

#fully connected layer 1
with tf.name_scope('fully_connected1') as scope:
	w = xavier_weights_variable([8*8*128,400])
	#w = weights_variable([8*8*128,400], 0.01)
	b = bias_variable([400])
	pool3_flat = tf.reshape(pool3,[-1,8*8*128])
	fully_connected1 = tf.nn.relu(tf.matmul(pool3_flat, w) + b, name="activations")

#fully connected layer 2
with tf.name_scope('fully_connected2') as scope:
	w = xavier_weights_variable([400,400])
	#w = weights_variable([400,400], 0.01)
	b = bias_variable([400])
	fully_connected2 = tf.nn.relu(tf.matmul(fully_connected1, w) + b, name="activations")

#fully connected layer 3
with tf.name_scope('fully_connected3') as scope:
	w = xavier_weights_variable([400,200])
	#w = weights_variable([400,200], 0.01)
	b = bias_variable([200])
	y_pred = tf.matmul(fully_connected2, w) + b

#loss measure
with tf.name_scope('loss') as scope:
	loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_pred, train_label_batch, name='loss'))

#optimizer here
opt = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
grads_and_vars = opt.compute_gradients(loss)

for grad, var in grads_and_vars:
  tf.histogram_summary('gradients/' + grad.op.name, grad)

train_step = opt.apply_gradients(grads_and_vars)

#accuracy measure here

# summary for loss
tf.scalar_summary("cross entropy loss", loss)
summary_op = tf.merge_all_summaries()

with tf.Session() as sess:
  
  # initialize the variables
  sess.run(tf.initialize_all_variables())

  # create log writer object, this writes the data to the directory /logs
  # to visualise enter the following into terminal: tensorboard --logdir=logs/ --port=6006
  # then navigate to http://0.0.0.0:6006/
  writer = tf.train.SummaryWriter("logs/", graph=tf.get_default_graph())
  
  # initialize the queue threads to start to shovel data
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  # # CODE FOR TESTING 
  # ashape_, a_ = sess.run([tf.shape(a),a])
  # print ashape_
  # print "\n"
  # print a

  # RUN CODE HERE
  for i in range(NUM_ITERATIONS):
  	_, summary, loss_ = sess.run([train_step, summary_op, loss])

  	# tensorboard wont show the x axis value if its an int
  	writer.add_summary(summary, float(i))
  	print "iteration: " + str((i+1)) + "    loss: " + str(loss_)

  # stop our queue threads and properly close the session
  coord.request_stop()
  coord.join(threads)
  sess.close()