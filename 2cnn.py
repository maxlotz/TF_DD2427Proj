'''
Testing status
read_labeled_image_list definitely works
input_pipeline definitely works
'''

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

validation_file = "tiny-imagenet-200/val/validation.txt"
training_file = "tiny-imagenet-200/train/training.txt"

IMAGE_HEIGHT = 64
IMAGE_WIDTH =  64
CHANNELS =  3
NUM_CLASSES = 200
NUM_ITERATIONS = 100000

BATCH_SIZE = 100
MIN_AFTER_DEQUEUE =	1000
CAPACITY = MIN_AFTER_DEQUEUE + 3 * BATCH_SIZE #As recommended on tf website

def read_labeled_image_list(image_list_file):
    '''
	Reads a .txt file containing paths and labels

    Args:
       image_list_file: a .txt file with one /path/to/image per line eg "fpath/file1.jpg"
       followed by a label corresponding to the class number eg. "156"; seperated with a ' '.
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
	        filename, label = line[:-1].split(' ')
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

	images = ops.convert_to_tensor(image_list, dtype=dtypes.string)
	labels = ops.convert_to_tensor(label_list, dtype=dtypes.string)

	input_queue = tf.train.slice_input_producer(
		                                [images, labels],
		                                shuffle=True)
	#File reader and decoder for images and labels goes here
	image_filename = input_queue[0]
	file_content = tf.read_file(image_filename)
	image_data = tf.image.decode_jpeg(file_content, channels=3)
	label_data = input_queue[1]

	#Any resizing or processing (eg. normalising) goes here 
	image_data = tf.cast(image_data, tf.float32)
	mean, variance = tf.nn.moments(image_data, [0, 1, 2])
	image_data = (image_data - mean) #/ tf.sqrt(variance)
	label_data = tf.string_to_number(label_data, out_type=tf.int32)
	
	image_data.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS])

	image_batch, label_batch, image_filenames = tf.train.shuffle_batch(
		                                [image_data, label_data, image_filename],
		                                batch_size=BATCH_SIZE,
										capacity=CAPACITY,
                                        min_after_dequeue=MIN_AFTER_DEQUEUE,
		                                #,num_threads=1
		                                )

	return image_batch, label_batch, image_filenames

# use this as the input to weights_variable to get a xavier normal distribution
# see Glorot, Bengio 2010 http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
def xavier(fan_in):
	return 1.0/fan_in # or fan out, used on first and last layer

# use this for a xavier uniform distribution
def weights_xav_uniform_variable(shape, fan_in, fan_out):
	u = 6.0**0.5/(fan_in + fan_out)**0.5
	initial = tf.random_uniform(shape, -u, u)
	return tf.Variable(initial)

def weights_variable(shape, std):
	initial = tf.truncated_normal(shape, stddev=std)
	return tf.Variable(initial)

# bias initialised as 0
def bias_variable(shape):
  initial = tf.constant(0.0, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# returns 1 for each batch image that was in top k predictions, 0 for each one that isn't. output type = tf.float32
def topk(prediction, target, k):
	topkbool = tf.nn.in_top_k(prediction, target, k)
	topkfloat = tf.cast(topkbool, tf.float32)
	return topkfloat

# BATCH FILE AND LABEL READING HERE

#validation_images, validation_labels = read_labeled_image_list(validation_file)
#validation_image_batch, validation_label_batch = input_pipeline(validation_images, validation_labels)

train_images, train_labels = read_labeled_image_list(training_file)
train_image_batch, train_label_batch, train_image_filenames = input_pipeline(train_images, train_labels)
train_image_batch = train_image_batch

print "input pipeline ready"

f = train_image_filenames
x = train_image_batch
y = train_label_batch # use this for accuracy
y_onehot = tf.one_hot(y,NUM_CLASSES) # use in one-hot format for loss

# NEURAL NETWORK BEGINS HERE!
# NETWORK ARCHITECTURE: conv,conv,maxpool,conv,conv,maxpool,conv,maxpool,full,full,full,softmax

#conv layer 1
#w_conv1 = weights_variable([3,3,3,128],xavier(3*3*128))
w_conv1 = weights_variable([3,3,3,128], 0.02)
b_conv1 = bias_variable([128])
h_conv1 = tf.nn.relu(conv2d(x,w_conv1) + b_conv1)

#conv layer 2
#w_conv2 = weights_xav_uniform_variable([3,3,128,128],3*3*128,3*3*128)
w_conv2 = weights_variable([3,3,128,128], 0.02)
b_conv2 = bias_variable([128])
h_conv2 = tf.nn.relu(conv2d(h_conv1,w_conv2) + b_conv2)

#maxpool1, images now 32x32
h_pool1 = max_pool_2x2(h_conv2)

#conv layer 3
#w_conv3 = weights_xav_uniform_variable([3,3,128,128],3*3*128,3*3*128)
w_conv3 = weights_variable([3,3,128,128], 0.01)
b_conv3 = bias_variable([128])
h_conv3 = tf.nn.relu(conv2d(h_pool1,w_conv3) + b_conv3)

#conv layer 4
#w_conv4 = weights_xav_uniform_variable([3,3,128,128],3*3*128,3*3*128)
w_conv4 = weights_variable([3,3,128,128], 0.01)
b_conv4 = bias_variable([128])
h_conv4 = tf.nn.relu(conv2d(h_conv3,w_conv4) + b_conv4)

#maxpool2, images now 16x16
h_pool2 = max_pool_2x2(h_conv4)

#conv layer 5
#w_conv5 = weights_xav_uniform_variable([3,3,128,128],3*3*128,400)
w_conv5 = weights_variable([3,3,128,128], 0.01)
b_conv5 = bias_variable([128])
h_conv5 = tf.nn.relu(conv2d(h_pool2,w_conv5) + b_conv5)

#maxpool3, imags now 8x8
h_pool3 = max_pool_2x2(h_conv5)

#fully connected layer 1
#w_fc1 = weights_xav_uniform_variable([8*8*128,400],400,400)
w_fc1 = weights_variable([8*8*128,400], 0.02)
b_fc1 = bias_variable([400])
h_pool_3_flat = tf.reshape(h_pool3,[-1,8*8*128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool_3_flat, w_fc1) + b_fc1)

#fully connected layer 2
#w_fc2 = weights_xav_uniform_variable([400,400],400,200)
w_fc2 = weights_variable([400,400], 0.02)
b_fc2 = bias_variable([400])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)

#fully connected layer 3
#w_fc3 = weights_variable([400,200],xavier(400))
w_fc3 = weights_variable([400,200], 0.04)
b_fc3 = bias_variable([200])
h_fc3 = tf.nn.relu(tf.matmul(h_fc1, w_fc3) + b_fc3)

#prediction layer
y_pred = tf.nn.softmax(h_fc3)

#loss measure: cross entropy
loss = tf.reduce_mean(-tf.reduce_sum(y_onehot * tf.log(y_pred), reduction_indices=[1]))

#optimizer here
train_step = tf.train.GradientDescentOptimizer(0.8).minimize(loss)

#accuracy measure here

with tf.Session() as sess:
  
  # initialize the variables
  sess.run(tf.initialize_all_variables())
  
  # initialize the queue threads to start to shovel data
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  # CODE FOR TESTING 
  x = sess.run(tf.shape(y)) # to see the shape eg [100 200]
  x_ = sess.run(tf.shape(x)) # to see value and the type eg [array [0.12...]dtype=float32]

  print x
  print x_

  # # RUN CODE HERE
  # for i in range(NUM_ITERATIONS):
  # 	_, x = sess.run([train_step, loss])
  # 	print "iteration: " + str(i*100) + "\t" + "loss: " + str(x)

  # stop our queue threads and properly close the session
  coord.request_stop()
  coord.join(threads)
  sess.close()

'''
TESTING CODE HERE

# SHOWS FILE PIXEL COLOURS, LABELS, FILENAMES FOR FIRST 3 FILES IN BATCH
# File reader is reading the files it says it is, with the correct label, and correct colours
x_,y_,f_ = sess.run([x, y, f])
print "image 1 channel 1" # BATCH X HEIGHT x WIDTH X CHANNEL is standard TF format
print x_[0,:,:,0]
print "image 1 channel 2"
print x_[0,:,:,1]
print "image 1 channel 3"
print x_[0,:,:,2]
print "image 2 channel 1"
print x_[1,:,:,0]
print "image 2 channel 2"
print x_[1,:,:,1]
print "image 2 channel 3"
print x_[1,:,:,2]
print "image 3 channel 1"
print x_[2,:,:,0]
print "image 3 channel 2"
print x_[2,:,:,1]
print "image 3 channel 3"
print x_[2,:,:,2]
print "batch labels are:"
print y_[:3]
print "batch filenames are:"
print f_[:3]
'''