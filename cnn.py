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

BATCH_SIZE = 1
MIN_AFTER_DEQUEUE =	1000
CAPACITY = MIN_AFTER_DEQUEUE + 3 * BATCH_SIZE #As recommended on tf website

def read_labeled_image_list(image_list_file):
    '''
	Reads a .txt file containing paths and labels

    Args:
       image_list_file: a .txt file with one /path/to/image per line eg "fpath/file1.jpg"
       followed by a label corresponding to the class number eg. "156"; seperated with a ' '

    Returns:
       a list with the filenames and a list with the labels (strings)
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
       image_list: a list of image filenames (list of strings) eg. ["file1.jpg", "file2.jpg"]
	   label_list: corresponding list of labels (list of strings) eg. ["3", "187"]

    Returns:
       image_batch: a batch of images tensors (height x width x channels x N) of type tf.float32 
	   label_batch: a batch of one-hot encoded label tensors (N x classes) of type tf.int32 eg. [[0, 0, 0, 1, 0, 0, ... 0],[0, 1, 0, 0, 0, ... 0]...]
    '''
	images = ops.convert_to_tensor(image_list, dtype=dtypes.string)
	labels = ops.convert_to_tensor(label_list, dtype=dtypes.string)

	input_queue = tf.train.slice_input_producer(
		                                [images, labels],
		                                shuffle=True)
	#File reader and decoder for images and labels goes here
	file_content = tf.read_file(input_queue[0])
	image_data = tf.image.decode_jpeg(file_content, channels=3)
	label_data = input_queue[1]

	#Any resizing or processing (eg. normalising) goes here 
	image_data = tf.cast(image_data, tf.float32)
	mean, variance = tf.nn.moments(image_data, [0, 1, 2])
	image_data = (image_data - mean) #/ tf.sqrt(variance)
	label_data = tf.string_to_number(label_data, out_type=tf.int32)
	label_data = tf.one_hot(label_data, NUM_CLASSES, dtype=tf.int32)
	label_data = tf.cast(label_data, tf.float32)
	
	image_data.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS])

	image_batch, label_batch = tf.train.shuffle_batch(
		                                [image_data, label_data],
		                                batch_size=BATCH_SIZE,
										capacity=CAPACITY,
                                        min_after_dequeue=MIN_AFTER_DEQUEUE,
		                                #,num_threads=1
		                                )
	
	return image_batch, label_batch



def bias_variable(shape):
  initial = tf.constant(0.0, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def topk(prediction, target_onehot, k):
	'''
	Returns a 1 for each image where the label was in the top k softmax predictions

	Args:
		prediction = softmax prediction for the batch, dimension BATCH x NUM_CLASSES
		target_onehot = label data in onehot format, dimension BATCH x NUM_CLASSES
		k = an integer	
	'''
	target = tf.argmax(target_onehot, 1) # turns one-hot labels back into int
	topkbool = tf.nn.in_top_k(prediction, target, k) # TARGET MUST BE INT LABELS, NOT ONE-HOT
	topkfloat = tf.cast(topkbool, tf.float32) # casts boolean to a float, so that operations with other floats can be performed (tf.reduce_mean)
	return topkfloat

# NETWORK ARCHITECTURE: 5 conv, 3 fully, 1 softmax

#validation_images, validation_labels = read_labeled_image_list(validation_file)
#validation_image_batch, validation_label_batch = input_pipeline(validation_images, validation_labels)

train_images, train_labels = read_labeled_image_list(training_file)
train_image_batch, train_label_batch = input_pipeline(train_images, train_labels)

print "input pipeline ready"

x = train_image_batch
y = train_label_batch

with tf.variable_scope('conv1') as scope:
	kernel = tf.get_variable("W1", shape=[3,3,3,128],
           initializer=tf.contrib.layers.xavier_initializer())
	bias = bias_variable([128])
	conv1 = tf.nn.relu(conv2d(x,kernel) + bias, name=scope.name)

with tf.variable_scope('conv2') as scope:
	kernel = tf.get_variable("W2", shape=[3,3,128,128],
           initializer=tf.contrib.layers.xavier_initializer())
	bias = bias_variable([128])
	conv2 = tf.nn.relu(conv2d(conv1,kernel) + bias, name=scope.name)

pool1 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                       strides=[1, 2, 2, 1], padding='SAME', name='pool1')

with tf.variable_scope('conv3') as scope:
	kernel = tf.get_variable("W3", shape=[3,3,128,128],
           initializer=tf.contrib.layers.xavier_initializer())
	bias = bias_variable([128])
	conv3 = tf.nn.relu(conv2d(pool1,kernel) + bias, name=scope.name)

with tf.variable_scope('conv4') as scope:
	kernel = tf.get_variable("W4", shape=[3,3,128,128],
           initializer=tf.contrib.layers.xavier_initializer())
	bias = bias_variable([128])
	conv4 = tf.nn.relu(conv2d(conv3,kernel) + bias, name=scope.name)

pool2 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1],
                       strides=[1, 2, 2, 1], padding='SAME', name='pool2')

with tf.variable_scope('conv5') as scope:
	kernel = tf.get_variable("W5", shape=[3,3,128,128],
           initializer=tf.contrib.layers.xavier_initializer())
	bias = bias_variable([128])
	conv5 = tf.nn.relu(conv2d(pool2,kernel) + bias, name=scope.name)

pool3 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1],
                       strides=[1, 2, 2, 1], padding='SAME', name='pool3')

with tf.variable_scope('full1') as scope:
	kernel = tf.get_variable("W6", shape=[8*8*128,400],
           initializer=tf.contrib.layers.xavier_initializer())
	bias = bias_variable([400])
	poolflat = tf.reshape(pool3, [-1, 8*8*128])
	full1 = tf.nn.relu(tf.matmul(poolflat,kernel) + bias, name=scope.name)

with tf.variable_scope('full2') as scope:
	kernel = tf.get_variable("W7", shape=[400,400],
           initializer=tf.contrib.layers.xavier_initializer())
	bias = bias_variable([400])
	full2 = tf.nn.relu(tf.matmul(full1,kernel) + bias, name=scope.name)

with tf.variable_scope('full3') as scope:
	kernel = tf.get_variable("W8", shape=[400,200],
           initializer=tf.contrib.layers.xavier_initializer())
	bias = bias_variable([400])
	bias = bias_variable([200])
	full3 = tf.nn.relu(tf.matmul(full2,kernel) + bias, name=scope.name)

# softmax layer, CHECK VALUES ARE <1, this is massively effected by weight values
y_pred = tf.nn.softmax(full3) 

# loss calculation method is cross entropy, requires y as tf.float32
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred), reduction_indices=[1]))

# optimisation method for training
train_step = tf.train.GradientDescentOptimizer(0.9).minimize(loss)

# top5 = topk(y_pred, y, 5)
# top5_batch = tf.reduce_mean(top5)*100
# top1 = topk(y_pred, y, 1)
# top1_batch = tf.reduce_mean(top1)*100

with tf.Session() as sess:
  
  # initialize the variables
  sess.run(tf.initialize_all_variables())
  
  # initialize the queue threads to start to shovel data
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  # # CODE FOR TESTING use tf.shape() to get the shape of any tensor
  # a,b, = sess.run([full3, y_pred])
  # print a
  # print b

  # MAIN RUN CODE, LOSS SHOULD DECREASE, IF NOT CHECK WEIGHT INTIALISATION

  x_,y_pred_ = sess.run([x, y_pred])
  print tf.shape(x_)
  print tf.shape(y_pred)

  # stop our queue threads and properly close the session
  coord.request_stop()
  coord.join(threads)
  sess.close()