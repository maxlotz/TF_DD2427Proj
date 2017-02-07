import tensorflow as tf

def read_labeled_image_list(image_list_file):
	#reads csv file containing filenames and labels.
    with open(image_list_file, 'r') as f:
	    filenames = []
	    labels = []
	    for line in f:
	        filename, label = line[:-1].split(',')
	        filenames.append(filename)
	        labels.append(label)
	    return filenames, labels

def load_batches(image_filenames=None,
                 label_filenames=None,
                 shape=(64, 64, 3),
                 batch_size=100):

	min_after_dequeue =	1000
	capacity = min_after_dequeue + 3 * batch_size #As recommended on tf website

	with tf.name_scope('input'):
		image_queue, label_queue = tf.train.slice_input_producer(
			                                [image_filenames, label_filenames],
			                                shuffle=True)

		#File reader and decoder for images and labels goes here
		with tf.name_scope('image'): 

			with tf.name_scope('decode'): 
				file_content = tf.read_file(image_queue)
				image_data = tf.image.decode_jpeg(file_content, channels=3)
		
			#Any resizing or processing (eg. normalising) goes here
			with tf.name_scope('normalize'): 
				image_data = tf.cast(image_data, tf.float32)
				image_data = tf.image.per_image_standardization(image_data)
				image_data.set_shape(shape)

		
		with tf.name_scope('label'): 

			with tf.name_scope('decode'): 
				label_data = tf.string_to_number(label_queue, out_type=tf.int32)

		image_batch, label_batch = tf.train.shuffle_batch(
			                                [image_data, label_data],
			                                batch_size=batch_size,
											capacity=capacity,
	                                        min_after_dequeue=min_after_dequeue,
			                                #,num_threads=1
			                                )

	return image_batch, label_batch