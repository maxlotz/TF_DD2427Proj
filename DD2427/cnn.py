import tensorflow as tf
import numpy as np

import input_pipeline
import tiny_image_network

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

validation_file = "validation_labels.csv"
training_file = "training_labels.csv"

BATCH_SIZE = 100
NUM_ITERATIONS = 100000
train_size = 100000
val_size = 10000
test_size = 10000

validation_images, validation_labels = input_pipeline.read_labeled_image_list(validation_file)
#validation_image_batch, validation_label_batch = input_pipeline(validation_images, validation_labels)

train_images, train_labels = input_pipeline.read_labeled_image_list(training_file)
train_image_batch, train_label_batch = input_pipeline.load_batches(train_images, train_labels, batch_size=BATCH_SIZE)

y_pred = tiny_image_network.inference(train_image_batch)
loss = tiny_image_network.loss(y_pred,train_label_batch)
train_step = tiny_image_network.training(loss)
summary_op = tf.merge_all_summaries()

with tf.Session() as sess:
  
  # initialize the variables
  sess.run(tf.initialize_all_variables())

  # enter into terminal: tensorboard --logdir=logs/ --port=6006 then navigate to http://0.0.0.0:6006/
  writer = tf.train.SummaryWriter("logs/", graph=tf.get_default_graph())
  
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  # RUN CODE HERE
  for i in range(NUM_ITERATIONS):
  	_, summary, loss_ = sess.run([train_step, summary_op, loss])

  	# tensorboard wont show the x axis value if its an int
  	writer.add_summary(summary, float(i))
  	print "iteration: " + str((i+1)) + "    loss: " + str(loss_)
  	print str(global_step)

  # stop our queue threads and properly close the session
  coord.request_stop()
  coord.join(threads)
  sess.close()