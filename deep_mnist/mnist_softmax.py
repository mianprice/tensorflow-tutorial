# Import MNIST data (55,000 points training, 10,000 points testing, 5,000 points validation)
# Each point includes as image and a label describing the contents of the image
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None,784])
y_ = tf.placeholder(tf.float32, shape=[None,10])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
sess.run(tf.global_variables_initializer())

y = tf.matmul(x,W) + b

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Run the training
for _ in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x:batch[0], y_:batch[1]})

# Evaluate the model
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

print accuracy.eval(feed_dict={x: mnist.test.images, y_:mnist.test.labels})
