import tensorflow as tf
# Import MNIST data (55,000 points training, 10,000 points testing, 5,000 points validation)
# Each point includes as image and a label describing the contents of the image
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

# Model definition
x = tf.placeholder(tf.float32, [None,784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W) + b)

# Model Training
y_ = tf.placeholder(tf.float32, [None,10])
# Cross Entropy Function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# Apply optimization pattern
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# Launch Interactive Session and initialize variables
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# Run the training
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_xs, y_: batch_ys})

# Evaluate the model
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

print sess.run(accuracy,feed_dict={x: mnist.test.images, y_: mnist.test.labels})
