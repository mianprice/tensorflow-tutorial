##################
##################
# MODEL TRAINING #
##################
##################
# Import TensorFlow
import tensorflow as tf

# Rebuild Computational Graph
sess = tf.Session()
W = tf.Variable([.3],tf.float32)
b = tf.Variable([-.3],tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
init = tf.global_variables_initializer()
sess.run(init)
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

##############
# OPTIMIZERS #
##############
# Optimizers slowly change each variable in order to minimize loss function
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init)
print sess.run([W,b])
for i in range(1000):
	sess.run(train, {x:[1,2,3,4],y:[0,-1,-2,-3]})
print sess.run([W,b])
