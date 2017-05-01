#####################
#####################
## TENSORFLOW CORE ##
#####################
#####################

# IMPORT TENSORFLOW
import tensorflow as tf

###############
# BUILD MODEL #
###############
# Build Computational Graph
# Constant Nodes
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
print(node1, node2)

# Evaluating nodes and graph
sess = tf.Session()
print(sess.run([node1,node2]))

# Combining nodes within graph with operations
node3 = tf.add(node1,node2)
print("node3: ",node3)
print("sess.run(node3): ", sess.run(node3))

# Placeholder nodes
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b
print sess.run(adder_node, {a: 3, b:4.5})
print sess.run(adder_node, {a: [1,3], b:[2,4]})

# Multiple operations in graph
add_and_triple = adder_node * 3
print sess.run(add_and_triple, {a: 3, b: 4.5})

# Variable nodes allow for trainable parameters to be added to the graph
# Variables are initialized with a type and initial value
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

# Constants initalize when tf.constant is called
# Variables must be initialized before use
init = tf.global_variables_initializer()
sess.run(init)

# Can evaluate model for multiple inputs at once
print sess.run(linear_model, {x:[1,2,3,4]})

##################
# EVALUATE MODEL #
##################
# Provide output placeholder, desired values, and loss function
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]})

# Variables can be reassigned to optimal parameters
fixW = tf.assign(W,[-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]})

