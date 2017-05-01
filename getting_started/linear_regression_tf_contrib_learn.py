import tensorflow as tf
import numpy as np

# Declare list of features
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

# Estimator => Front End for training and evaluation (fitting and inference)
# Predefined types: {linear,logistic} {regression,classification}, and neual network classifiers and regressors
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

# Setup Dataset (Define number of batches of data as num_epchs and size of each batch)
x = np.array([1.,2.,3.,4.])
y = np.array([0.,-1.,-2.,-3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, batch_size=4, num_epochs=1000)

# Invoke training steps using `fit` method passing the dataset
estimator.fit(input_fn=input_fn, steps=1000)

# Evaluate the model
# Best Practice: Use separate data sets for validation and testing in order to avoid overfitting
print estimator.evaluate(input_fn=input_fn)


