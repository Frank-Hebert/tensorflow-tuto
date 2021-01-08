import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf


# Iintiatilization of Tensors

x = tf.constant(4.0, shape=(1,1))

y = tf.eye(3)


# Mathematical Operations

x = tf.constant([1,2,3])
y = tf.constant([9,8,7])
print(x+y)
# Indexing

# Reshaping