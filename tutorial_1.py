import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
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