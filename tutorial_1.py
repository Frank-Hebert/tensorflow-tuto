import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf


# Iintiatilization of Tensors

x = tf.constant(4.0, shape=(1,1))
print(x)
y = tf.eye(3)
print(y)

# Mathematical Operations

# Indexing

# Reshaping