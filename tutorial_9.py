# SOURCE : https://www.youtube.com/watch?v=cKMJDkWSDnY
# Custom Layers
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0


class Dense(layers.Layer):
    def __init__(self, units):
        super(Dense, self).__init__()
        self.units = units

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def build(self, input_shape):
        self.w = self.add_weight(
            name="w",
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )

        self.b = self.add_weight(
            name="b",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )


class MyRelu(layers.Layer):
    def __init__(self):
        super(MyRelu, self).__init__()

    def call(self, x):
        return tf.math.maximum(x, 0)


class MyModel(keras.Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()
        self.dense1 = Dense(64)
        self.dense2 = Dense(num_classes)
        self.relu = MyRelu()

        # self.dense1 = layers.Dense(64)
        # self.dense2 = layers.Dense(num_classes)

    def call(self, input_tensor):
        x = self.relu(self.dense1(input_tensor))
        return self.dense2(x)


model = MyModel()
model.compile(
    optimizer=keras.optimizers.Adam(lr=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=64, epochs=2, verbose=1)
model.evaluate(x_test, y_test, batch_size=64, verbose=1)