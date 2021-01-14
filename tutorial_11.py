# SOURCE : https://www.youtube.com/watch?v=WJZoywOG1cs
# Transfer Learning, Fine Tuning and TensorFlow Hub

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow_hub as hub

# ? WITH PRETRAINED MODEL

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
# x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0

# model = keras.models.load_model("pretrained_model/")
# model.trainable = False

# for layer in model.layers[0:1]:
#     assert layer.trainable == False
# # print(model.summary())

# base_inputs = model.layers[0].input
# base_outputs = model.layers[-2].output
# final_outputs = layers.Dense(10, name="final")(base_outputs)

# new_model = keras.Model(inputs=base_inputs, outputs=final_outputs)
# # print(new_model.summary())

# model.compile(
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     optimizer=keras.optimizers.Adam(lr=0.001),
#     metrics=["accuracy"],
# )

# # model = keras.models.load_model("complete_saved_model/")

# model.fit(x_train, y_train, batch_size=64, epochs=2, verbose=1)
# model.evaluate(x_test, y_test, batch_size=64, verbose=1)


# ? WITH PRETRAINED KERAS MODEL

# x = tf.random.normal(shape=(5, 299, 299, 3))
# y = tf.constant([0, 1, 2, 3, 4])

# model = keras.applications.InceptionV3(include_top=True)
# # print(model.summary())

# base_inputs = model.layers[0].input
# base_outputs = model.layers[-2].output
# final_outputs = layers.Dense(5)(base_outputs)

# new_model = keras.Model(inputs=base_inputs, outputs=final_outputs)

# new_model.compile(
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     optimizer=keras.optimizers.Adam(lr=0.001),
#     metrics=["accuracy"],
# )

# # model = keras.models.load_model("complete_saved_model/")

# new_model.fit(x, y, batch_size=64, epochs=2, verbose=1)

# ? WITH PRETRAINED HUB MODEL https://www.tensorflow.org/hub

x = tf.random.normal(shape=(5, 299, 299, 3))
y = tf.constant([0, 1, 2, 3, 4])

url = "https://tfhub.dev/google/imagenet/inception_v2/feature_vector/4"

base_model = hub.KerasLayer(url, input_shape=(299, 299, 3))
base_model.trainable = False

new_model = keras.Sequential(
    [
        base_model,
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(5),
    ]
)


new_model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"],
)

new_model.fit(x, y, batch_size=64, epochs=10, verbose=1)