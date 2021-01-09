# SOURCE : https://www.youtube.com/channel/UCkzW5JSFwvKRjXABI-UTAkQ
# RNNs, GRUs, LSTMs and Bidirectionality

import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# model = keras.Sequential()
# model.add(keras.Input(shape=(None, 28)))
# model.add(layers.SimpleRNN(512, return_sequences=True, activation="relu"))
# model.add(layers.SimpleRNN(512, activation="relu"))
# model.add(layers.Dense(10))

# model = keras.Sequential()
# model.add(keras.Input(shape=(None, 28)))
# model.add(layers.GRU(512, return_sequences=True, activation="tanh"))
# model.add(layers.GRU(512, activation="tanh"))
# model.add(layers.Dense(10))

model = keras.Sequential()
model.add(keras.Input(shape=(None, 28)))
model.add(
    layers.Bidirectional(layers.LSTM(256, return_sequences=True, activation="tanh"))
)
model.add(layers.Bidirectional(layers.LSTM(256, activation="tanh")))
model.add(layers.Dense(10))

print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"],
)

model.fit(x=x_train, y=y_train, batch_size=64, epochs=2, verbose=1)
model.evaluate(x=x_test, y=y_test, batch_size=64, verbose=1)
