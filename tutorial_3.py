# SOURCE : https://www.youtube.com/channel/UCkzW5JSFwvKRjXABI-UTAkQ
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Neural Networks with Sequential and Functional API


# Loading and reshaping the dataset

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0

# Create model with Sequential API (convinient, not very flexible)

model = keras.Sequential(
    [
        keras.Input(shape=(28 * 28)),
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(10),
    ]
)


# 2nd method to create model with Sequential

model = keras.Sequential()
model.add(keras.Input(shape=(28 * 28)))
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(10))


# model = keras.Model(inputs=model.inputs,
#                     outputs=[model.layers[-2].output]) #-1 is dense(10), -2 is dense(256)...

# model = keras.Model(inputs=model.inputs,
#                     outputs=[model.output for layer in model.layers])
# features = model.predict(x_train)

# for feature in features:
#     print(feature.shape)


# sys.exit()

# Functional API (A bit more flexible)

inputs = keras.Input(shape=(28 * 28))
x = layers.Dense(512, activation="relu", name="first_layer")(inputs)
x = layers.Dense(256, activation="relu", name="2nd_layer")(x)
outputs = layers.Dense(10, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    # from_logits = False when softmax is specified, else set to true
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
