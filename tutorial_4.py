# SOURCE : https://www.youtube.com/channel/UCkzW5JSFwvKRjXABI-UTAkQ
# Convolutional Neural Networks with Sequential and Functional API
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


model = keras.Sequential(
    [
        keras.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, (3, 3), padding="valid", activation="relu"),
        layers.MaxPool2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10),
    ]
)


def my_model():
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, (3, 3))(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, (5, 5), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(128, 3)(x)
    x = keras.activations.relu(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


model = my_model()

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=3e-4),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=64, epochs=2, verbose=1)
model.evaluate(x_test, y_test, batch_size=64, verbose=2)


# commentaire
# todo FAIT CA
# ! c'est poche ce que t'as fait
# ? bleu