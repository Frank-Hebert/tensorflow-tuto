# SOURCE : https://www.youtube.com/watch?v=YrMy-BAqk8k
# TensorFlow Datasets
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import tensorflow_text

# ? EXAMPLE WITH MNIST

# (ds_train, ds_test), ds_info = tfds.load(
#     "mnist",
#     split=["train", "test"],
#     shuffle_files=True,
#     as_supervised=True,  # (img, label) False : dictionnary
#     with_info=True,
# )

# fig = tfds.show_examples(ds_train, ds_info, rows=4, cols=4)
# print(ds_info)


# def normalize_img(image, label):
#     return tf.cast(image, tf.float32) / 255.0, label


# AUTOTUNE = tf.data.experimental.AUTOTUNE
# print(f"AUTOTUNE : {AUTOTUNE}")

# BATCH_SIZE = 64
# ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
# ds_train = ds_train.cache()
# ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
# ds_train = ds_train.batch(BATCH_SIZE)
# ds_train = ds_train.prefetch(AUTOTUNE)

# ds_test = ds_test.map(normalize_img, num_parallel_calls=AUTOTUNE)
# ds_test = ds_test.batch(128)
# ds_test = ds_test.prefetch(AUTOTUNE)

# model = keras.Sequential(
#     [
#         keras.Input((28, 28, 1)),
#         layers.Conv2D(32, 3, activation="relu"),
#         layers.Flatten(),
#         layers.Dense(10),
#     ]
# )

# model.compile(
#     optimizer=keras.optimizers.Adam(lr=0.001),
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=["accuracy"],
# )

# model.fit(ds_train, epochs=5, verbose=1)
# model.evaluate(ds_test, verbose=1)

# ? EXAMPLE TEXT CLASSIFICATION

(ds_train, ds_test), ds_info = tfds.load(
    "imdb_reviews",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# print(ds_info)

# for text, label in ds_train:
#     print(text)
#     import sys

#     sys.exit()

tokenizer = tfds.deprecated.text.Tokenizer()


def build_vocabulary():
    vocabulary = set()
    for text, _ in ds_train:
        vocabulary.update(tokenizer.tokenize(text.numpy().lower()))
    return vocabulary


vocabulary = build_vocabulary()

encoder = tfds.deprecated.text.TokenTextEncoder(
    vocabulary, oov_token="<UNK>", lowercase=True, tokenizer=tokenizer
)


def my_encoding(text_tensor, label):
    return encoder.encode(text_tensor.numpy()), label


def encode_map(text, label):
    encoded_text, label = tf.py_function(
        my_encoding, inp=[text, label], Tout=(tf.int64, tf.int64)
    )
    encoded_text.set_shape([None])
    label.set_shape([])
    return encoded_text, label


AUTOTUNE = tf.data.experimental.AUTOTUNE
print(f"autotune : {AUTOTUNE}")
ds_train = ds_train.map(encode_map, num_parallel_calls=AUTOTUNE).cache()
ds_train = ds_train.shuffle(10000)
ds_train = ds_train.padded_batch(32, padded_shapes=([None], ()))

ds_train_ = ds_train.prefetch(AUTOTUNE)

ds_test = ds_test.map(encode_map)
ds_test = ds_test.padded_batch(32, padded_shapes=([None], ()))


model = keras.Sequential(
    [
        layers.Masking(mask_value=0),
        layers.Embedding(
            input_dim=len(vocabulary) + 2, output_dim=32
        ),  # padded_batch + <UNK>
        # BATCH_SIZE x 1000 -> BATCH_SIZE x 1000 x 32
        layers.GlobalAveragePooling1D(),
        # BATCH_SIZE x 32
        layers.Dense(64, "relu"),
        layers.Dense(1),  # less than 0 negative, greater  or equal than 0 positive
    ]
)

model.compile(
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.0001, clipnorm=1),
    metrics=["accuracy"],
)

model.fit(ds_train, epochs=10, verbose=1)
model.evaluate(ds_test)
