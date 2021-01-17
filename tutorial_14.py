# SOURCE : https://www.youtube.com/watch?v=WUzLJZCKNu4
# Callbacks with Keras and Writing Custom Callbacks
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds

(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


def normalize_img(image, label):
    """Normalize the image values between 0 and 1 and change the type as float32

    Args:
        image (array): (28,28,1) for mnist
        label (int): Values between 0 and 9 for mnist

    Returns:
        tuple: (image, label)
    """
    return tf.cast(image, tf.float32) / 255.0, label


AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 128

# Train dataset
ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

model = keras.Sequential(
    [
        keras.Input((28, 28, 1)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.Flatten(),
        layers.Dense(10),
    ]
)

saved_callback = keras.callbacks.ModelCheckpoint(
    filepath="checkpoint/",
    save_weights_only=True,
    monitor="accuracy",
    save_best_only=False,
)


def scheduler(epoch, lr):
    if epoch < 2:
        return lr
    else:
        return lr * 0.99


lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler, verbose=1)


class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get("accuracy") > 0.90:
            print("Accuracy over 90%, quitting training")
            self.model.stop_training = True


model.compile(
    optimizer=keras.optimizers.Adam(lr=3e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(
    ds_train,
    epochs=5,
    verbose=1,
    callbacks=[saved_callback, lr_scheduler, CustomCallback()],
)
