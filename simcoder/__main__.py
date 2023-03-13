import logging
from pathlib import Path
import click

import os

import tensorflow as tf

tf.compat.v1.enable_v2_behavior()
from tensorflow import keras
import tensorflow_hub as hub

import numpy as np
import h5py

from preprocess import preprocess_image

# let's output the info
logging.basicConfig(level=logging.INFO)


def get_resnet50():
    model = keras.applications.resnet50.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),  # resize to this scale
        pooling="max",
    )
    return model


def get_simclr2():
    saved_model_path = (
        "gs://simclr-checkpoints-tf2/simclrv2/finetuned_100pct/r50_1x_sk0/saved_model/"
    )
    saved_model = tf.saved_model.load(saved_model_path)
    return saved_model


@click.command()
@click.argument("input_dir", type=click.Path(exists=False))
@click.argument("output_dir", type=click.Path(exists=False))
def encode(input_dir, output_dir):
    print(input_dir)
    print(output_dir)

    batch_size = 32  # should this be a parameter?

    # model = tf.keras.models.load_model('/input/r152_3x_sk1/saved_model')
    # model = tf.saved_model.load('/input/r152_3x_sk1/saved_model')

    model = get_simclr2()

    # load in the image from the input_dir
    loaded_ds = tf.keras.utils.image_dataset_from_directory(
        f"{input_dir}/images",
        labels=None,
        label_mode=None,
        class_names=None,
        color_mode="rgb",
        batch_size=None,  # batch after preprocessing
        # image_size=(256, 256),
        shuffle=False,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=False,
    )

    def _preprocess(x):
        x = preprocess_image(x, 224, 224, is_training=False, color_distort=False)
        return x

    ds = loaded_ds.map(_preprocess).batch(batch_size)

    # for img in ds.take(1):
    #    print(type(img))

    # features = model.predict(ds)

    for x in ds:
        logits = model(x, trainable=False)["logits_sup"]
        print(logits[0, :])

    # f = h5py.File(f"{output_dir}/resnet50_encodings.h5", "w")
    # f.create_dataset("resnet50_encodings", data=features, dtype=np.float32)


if __name__ == "__main__":
    encode()
