import logging
from pathlib import Path
import click

import os

import tensorflow as tf
from tensorflow import keras

import numpy as np
from scipy.io import savemat
from tqdm import tqdm

from preprocess import preprocess_image

# let's output the info
logging.basicConfig(level=logging.INFO)

# configure Tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.enable_v2_behavior()


def get_resnet50():
    model = keras.applications.resnet50.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling="max",
    )
    return model


def get_simclr2():
    saved_model_path = (
        "gs://simclr-checkpoints-tf2/simclrv2/finetuned_100pct/r152_3x_sk1/saved_model/"
    )
    saved_model = tf.saved_model.load(saved_model_path)
    return saved_model


@click.command()
@click.argument("input_dir", type=click.Path(exists=False))
@click.argument("output_dir", type=click.Path(exists=False))
def encode(input_dir, output_dir):
    logging.info("Starting encoding ...")

    batch_size = 32  # should this be a parameter?

    # load the SimCLR2 model
    logging.info(f"Load SimCLR2 model.")
    model = get_simclr2()

    # load in the image from the input_dir
    logging.info(f"Load the dataset from {input_dir}/images.")
    loaded_ds = tf.keras.utils.image_dataset_from_directory(
        f"{input_dir}/images",
        labels=None,
        label_mode=None,
        class_names=None,
        color_mode="rgb",
        batch_size=None,
        shuffle=False,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=False,
    )

    # apply the preprocessing function to the dataset
    logging.info("Running preprocessor.")

    def _preprocess(x):
        x = preprocess_image(x, 224, 224, is_training=False, color_distort=False)
        return x

    ds = loaded_ds.map(_preprocess).batch(batch_size)

    # generate the features from the final average pooling layer for each batch
    logging.info("Generating embeddings.")
    fs = []
    for x in tqdm(ds):
        features = model(x, trainable=False)["final_avg_pool"]
        fs.append(features.numpy())
    all_features = np.concatenate(fs, axis=0)

    # write it out in the matlab compatible format
    savemat(
        f"{output_dir}/embeddings.mat",
        {"features": all_features, "label": "embeddings"},
    )


if __name__ == "__main__":
    encode()
