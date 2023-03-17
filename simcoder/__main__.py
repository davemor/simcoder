import logging

import click
import numpy as np
from scipy.io import savemat
from tqdm import tqdm

from preprocess import preprocess_image
from util import turn_output_off

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

with turn_output_off():
    import tensorflow as tf

tf.compat.v1.enable_v2_behavior()

logging.basicConfig(level=logging.INFO)


def get_resnet50():
    model = tf.keras.applications.resnet50.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling="max",
    )
    return model


def get_simclr2(model_name):
    saved_model_path = (
        "gs://simclr-checkpoints-tf2/simclrv2/finetuned_100pct/r50_1x_sk0/saved_model/"
    )
    with turn_output_off():
        saved_model = tf.saved_model.load(saved_model_path)
    return saved_model


def save_array(arr, output_path, format):
    logging.info(f"Saving embeddings to {output_path}.")

    save_path = output_path.with_suffix(format)

    if format == 'csv':
        np.savetxt(save_path, arr, delimiter=',')
    elif format == 'npy':
        np.save(save_path, arr)
    elif format == 'mat':
        savemat(
            save_path,
            {"features": arr, "label": "simclr2_final_avg_pool_embeddings"},
        )
    else:
        logging.error("Can't save! Unknown format.")


@click.command()
@click.argument('input_dir', type=click.Path(exists=False))
@click.argument('output_path', type=click.Path(exists=False))
@click.option('--format', type=click.Choice(['csv', 'npy', 'mat']), default='csv', help='output format')
@click.option('--chunksize', type=int, default=10000, help='number of rows to process at a time')
def encode(input_dir, output_path, format, chunksize):
    logging.info("Welcome to Simcoder.")

    batch_size = 32  # should this be a parameter?

    # load the SimCLR2 model
    logging.info(f"Loading SimCLR2 model - this may take some time.")
    model = get_simclr2()

    # load in the image from the input_dir
    logging.info(f"Loading image dataset from {input_dir}")

    loaded_ds = tf.keras.utils.image_dataset_from_directory(
        f"{input_dir}",
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

    with turn_output_off():
        ds = loaded_ds.map(_preprocess).batch(batch_size)

    # generate the features from the final average pooling layer for each batch
    logging.info("Generating embeddings.")

    fs = []
    for x in tqdm(ds):
        with turn_output_off():
            features = model(x, trainable=False)["final_avg_pool"]
            fs.append(features.numpy())
    features = np.concatenate(fs, axis=0)

    if chunksize:
        chunks = np.array_split(x, np.arange(chunksize, len(features), chunksize))
        logging.info(len(chunks))
        for idx, chunk in enumerate(chunks):
            save_array(chunk, output_path / f"{idx}")
    else:
        save_array(features, output_path)


if __name__ == "__main__":
    encode()
