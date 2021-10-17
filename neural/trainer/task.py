from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from trainer import model_builder
from trainer.utils.callbacks import get_callbacks
from trainer.utils.read_data import get_images_and_masks
from trainer.utils.utils import get_save_model_path, get_final_save_model_path
from utils.consts import UNET_4_LAYERS, UNET_6_LAYERS, UNET_DENSE_4_LAYERS, UNET_PLUS_PLUS_4_LAYERS, RES_NET_152


def get_args():
    """Argument parser.

    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='local or GCS location for writing checkpoints and exporting '
             'models')
    parser.add_argument(
        '--to-grayscale',
        type=bool,
        default=True,
        help='number of the channels of the images')
    parser.add_argument(
        '--img-size',
        type=int,
        default=128,
        help='size of the images')
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='number of times to go through the data')
    parser.add_argument(
        '--batch-size',
        default=32,
        type=int,
        help='number of records to read during each training step')
    parser.add_argument(
        '--runs-no',
        default=5,
        type=int,
        help='number of records to read during each training step')
    parser.add_argument(
        '--start-neurons',
        default=16,
        type=int,
        help='number of start neurons')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')
    parser.add_argument(
        '--models',
        choices=[UNET_4_LAYERS, UNET_6_LAYERS, UNET_DENSE_4_LAYERS, UNET_PLUS_PLUS_4_LAYERS, RES_NET_152],
        nargs='+',
        required=True
    )
    args, _ = parser.parse_known_args()
    return args


def train_and_evaluate_models(args):
    x_train, y_train, x_test, y_test = get_images_and_masks(args.img_size, args.img_size, True, args.to_grayscale)
    for run_id in range(1, args.runs_no + 1):
        for model_name in args.models:
            train_and_evaluate(run_id, model_name, x_train, y_train, x_test, y_test, args)


def train_and_evaluate(run_id, model_name, x_train, y_train, x_test, y_test, args):
    model = model_builder.build(model_name, args.img_size, 1 if args.to_grayscale else 3, args.start_neurons)
    model_save_path = get_save_model_path(run_id, model_name, args.to_grayscale)
    callbacks = get_callbacks(model_save_path)

    model.fit(x=x_train,
              y=y_train,
              validation_split=0.2,
              batch_size=args.batch_size,
              epochs=args.epochs,
              callbacks=callbacks)

    model.summary()

    print("[LOG] Evaluating model")
    loss, acc = model.evaluate(x_test, y_test, verbose=1)
    print("[LOG] Current model accuracy: {:5.2f}%".format(100 * acc))

    pred_test = model.predict(x_test, verbose=1)

    model_export_path = os.path.join(args.job_dir, model_save_path)
    tf.keras.models.save_model(model, model_export_path)

    pred_test = (pred_test > 0.6).astype(np.uint8)
    f1score = f1_score(y_test.flatten().flatten(), pred_test.flatten().flatten())
    print(f'[LOG] F1 score: {f1score}')

    final_model_path = get_final_save_model_path(run_id, model_name, f1score, args.to_grayscale)
    model_export_f1_score_path = os.path.join(args.job_dir, final_model_path)
    tf.keras.models.save_model(model, model_export_f1_score_path)


if __name__ == '__main__':
    args = get_args()
    tf.compat.v1.logging.set_verbosity(args.verbosity)
    train_and_evaluate_models(args)
