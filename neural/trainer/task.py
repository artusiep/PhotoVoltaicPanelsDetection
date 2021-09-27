# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Trains a Keras model to predict income bracket from other Census data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import numpy as np
import tensorflow as tf

from utils.callbacks import get_callbacks
from utils.consts import UNET_4_LAYERS, UNET_6_LAYERS, UNET_DENSE_4_LAYERS, UNET_PLUS_PLUS_4_LAYERS
from utils.utils import get_save_model_path
from . import model_builder
from . import util

from sklearn.metrics import f1_score


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
        '--channels-number',
        type=int,
        default=1,
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
        help='number of times to go through the data, default=10')
    parser.add_argument(
        '--batch-size',
        default=32,
        type=int,
        help='number of records to read during each training step, default=32')
    parser.add_argument(
        '--start-neurons',
        default=16,
        type=int,
        help='number of start neurons, default=16')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')
    args, _ = parser.parse_known_args()
    return args


model_names = [UNET_4_LAYERS, UNET_6_LAYERS, UNET_DENSE_4_LAYERS, UNET_PLUS_PLUS_4_LAYERS]


def train_and_evaluate_models(args):
    x_train, y_train = util.load_data('dataset_name')
    for model_name in model_names:
        train_and_evaluate(model_name, x_train, y_train, args)


def train_and_evaluate(model_name, x_train, y_train, x_test, y_test, args):
    model = model_builder.build(model_name, args.img_size, args.channels_number, args.start_neurons)
    model_save_path = get_save_model_path(model_name)
    callbacks = get_callbacks(model_save_path)

    model.add_metric()

    model.fit(x=x_train,
              y=y_train,
              validation_split=0.2,
              batch_size=args.batch_size,
              epochs=args.epochs,
              callbacks=callbacks)

    model.summary()

    loss, acc = model.evaluate(x_test, y_test, verbose=1)

    print("[LOG] Did evaluate model")
    print("[LOG] Current model accuracy: {:5.2f}%".format(100 * acc))

    print("[LOG] Start model predictions with test data")
    pred_test = model.predict(x_test, verbose=1)

    model_export_path = os.path.join(args.job_dir, model_save_path)
    tf.keras.models.save_model(model, model_export_path)

    print('Model exported to: {}'.format(model_export_path))

    pred_test_t = (pred_test > 0.6).astype(np.uint8)
    f1score = f1_score(x_test.flatten().flatten(), pred_test_t.flatten().flatten())

    model_export_f1_score_path = os.path.join(args.job_dir, "final" + model_save_path + "_" + f1score)
    tf.keras.models.save_model(model, model_export_f1_score_path)

if __name__ == '__main__':
    args = get_args()
    tf.compat.v1.logging.set_verbosity(args.verbosity)
    train_and_evaluate_models(args)
