import os

import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.transform import resize

from models_builders import get_model_builder
from utils.read_data import get_images_and_masks
from utils.utils import load_variables

# 'training_result/training_unet_4_layers_21092021_0052/cp.ckpt' 512
weights_path = 'training_result/training_unet_4_layers_22092021_2029/cp.ckpt'

model_name, channel_numbers, img_size, epochs, batch_size, starts_neuron, trained_model_weights_path = load_variables()

test_images, test_labels = get_images_and_masks(img_size, img_size, True)

print('[LOG] Building model')
model = get_model_builder(model_name)(img_size, img_size, channel_numbers, starts_neuron)

print('[LOG] Pre loading weights')


def resize_and_get_image(expected_img_height, expected_img_width, img_name, should_resize=False):
    images_dir = '/Users/artursiepietwoski/Developer/Private/PhotoVoltaicPanelsDetection/experiments/data/thermal2/'
    img = imread(images_dir + img_name)
    if should_resize:
        img = resize(img, (expected_img_height, expected_img_width), mode='constant', preserve_range=True)
    return img


def get_images(expected_img_width, expected_img_height, should_resize=False, to_gray_scale=True):
    images_dir = '/Users/artursiepietwoski/Developer/Private/PhotoVoltaicPanelsDetection/experiments/data/thermal2'
    img_names = os.listdir(images_dir)
    number_of_images = len(img_names)

    if to_gray_scale:
        x_train = np.zeros((number_of_images, expected_img_width, expected_img_height, 1), dtype=np.uint8)
    else:
        x_train = np.zeros((number_of_images, expected_img_width, expected_img_height, 3), dtype=np.uint8)

    for img_index, img_name in enumerate(img_names):
        image = resize_and_get_image(expected_img_height, expected_img_width, img_name, should_resize)
        gray_image = rgb2gray(image).astype(np.uint8).reshape((expected_img_width, expected_img_height, 1))
        x_train[img_index] = gray_image

    print('Loaded data for:', images_dir)
    return x_train


def predict(images, output_path):
    model.load_weights(weights_path)

    print('[LOG] Pre loading weights')

    pred_test = model.predict(images, verbose=1)

    for id, pred_mask in enumerate(zip(pred_test, test_images)):
        Image.fromarray(np.squeeze((pred_mask[0] * 255).astype(np.uint8), axis=2)).save(
            f'data/{output_path}/{id}-result.png')
        Image.fromarray(np.squeeze((pred_mask[1]).astype(np.uint8), axis=2)).save(f'data/{output_path}/{id}-source.png')


test_images = get_images(img_size, img_size, True)

predict(test_images, 'test-512')
