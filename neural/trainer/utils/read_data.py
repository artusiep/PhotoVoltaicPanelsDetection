import os
from functools import lru_cache

import numpy as np
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.transform import resize

from trainer.utils.paths_definition import get_images_dir, get_panels_masks_dir

images_dir = get_images_dir()
panels_masks_dir = get_panels_masks_dir()


@lru_cache(maxsize=12)
def get_images_and_masks(expected_img_width, expected_img_height, should_resize, grayscale):
    img_names = os.listdir(images_dir)
    number_of_images = len(img_names)

    if grayscale:
        x_train = np.zeros((number_of_images, expected_img_width, expected_img_height, 1), dtype=np.uint8)
        x_test = np.zeros((number_of_images, expected_img_width, expected_img_height, 1), dtype=np.uint8)
    else:
        x_train = np.zeros((number_of_images, expected_img_width, expected_img_height, 3), dtype=np.uint8)
        x_test = np.zeros((number_of_images, expected_img_width, expected_img_height, 3), dtype=np.uint8)
    y_train = np.zeros((number_of_images, expected_img_width, expected_img_height, 1), dtype=np.bool)
    y_test = np.zeros((number_of_images, expected_img_width, expected_img_height, 1), dtype=np.bool)

    i = 0
    for img_index, img_name in enumerate(img_names):
        print("")
        if i % 5 == 0:
            if grayscale:
                x_test[img_index] = image_to_gray(expected_img_height, expected_img_width, img_name, should_resize)
            else:
                x_test[img_index] = resize_and_get_image(expected_img_height, expected_img_width, img_name,
                                                         should_resize)
            y_test[img_index] = resize_and_get_mask(expected_img_height, expected_img_width, img_name, should_resize)
        else:
            if grayscale:
                x_train[img_index] = image_to_gray(expected_img_height, expected_img_width, img_name, should_resize)
            else:
                x_train[img_index] = resize_and_get_image(expected_img_height, expected_img_width, img_name,
                                                          should_resize)
            y_train[img_index] = resize_and_get_mask(expected_img_height, expected_img_width, img_name, should_resize)
        i = i + 1

    print('Loaded data for:', images_dir)
    return x_train, y_train, x_test, y_test


def image_to_gray(expected_img_height, expected_img_width, img_name, should_resize):
    image = resize_and_get_image(expected_img_height, expected_img_width, img_name, should_resize)
    gray_image = rgb2gray(image).astype(np.uint8).reshape((expected_img_width, expected_img_height, 1))
    return gray_image


def resize_and_get_mask(expected_img_height, expected_img_width, img_name, should_resize=False):
    img = imread(panels_masks_dir + img_name)
    if should_resize:
        img = resize(img, (expected_img_height, expected_img_width), mode='constant', preserve_range=True)
    grey_img = rgb2gray(img)
    grey_img = np.expand_dims(grey_img, axis=-1)
    return grey_img


def resize_and_get_image(expected_img_height, expected_img_width, img_name, should_resize=False):
    img = imread(images_dir + img_name)
    if should_resize:
        img = resize(img, (expected_img_height, expected_img_width), mode='constant', preserve_range=True)
    return img


