import os
from functools import lru_cache

import cv2
import numpy as np
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.transform import resize
from math import ceil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

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
        if i % 100 == 0:
            print(f"Loaded {i} files...")
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
    img = cv2.cvtColor(cv2.imread(panels_masks_dir + img_name), cv2.COLOR_BGR2GRAY)
    if should_resize:
        img = cv2.resize(img, (expected_img_height, expected_img_width))
    grey_img = np.expand_dims(img, axis=-1)
    return grey_img


def resize_and_get_image(expected_img_height, expected_img_width, img_name, should_resize=False):
    img = cv2.cvtColor(cv2.imread(images_dir + img_name), cv2.COLOR_BGR2RGB)
    if should_resize:
        img = cv2.resize(img, (expected_img_height, expected_img_width))
    return img



def display_image_in_actual_size(base_image: np.ndarray, windows_name=None, rgb=False):
    if rgb:
        base_image = cv2.cvtColor(base_image, cv2.COLOR_RGBA2RGB)
    else:
        base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
    height, width, depth = base_image.shape

    figsize = width / 100, height / 100

    fig = plt.figure(figsize=figsize)
    window = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    window.axis('off')

    window.imshow(base_image)
    plt.title(windows_name)
    plt.show()