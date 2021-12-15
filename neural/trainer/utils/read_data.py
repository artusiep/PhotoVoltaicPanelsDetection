import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split

from trainer.utils.paths_definition import get_images_dir, get_panels_masks_dir

images_dir = get_images_dir()
panels_masks_dir = get_panels_masks_dir()


def load_npz_file(x_train, y_train, x_test, y_test):
    x_train_path = Path(os.path.dirname(os.path.abspath(__file__)), '../../data', x_train)
    y_train_path = Path(os.path.dirname(os.path.abspath(__file__)), '../../data', y_train)
    x_test_path = Path(os.path.dirname(os.path.abspath(__file__)), '../../data', x_test)
    y_test_path = Path(os.path.dirname(os.path.abspath(__file__)), '../../data', y_test)

    if x_train_path.exists() and y_train_path.exists() and x_test_path.exists() and y_test_path.exists():
        print("Using faster version with npz file loading")
        return np.load(x_train_path), \
               np.load(y_train_path), \
               np.load(x_test_path), \
               np.load(y_test_path)
    return None, None, None, None


def save_npz_file(x_train, y_train, x_test, y_test):
    x_train_path = Path(os.path.dirname(os.path.abspath(__file__)), '../../data/x_train')
    y_train_path = Path(os.path.dirname(os.path.abspath(__file__)), '../../data/y_train')
    x_test_path = Path(os.path.dirname(os.path.abspath(__file__)), '../../data/x_test')
    y_test_path = Path(os.path.dirname(os.path.abspath(__file__)), '../../data/y_test')
    np.save(f'{x_train_path}.npz', x_train)
    np.save(f'{y_train_path}.npz', y_train)
    np.save(f'{x_test_path}.npz', x_test)
    np.save(f'{y_test_path}.npz', y_test)


@lru_cache(maxsize=12)
def get_images_and_masks(expected_img_width, expected_img_height, should_resize, grayscale, test_size=0.2):
    x_train, y_train, x_test, y_test = load_npz_file('x_train.npz.npy', 'y_train.npz.npy', 'x_test.npz.npy',
                                                     'y_test.npz.npy')
    if x_train is not None:
        return x_train, y_train, x_test, y_test

    img_names = os.listdir(images_dir)
    number_of_images = len(img_names)

    if grayscale:
        data = np.zeros((number_of_images, expected_img_width, expected_img_height, 1), dtype=np.uint8)
    else:
        data = np.zeros((number_of_images, expected_img_width, expected_img_height, 3), dtype=np.uint8)
    labels = np.zeros((number_of_images, expected_img_width, expected_img_height, 1), dtype=np.bool)

    i = 0
    for img_index, img_name in enumerate(img_names):
        if i % 100 == 0:
            print(f"Loaded {i} files...")
        if grayscale:
            data[img_index] = image_to_gray(expected_img_height, expected_img_width, img_name, should_resize)
        else:
            data[img_index] = resize_and_get_image(expected_img_height, expected_img_width, img_name, should_resize)
        labels[img_index] = resize_and_get_mask(expected_img_height, expected_img_width, img_name, should_resize)
        i = i + 1

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=42)

    print(f"Loaded {i} files. Train data length {len(x_train)}, test data length {len(x_test)}")
    print('Loaded data for:', images_dir)
    save_npz_file(x_train, y_train, x_test, y_test)
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
