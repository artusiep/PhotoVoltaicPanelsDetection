import argparse
from datetime import datetime

import Augmentor
import numpy as np
from PIL import Image
import glob
from natsort import natsorted
import os
import random
import matplotlib.pyplot as plt
import cv2
from pathlib import Path


debug = False
count = 200  # num of images to generate
batch = 20  # size of a single batch
begin = 0  # indicate the current index, use only if you are continuing a crashed generation
input_image_dir = '../data/input'
input_ground_truth_dir = '../data/ground_truth'
extension = '*.png'

current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

parser = argparse.ArgumentParser()
parser.add_argument('--augmentation_id', type=str, nargs='?', default=current_time)
args = parser.parse_args()
augmentation_id = args.augmentation_id

out_image_dir = f'data/{augmentation_id}/input/'
out_ground_truth_dir = f'data/{augmentation_id}/ground_truth/'


def build_augmentation_pipeline(images):
    augmentation_pipeline = Augmentor.DataPipeline(images)
    augmentation_pipeline.zoom(1, 1.1, 1.3)
    # p.zoom_random(1, .9)
    augmentation_pipeline.skew(1, .8)
    augmentation_pipeline.flip_random(1)
    # p.random_distortion(.3, 10, 10, 7)
    # p.random_color(1, .3, 1.2)
    augmentation_pipeline.random_contrast(1, .1, .3)
    augmentation_pipeline.random_brightness(1, 0.2, 1.2)
    augmentation_pipeline.shear(.5, 15, 15)
    # p.random_erasing(.75, 0.25)
    # p.rotate_random_90(1)
    # p.rotate(1, max_left_rotation=15, max_right_rotation=15)
    return augmentation_pipeline


def read_images_and_ground_truth():
    # import images and corresponding ground truth images in natural order
    images = natsorted(glob.glob(os.path.join(input_image_dir, extension)))

    ground_truth_path = os.path.join(input_ground_truth_dir, extension)
    ground_truth_images = natsorted(glob.glob(ground_truth_path))
    print(input_ground_truth_dir + ":" + str(len(ground_truth_images)))

    return [[np.asarray(Image.open(y)) for y in x] for x in list(zip(images, ground_truth_images))]


def generate_augmented_data(augmentation_pipeline):
    Path(out_image_dir).mkdir(parents=True, exist_ok=True)
    Path(out_ground_truth_dir).mkdir(parents=True, exist_ok=True)

    # begin generation
    for i in range(begin // batch, count // batch):
        print(str(i) + " st batch begin")
        augmented_images = augmentation_pipeline.sample(batch)
        print(str(i) + " st batch sampled")

        if debug:
            r_index = random.randint(0, len(augmented_images) - 1)
            f, axarr = plt.subplots(1, 2, figsize=(20, 15))
            axarr[0].imshow(augmented_images[r_index][0])
            axarr[1].imshow(augmented_images[r_index][1], cmap="gray")
            plt.show()

        for j in range(batch):
            # save image and its corresponding ground truth image
            file_name = str(i * batch + j) + "_" + augmentation_id + ".png"

            # image
            out_image_path = os.path.join(out_image_dir, file_name)
            cv2.imwrite(out_image_path, augmented_images[j][0])

            # ground truth
            out_ground_truth_path = os.path.join(out_ground_truth_dir, file_name)
            augmented_ground_truth = augmented_images[j][1]
            augmented_ground_truth = cv2.cvtColor(augmented_ground_truth, cv2.COLOR_BGR2GRAY)

            thresh = np.mean(augmented_ground_truth)
            _, augmented_ground_truth = cv2.threshold(augmented_ground_truth, thresh, 255, cv2.THRESH_BINARY)
            cv2.imwrite(out_ground_truth_path, augmented_ground_truth)

            print(str(i * batch + j) + " st image success")
            print(str(i) + " st batch finish")


images_and_ground_truth = read_images_and_ground_truth()
pipeline = build_augmentation_pipeline(images_and_ground_truth)
generate_augmented_data(pipeline)
