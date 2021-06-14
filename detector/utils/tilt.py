import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, sobel
from skimage.morphology import skeletonize, dilation
from skimage.transform import (hough_line, hough_line_peaks)
from skimage.transform import rotate, resize

from main import preprocess_frame_funcitonal
from thermography.detection import PreprocessingParams


def prepocess_image(image, show=False):
    preprocesed_img, _ = preprocess_frame_funcitonal(image, PreprocessingParams(
        gaussian_blur=9,
        image_scaling=6,
        image_rotation=0,
        red_threshold=90,
        min_area=(100 * (6)) ** 2
    ))
    if show:
        plt.imshow(preprocesed_img)
        plt.axis('off')
        plt.title('Preprocessed Image')
        plt.savefig('preprocessed_image.jpg')
        plt.show()
    return preprocesed_img


def binarizeImage(RGB_image):
    try:
        image = rgb2gray(RGB_image)
        threshold = threshold_otsu(image)
        bina_image = image < threshold
    except Exception as e:
        logging.error("This image is not RGB")
        return RGB_image
    return bina_image


def sceletonize(bina_image):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    dilated = cv2.dilate(bina_image, kernel,
                         iterations=4)

    size = np.size(dilated)
    skel = np.zeros(dilated.shape, np.uint8)

    img = dilated
    done = False
    while not done:
        eroded = cv2.erode(img, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True

    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title('Skeletons')
    plt.savefig('skeleton_image.jpg')

    return img


def findEdges(bina_image):
    image_edges = sobel(bina_image)
    plt.imshow(bina_image, cmap='gray')
    plt.axis('off')
    plt.title('Binary Image Edges')
    plt.savefig('binary_image.jpg')

    return image_edges


def findTiltAngle(image_edges):
    h, theta, d = hough_line(image_edges)
    accum, angles, dists = hough_line_peaks(h, theta, d)
    angle = np.rad2deg(mode(angles)[0][0])

    if (angle < 0):

        r_angle = angle + 90

    else:

        r_angle = angle - 90

    # Plot Image and Lines
    fig, ax = plt.subplots()

    ax.imshow(image_edges, cmap='gray')

    origin = np.array((0, image_edges.shape[1]))

    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        ax.plot(origin, (y0, y1), '-r')

    ax.set_xlim(origin)
    ax.set_ylim((image_edges.shape[0], 0))
    ax.set_axis_off()
    ax.set_title('Detected lines')

    plt.savefig('hough_lines.jpg')

    plt.show()

    return r_angle


def rotateImage(RGB_image, angle, save=False):
    fixed_image = rotate(RGB_image, angle, resize=True, mode='constant')

    if save:
        plt.imshow(fixed_image)
        plt.axis('off')
        plt.title('Fixed Image')
        plt.savefig('fixed_image.jpg')
        plt.show()

    return fixed_image


def resize_to_yolo(fixed_image, shape=(416, 416), save=False):
    resized_image = resize(fixed_image, shape, order=0, preserve_range=True, anti_aliasing=True)

    if save:
        plt.imshow(resized_image)
        plt.axis('off')
        plt.title('Fixed Resized Image')
        plt.savefig('fixed_resized_image.jpg')
        plt.show()
    return resized_image


def generalPipeline(img):
    image = io.imread(img)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    preprocesed_img = prepocess_image(image_bgr, True)
    bina_image = binarizeImage(preprocesed_img)

    image_edges = findEdges(bina_image)
    # sceletonize_img = sceletonize(image_edges)
    angle = findTiltAngle(image_edges)
    fixed = rotateImage(io.imread(img), angle, save=True)
    resize_to_yolo(fixed, save=True)


image_path = '../../data/thermal/TEMP_DJI_1_R (521).JPG'
generalPipeline(image_path)
