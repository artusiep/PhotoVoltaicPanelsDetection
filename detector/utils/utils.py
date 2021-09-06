import logging
import os

import cv2
import numpy as np
from matplotlib import cm


def rectangle_annotated_photos(rectangles: list, base_image: np.ndarray):
    """Draws the rectangles contained in the first parameter onto the base image passed as second parameter.
    It also draw point in central of rectangle.
    :param rectangles: List of rectangles.
    :param base_image: Base image over which to render the rectangles.
    """
    mean_color = np.mean(base_image, axis=(0, 1))
    mask = np.zeros_like(base_image)
    if mean_color[0] == mean_color[1] == mean_color[2]:
        mean_color = np.array([255, 255, 0])
    opposite_color = np.array([255, 255, 255]) - mean_color
    opposite_color = (int(opposite_color[0]), int(opposite_color[1]), int(opposite_color[2]))
    for rectangle in rectangles:
        np_array = np.int32([rectangle])
        cv2.polylines(base_image, np_array, True, opposite_color, 5, cv2.LINE_AA)
        cv2.fillConvexPoly(mask, np_array, (255, 0, 0), cv2.LINE_4)
        x_central, y_central, width, height = calculate_centers(rectangle)
        cv2.circle(mask, (x_central, y_central), radius=10, color=(0, 0, 0), thickness=10)

    cv2.addWeighted(base_image, 1, mask, 0.5, 0, base_image)

    return base_image


def calculate_centers(rectangle):
    xmin, ymin = rectangle.min(axis=0)
    xmax, ymax = rectangle.max(axis=0)
    width = int(xmax) - int(xmin)
    height = int(ymax) - int(ymin)
    x_central = int(width / 2 + int(xmin))
    y_central = int(height / 2 + int(ymin))
    return x_central, y_central, width, height


def read_bgr_img(path):
    return cv2.imread(path)


def save_img(img, path):
    if not path.lower().endswith(('.jpg', '.jpeg')):
        path = path + '.jpg'
    catalogues = "/".join(path.split('/')[:-1])
    try:
        os.makedirs(catalogues)
    except FileExistsError:
        pass
    except Exception as e:
        logging.error(f"Failed to create path to save image {e}")
    result = cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    assert result


def auto_canny(image):
    sigma = 0.33
    v = np.ma.median(np.ma.masked_equal(image, 0))
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return lower, upper


def available_color_maps():
    return cm._cmap_registry.keys()


def get_color_map_by_name(name):
    return cm._cmap_registry[name]
