import os

import cv2
import numpy as np


def rectangle_annotated_photos(rectangles: list, base_image: np.ndarray):
    """Draws the rectangles contained in the first parameter onto the base image passed as second parameter.

    This function displays the image using the third parameter as title.

    :param rectangles: List of rectangles.
    :param base_image: Base image over which to render the rectangles.
    :param windows_name: Title to give to the rendered image.
    """
    mean_color = np.mean(base_image, axis=(0, 1))
    mask = np.zeros_like(base_image)
    if mean_color[0] == mean_color[1] == mean_color[2]:
        mean_color = np.array([255, 255, 0])
    opposite_color = np.array([255, 255, 255]) - mean_color
    opposite_color = (int(opposite_color[0]), int(opposite_color[1]), int(opposite_color[2]))
    for rectangle in rectangles:
        xmin, ymin = rectangle.min(axis=0)
        xmax, ymax = rectangle.max(axis=0)
        width = int(xmax) - int(xmin)
        height = int(ymax) - int(ymin)

        xcentral = int(width / 2 + int(xmin))
        ycentral = int(height / 2 + int(ymin))

        cv2.polylines(base_image, np.int32([rectangle]), True, opposite_color, 5, cv2.LINE_AA)
        cv2.fillConvexPoly(mask, np.int32([rectangle]), (255, 0, 0), cv2.LINE_4)
        cv2.circle(mask, (xcentral, ycentral), radius=10, color=(0, 0, 0), thickness=10)

    cv2.addWeighted(base_image, 1, mask, 0.5, 0, base_image)

    return base_image


def read_bgr_img(path):
    return cv2.imread(path)

def save_img(img, path):
    catalogues = "/".join(path.split('/')[:-1])
    try:
        os.makedirs(catalogues)
    except FileExistsError:
        pass
    except Exception as e:
        print("Failed to create path to save image {e}")
    result = cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    assert result


def auto_canny(image):
    sigma = 0.33
    v = np.ma.median(np.ma.masked_equal(image, 0))
    print(v)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return lower, upper
