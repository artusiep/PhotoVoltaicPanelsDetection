from pathlib import Path

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
        cv2.polylines(base_image, np_array, True, opposite_color, 1, cv2.LINE_AA)
        cv2.fillConvexPoly(mask, np_array, (255, 0, 0), cv2.LINE_4)
        x_central, y_central, width, height = calculate_centers(rectangle)
        cv2.circle(mask, (x_central, y_central), radius=4, color=(0, 0, 0), thickness=2)

    cv2.addWeighted(base_image, 1, mask, 0.5, 0, base_image)

    return base_image


def iou_rectangle_annotated_photos(zip_rectangles: list, base_image: np.ndarray):
    mean_color = np.mean(base_image, axis=(0, 1))
    if mean_color[0] == mean_color[1] == mean_color[2]:
        mean_color = np.array([255, 255, 0])
    opposite_color = np.array([255, 255, 255]) - mean_color
    opposite_color = (int(opposite_color[0]), int(opposite_color[1]), int(opposite_color[2]))

    ground = np.zeros_like(base_image)
    pred = np.zeros_like(base_image)
    for ground_rect, pred_rect, iou in zip_rectangles:
        ground_array = np.int32([ground_rect])
        pred_array = np.int32([pred_rect])
        # cv2.polylines(base_image, ground_array, True, (64, 0, 0), 1, cv2.LINE_AA)
        cv2.fillConvexPoly(ground, ground_array, opposite_color, cv2.LINE_4)
        # cv2.polylines(base_image, pred_array, True, (0, 64, 0), 1, cv2.LINE_AA)
        cv2.fillConvexPoly(pred, pred_array, opposite_color, cv2.LINE_4)

        x_central, y_central, width, height = calculate_centers(np.array(ground_rect))
        if iou < 0.4:
            color = (0, 0, 255)
        elif iou < 0.6:
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
        cv2.putText(base_image, f'{iou:.2f}', (x_central - 15, y_central), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color)

    result = cv2.addWeighted(pred, 0.25, ground, 0.25, 0.25)
    result = cv2.addWeighted(base_image, 1, result, -0.5, 0)

    return result


def calculate_centers(rectangle):
    xmin, ymin = rectangle.min(axis=0)
    xmax, ymax = rectangle.max(axis=0)
    width = int(xmax) - int(xmin)
    height = int(ymax) - int(ymin)
    x_central = int(width / 2 + int(xmin))
    y_central = int(height / 2 + int(ymin))
    return x_central, y_central, width, height


def read_bgr_img(path: str):
    return cv2.imread(path)


def save_img(img: np.ndarray, path: str):
    if not path.lower().endswith(('.jpg', '.jpeg')):
        path = path + '.jpg'
    Path(path).parent.mkdir(parents=True, exist_ok=True)
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
