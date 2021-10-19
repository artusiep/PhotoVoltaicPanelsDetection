import logging
from dataclasses import dataclass, field
from typing import Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import morphology

from detector.utils.images import scale_image


from detector.utils.display import display_image_in_actual_size

@dataclass
class EdgeDetectorParams:
    """Parameters used by the :class:`.EdgeDetector`.

    Initializes the edge detector parameters to their default value.

    Attributes:
        :param hysteresis_min_thresh: Canny candidate edges whose weight is smaller than this threshold are ignored.
        :param hysteresis_max_thresh: Canny candidate edges whose weights is larger than this threshold are considered as edges without hysteresis.
        :param kernel_size: Kernel shape to use when performing dilation and erosion of binary edge image.
        :param kernel_shape: Kernel shape to use when performing dilation and erosion of binary edge image. One of MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE
        :param dilation_steps: Number of dilation steps to take before skeletonization.
    """
    image_scaling: int = 3
    sigma: int = 0.33
    hysteresis_min_thresh: int = 50
    hysteresis_max_thresh: int = 100

    kernel_size: Tuple[int, int] = (3, 3)
    kernel_shape: int = cv2.MORPH_CROSS
    kernel: None = field(init=False, default_factory=tuple)

    dilation_steps: int = 4

    def __post_init__(self):
        self.kernel = cv2.getStructuringElement(self.kernel_shape, self.kernel_size)


class EdgeDetector:
    """Class responsible for detecting edges in greyscale images.
    The approach taken to detect edges in the input greyscale image is the following:

        1. Perform a canny edge detection on the input image.
        2. Dilate the resulting binary image for a parametrized number of steps in order to connect short edges and smooth out the edge shape.
        3. Erode the dilated edge image in order to obtain a 1-pixel wide skeletonization of the edges in the input image.
    """

    def __init__(self, input_image: np.ndarray, params: EdgeDetectorParams):
        """
        :param input_image: Input greyscale image where edges must be detected.
        :param params: Parameters used for edge detection.
        """
        self.input_image = input_image
        self.params = params

        self.edge_image = None

    @staticmethod
    def apply_brightness_contrast(input_img, brightness=0, contrast=0):
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow) / 255
            gamma_b = shadow

            buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
        else:
            buf = input_img.copy()

        if contrast != 0:
            f = 131 * (contrast + 127) / (127 * (131 - contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)

            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
        return buf

    def strengthen_edges(self, input_image):
        # blurring (probably could be improved, reduced to one)
        blurred_image = cv2.bilateralFilter(input_image, 3, 200, 200)
        blurred_image = cv2.GaussianBlur(blurred_image, (7, 7), 0)

        # adaptive thresholding
        adaptive_threshold = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY, 11, 2)
        denoised_image = cv2.fastNlMeansDenoising(adaptive_threshold, 9, 21, 7)  # 30, 7, 25
        # contrasting
        brightness = 0
        contrast = 64  # contrast_const

        contrasted_image = self.apply_brightness_contrast(denoised_image, brightness, contrast)
        return contrasted_image

    def detect(self) -> None:
        """Detects the edges in the image passed to the constructor using the parameters passed to the constructor.
        """

        strengthen_edges_img = self.strengthen_edges(self.input_image)
        _, global_thresh = cv2.threshold(self.input_image, 0, 255, cv2.THRESH_BINARY)
        strengthened_edges_image = cv2.bitwise_and(strengthen_edges_img, global_thresh)

        strengthened_edges_image = cv2.fastNlMeansDenoising(strengthened_edges_image, 30, 40, 10)

        scaled_strengthened_edges_image = scale_image(strengthened_edges_image, self.params.image_scaling)

        blurred_image = cv2.bilateralFilter(scaled_strengthened_edges_image, 9, 200, 200)
        canny = cv2.Canny(image=blurred_image, threshold1=self.params.hysteresis_min_thresh,
                          threshold2=self.params.hysteresis_max_thresh, apertureSize=3)
        logging.debug("Canny edges computed")

        dilated = cv2.dilate(canny, (3, 3), iterations=self.params.dilation_steps)
        logging.debug("Dilate canny edges with {} steps".format(self.params.dilation_steps))

        size = np.size(dilated)
        skel = np.zeros(dilated.shape, np.uint8)

        img = dilated
        done = False

        kernel_size = (7, 7)
        kernel_shape = cv2.MORPH_CROSS
        kernel = cv2.getStructuringElement(kernel_shape, kernel_size)
        while not done:
            logging.debug("Eroding canny edges")
            eroded = cv2.erode(img, kernel)
            temp = cv2.dilate(eroded, kernel)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()

            zeros = size - cv2.countNonZero(img)
            if zeros == size:
                done = True

        self.edge_image = skel
