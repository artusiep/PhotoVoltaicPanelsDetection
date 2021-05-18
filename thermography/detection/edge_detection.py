import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

__all__ = ["EdgeDetectorParams", "EdgeDetector"]

from typing import Tuple


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

    def detect(self) -> None:
        """Detects the edges in the image passed to the constructor using the parameters passed to the constructor.
        """
        canny = cv2.Canny(image=self.input_image, threshold1=self.params.hysteresis_min_thresh,
                          threshold2=self.params.hysteresis_max_thresh, apertureSize=3)
        logging.debug("Canny edges computed")

        dilated = cv2.dilate(canny, self.params.kernel,
                             iterations=self.params.dilation_steps)
        logging.debug("Dilate canny edges with {} steps".format(self.params.dilation_steps))

        size = np.size(dilated)
        skel = np.zeros(dilated.shape, np.uint8)

        img = dilated
        done = False

        while not done:
            logging.debug("Eroding canny edges")
            eroded = cv2.erode(img, self.params.kernel)
            temp = cv2.dilate(eroded, self.params.kernel)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()

            zeros = size - cv2.countNonZero(img)
            if zeros == size:
                done = True

        self.edge_image = skel
