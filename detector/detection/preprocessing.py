import logging
from dataclasses import dataclass

import cv2
import numpy as np

from detector.utils.images import scale_image, rotate_image


@dataclass
class PreprocessingParams:
    """Parameters used by the :class:`.Preprocessor`.
    Initializes the preprocessing parameters to their default value.

    Attributes:
        :param gaussian_blur: Radius of the gaussian blur to apply to the input image.
        :param image_scaling: Scaling factor to apply to the input image.
        :param image_rotation: Angle expressed in radiants used to rotate the input image.
        :param red_threshold: Temperature threshold used to discard `cold` unimportant areas in the image.
        :param min_area: Minimal surface of retained `important` areas of the image. Warm regions whose surface
        is smaller than this threshold are discarded.
    """
    gaussian_blur: int = 11
    image_scaling: float = 8.0
    image_rotation: int = 0
    red_threshold: int = 120
    min_red_contour: int = 120
    min_area: int = 250 * 250
    pixel_outsider_percentage: int = 0.03


class Preprocessor:
    """Class responsible for preprocessing an image frame."""

    def __init__(self, input_image: np.ndarray, params: PreprocessingParams = PreprocessingParams()):
        """Initializes the frame preprocessor with the input image and the preprocessor parameters.

        :param input_image: RGB or greyscale input image to be preprocessed.
        :param params: Preprocessing parameters.
        """
        self.input_image = input_image
        self.params = params
        self.preprocessed_image = None
        self.scaled_image_rgb = None
        self.scaled_image = None
        self.attention_image = None
        self.mask = None

    @property
    def channels(self) -> int:
        """Returns the number of channels of the :attr:`self.input_image` image."""
        if len(self.input_image.shape) < 3:
            return 1
        elif len(self.input_image.shape) == 3:
            return 3
        else:
            raise ValueError("Input image has {} channels.".format(len(self.input_image.shape)))

    @property
    def gray_scale(self) -> bool:
        """Returns a boolean indicating whether :attr:`self.input_image` is a greyscale image
         (or an RGB image where all channels are identical).
         """
        if self.channels == 1:
            return True
        elif self.channels == 3:
            return (self.input_image[:, :, 0] == self.input_image[:, :, 1]).all() and \
                   (self.input_image[:, :, 0] == self.input_image[:, :, 2]).all()
        else:
            raise ValueError("Input image has {} channels.".format(len(self.input_image.shape)))

    def remove_reflections(self, input_image: np.ndarray) -> np.ndarray:
        """Returns an image with removed the most bright pixels (reflections), based on
        calculated image color histogram
        """
        hi_percentage = self.params.pixel_outsider_percentage
        if hi_percentage == 0.0:
            return input_image
        grayed_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        # we want the hi_percentage brightest pixels
        hist = cv2.calcHist([grayed_image], [0], None, [256], [0, 256]).flatten()

        total_count = grayed_image.shape[0] * grayed_image.shape[1]  # height * width
        target_count = hi_percentage * total_count  # bright pixels look for
        summed = 0
        for i in range(255, 0, -1):
            summed += int(hist[i])
            if target_count <= summed:
                hi_thresh = i
                break
        else:
            hi_thresh = 0

        brightest_pixels_image = cv2.threshold(grayed_image, hi_thresh, 0, cv2.THRESH_TOZERO)[1]
        inpainted_image = cv2.inpaint(input_image, brightest_pixels_image, 9, cv2.INPAINT_TELEA)

        return inpainted_image

    def remove_background(self, input_image):
        gaussian_blur = 3
        # blurred_image = cv2.blur(input_image, (gaussian_blur, gaussian_blur))
        # blurred_image = cv2.GaussianBlur(scaled_image, (3,3), 0)
        blurred_image = cv2.bilateralFilter(input_image, 5, 120, 120)

        grayed_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
        thresholded_image = cv2.adaptiveThreshold(grayed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY, 713, 1)

        # noise removal
        kernel = np.ones((7, 7), np.uint8)
        opening = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel, iterations=2)

        # find sure foreground area
        distance_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
        _, sure_foreground_image = cv2.threshold(distance_transform, 0.01 * distance_transform.max(), 255, 0)
        sure_foreground_image = np.uint8(sure_foreground_image)

        background_removed_image = cv2.bitwise_and(grayed_image, sure_foreground_image)

        # discard small contours/areas

        contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [contour for contour in contours if
                    cv2.contourArea(contour) > (self.params.min_area / self.params.image_scaling)]
        mask = np.zeros_like(background_removed_image)
        cv2.drawContours(mask, contours, -1, 255, cv2.FILLED)

        mask = cv2.dilate(mask, kernel, iterations=5)
        mask = cv2.blur(mask, (25, 25))
        mask = mask.astype(np.float) / 255.
        retained_contours_image = (background_removed_image * mask).astype(np.uint8)
        return retained_contours_image

    def preprocess(self) -> None:
        """Preprocesses the :attr:`self.input_image` following this steps:

            1. The image is scaled using the :attr:`self.params.image_scaling` parameter.
            2. The image is rotated using the :attr:`self.params.image_rotation` parameter.
            3. Attention detection.

                a. If the image is RGB, the :attr:`self.params.red_threshold` parameter is used to determine the attention areas of the image.
                b. Otherwise the entire image is kept as attention.

        """
        removed_reflections_img = self.remove_reflections(self.input_image)
        scaled_image = scale_image(removed_reflections_img, self.params.image_scaling)
        rotated_frame = rotate_image(scaled_image, self.params.image_rotation)

        if self.params.gaussian_blur > 0:
            rotated_frame = cv2.blur(rotated_frame, (self.params.gaussian_blur, self.params.gaussian_blur))

        if self.channels == 1:
            self.scaled_image = rotated_frame
            self.scaled_image_rgb = cv2.cvtColor(self.scaled_image, cv2.COLOR_GRAY2BGR)
            self.preprocessed_image = self.scaled_image.astype(np.uint8)
            mask = np.ones_like(self.scaled_image).astype(np.uint8) * 255
        else:
            if self.gray_scale:
                self.scaled_image_rgb = rotated_frame
                self.scaled_image = rotated_frame[:, :, 0]
                self.preprocessed_image = self.scaled_image.astype(np.uint8)
                mask = np.ones_like(self.scaled_image).astype(np.uint8) * 255
            else:
                self.scaled_image_rgb = rotated_frame
                self.scaled_image = cv2.cvtColor(self.scaled_image_rgb, cv2.COLOR_BGR2GRAY)

                thresholded_image = cv2.adaptiveThreshold(self.scaled_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                          cv2.THRESH_BINARY, 713, 1)
                # Perform dilation and erosion on the thresholded image to remove holes and small islands.
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

                opening = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel, iterations=2)
                distance_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
                _, sure_foreground_image = cv2.threshold(distance_transform, 0.01 * distance_transform.max(), 255, 0)
                sure_foreground_image = np.uint8(sure_foreground_image)
                background_removed_image = cv2.bitwise_and(self.scaled_image, sure_foreground_image)

                closing = cv2.morphologyEx(background_removed_image, cv2.MORPH_CLOSE, kernel)
                opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

                # contours is a python list of all detected contours which are represented as numpy arrays of (x,y) coords.
                contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours = [contour for contour in contours if cv2.contourArea(contour) > self.params.min_area]

                hulls = [cv2.convexHull(contour) for contour in contours]
                contours = hulls
                result_contours = []
                for contour in contours:
                    mask = np.zeros_like(self.scaled_image)
                    cv2.drawContours(mask, [contour], 0, (255), cv2.FILLED)
                    mask = mask.astype(np.float) / 255.
                    result = self.scaled_image_rgb[:, :, 2] * mask
                    red_channel = np.ma.masked_equal(result, 0)
                    v = np.ma.average(np.ma.masked_equal(red_channel, 0))
                    if v >= self.params.min_red_contour:
                        result_contours.append(contour)
                if len(contours) > len(result_contours):
                    logging.info("Reduced number of contours")

                mask = np.zeros_like(self.scaled_image)
                cv2.drawContours(mask, result_contours, -1, (255), cv2.FILLED)
                mask = cv2.dilate(mask, kernel, iterations=4)
                self.mask = mask
                mask = cv2.blur(mask, (25, 25))
                mask = mask.astype(np.float) / 255.
                self.preprocessed_image = (background_removed_image * mask).astype(np.uint8)
                mask = (mask * 255).astype(np.uint8)

        attention_mask = cv2.applyColorMap(mask, cv2.COLORMAP_WINTER)
        self.attention_image = np.empty_like(self.scaled_image_rgb)
        cv2.addWeighted(cv2.cvtColor(self.scaled_image, cv2.COLOR_GRAY2BGR), 0.7, attention_mask, 0.3, 0,
                        self.attention_image)
