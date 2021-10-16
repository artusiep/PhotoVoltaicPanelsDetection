from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

import cv2
import numpy as np

from detector.utils.display import display_image_in_actual_size
from detector.utils.utils import read_bgr_img
from models.models_builders import get_model_builder
from trainer.utils.consts import UNET_6_LAYERS


@dataclass
class PreprocessingMlParams:
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
    model_name: str
    weight_path: str
    gray: bool = True
    model_image_size: Tuple[int, int] = (128, 128)
    start_neurons: int = 16
    gaussian_blur: int = 11

    @property
    def channels(self):
        return 1 if self.gray else 3

    def __hash__(self):
        return hash((self.model_name,
                     self.weight_path,
                     self.gray,
                     self.model_image_size,
                     self.start_neurons,
                     self.gaussian_blur))

    def __eq__(self, other):
        if not isinstance(other, PreprocessingMlParams):
            return False
        return (self.model_name == other.model_name and
                self.weight_path == other.weight_path and
                self.gray == other.gray and
                self.model_image_size == other.model_image_size and
                self.start_neurons == self.start_neurons and
                self.gaussian_blur == self.gaussian_blur)


class PreprocessorMl:
    """Class responsible for preprocessing an image frame."""

    def __init__(self, input_image: np.ndarray, params: PreprocessingMlParams):
        """Initializes the frame preprocessor with the input image and the preprocessor parameters.

        :param input_image: RGB or greyscale input image to be preprocessed.
        :param params: Preprocessing parameters.
        """
        self.input_image = input_image
        self.params = params
        self.preprocessed_image = None
        self.scaled_image_rgb = input_image
        self.scaled_image = None
        self.attention_image = None
        self.centroids = None
        self.mask = None
        self.model = self.prepare_model(params)

    @staticmethod
    @lru_cache
    def prepare_model(params: PreprocessingMlParams):
        model = get_model_builder(params.model_name)(params.model_image_size[0], params.model_image_size[1],
                                                     params.channels, params.start_neurons)
        model.load_weights(params.weight_path)
        return model

    def prepare_image(self):
        if self.params.gray:
            image = cv2.cv2.cvtColor(self.input_image, cv2.COLOR_BGR2GRAY)
        else:
            image = self.input_image
        image = cv2.resize(image, dsize=self.params.model_image_size)
        return image.reshape((1, *self.params.model_image_size))

    def to_original_size(self, result_image):
        # Maybe consider other interpolation
        result_image = (result_image.reshape(*self.params.model_image_size) * 255).astype('uint8')
        result_scaled_image = cv2.resize(result_image, dsize=self.input_image.shape[1::-1])
        ret, threshold_image = cv2.threshold(result_scaled_image, 32, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(threshold_image, cv2.MORPH_OPEN, kernel, iterations=2)
        self.mask = mask
        return self.mask

    def remove_reflections_2(self, input_image: np.ndarray) -> np.ndarray:
        """Returns an image with removed the most bright pixels (reflections), based on
        calculated image color histogram
        """
        hi_percentage = 0.02
        if hi_percentage == 0.0:
            return input_image
        grayed_image = input_image

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
        images_without_brightest_pixels = cv2.subtract(input_image, brightest_pixels_image)

        return images_without_brightest_pixels

    def remove_reflections(self, input_image: np.ndarray) -> np.ndarray:
        """Returns an image with removed the most bright pixels (reflections), based on
        calculated image color histogram
        """
        hi_percentage = 0.03
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

    def preprocess(self) -> None:
        """Preprocesses the :attr:`self.input_image` following this steps:

            1. The image is scaled using the :attr:`self.params.image_scaling` parameter.
            2. The image is rotated using the :attr:`self.params.image_rotation` parameter.
            3. Attention detection.

                a. If the image is RGB, the :attr:`self.params.red_threshold` parameter is used to determine the attention areas of the image.
                b. Otherwise the entire image is kept as attention.

        """
        display_image_in_actual_size(self.input_image)
        removed_reflections_img = self.remove_reflections(self.input_image)
        display_image_in_actual_size(removed_reflections_img)
        self.scaled_image = cv2.cvtColor(removed_reflections_img, cv2.COLOR_BGR2GRAY)

        image = self.prepare_image()
        display_image_in_actual_size(image[0])
        result_image = self.model.predict(image, use_multiprocessing=True)

        display_image_in_actual_size(result_image[0])

        result_scaled_mask = self.to_original_size(result_image)

        display_image_in_actual_size(result_scaled_mask)

        attention_mask = cv2.applyColorMap(result_scaled_mask, cv2.COLORMAP_WINTER)
        self.attention_image = cv2.addWeighted(cv2.cvtColor(self.scaled_image, cv2.COLOR_GRAY2BGR), 0.7, attention_mask,
                                               0.3, 0)

        thresholded_image = cv2.adaptiveThreshold(self.scaled_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY, 713, 1)

        # Perform dilation and erosion on the thresholded image to remove holes and small islands.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(cv2.bitwise_and(thresholded_image, result_scaled_mask), cv2.MORPH_OPEN, kernel, iterations=2)

        blurred = cv2.blur(self.scaled_image, (self.params.gaussian_blur, self.params.gaussian_blur))

        self.preprocessed_image = self.remove_reflections_2(blurred * opening)


if __name__ == '__main__':
    thermal_image = read_bgr_img(
        '/Users/artursiepietwoski/Developer/Private/PhotoVoltaicPanelsDetection/data/labelme/plasma-DJI_1_R (459).JPG')
    preprocesor = PreprocessorMl(thermal_image, PreprocessingMlParams(model_name=UNET_6_LAYERS,
                                                                      weight_path='/Users/artursiepietwoski/Developer/Private/PhotoVoltaicPanelsDetection/neural/trainer/training_result/1_training_unet_6_layers_2021-09-28T00:23:27_gray/cp.ckpt'))
    preprocesor.preprocess()
