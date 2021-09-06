import os

import cv2
import numpy as np
from detector.labelers.abstract import RectangleLabeler
import tifffile as tiff


class TiffImageLabeler(RectangleLabeler):
    extension = 'tiff'

    @staticmethod
    def rectangle_annotated_photos(rectangles: list, base_image: np.ndarray):
        """Draws the rectangles contained in the first parameter onto the base image passed as second parameter.
        It also draw point in central of rectangle.
        :param rectangles: List of rectangles.
        :param base_image: Base image over which to render the rectangles.
        """
        mask = np.zeros_like(base_image)
        mask2 = np.zeros_like(base_image)
        result = np.zeros_like(base_image)

        for rectangle in rectangles:
            np_rectangle = np.int32([rectangle])
            cv2.fillConvexPoly(mask, np_rectangle, 255, cv2.LINE_4)
            cv2.polylines(mask2, np_rectangle, True, 255, 1, cv2.LINE_AA)

        cv2.addWeighted(mask, 1, mask2, -1, 0, result)

        return result

    def label_image(self):
        try:
            reference_image_shape = self.edge_images[0].shape
        except IndexError:
            return np.zeros(self.preprocessed_image.shape, np.uint8)
        blank_image = np.zeros(reference_image_shape, np.uint8)
        image = self.rectangle_annotated_photos(self.rectangles, blank_image)
        result = cv2.resize(image, self.preprocessed_image.shape[::-1])
        self.labeled = True
        return result

    def create_label_file(self):
        root, ext = os.path.splitext(self.label_path)
        file_name = f'{root}.{self.extension}'
        if not self.labeled:
            threshold_image = self.label_image()
            tiff.imsave(file_name, threshold_image)
        return file_name
