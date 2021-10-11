from abc import abstractmethod

import numpy as np

from detector.utils.pvpd_base_class import PVPDBaseClass


class RectangleLabeler(PVPDBaseClass):
    file_extension = None

    def __init__(self, rectangles: list, thermal_image: np.ndarray, image_path: str, preprocessed_image: np.ndarray,
                 label_path: str, tags: dict):
        self.rectangles = rectangles
        self.preprocessed_image = preprocessed_image
        self.thermal_image = thermal_image
        self.image_path = image_path
        self.label_path = label_path
        self.class_id = 0
        self.labels_collector = []
        self.labeled = False
        self.tags = tags

    @abstractmethod
    def label_image(self):
        pass

    @abstractmethod
    def create_label_file(self):
        pass
