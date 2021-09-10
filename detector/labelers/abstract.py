from abc import abstractmethod
from typing import List

import numpy as np

from detector.utils.pvpd_base_class import PVPDBaseClass


class RectangleLabeler(PVPDBaseClass):
    file_extension = None

    def __init__(self, rectangles: list, preprocessed_image: np.ndarray, label_path: str):
        self.rectangles = rectangles
        self.preprocessed_image = preprocessed_image
        self.label_path = label_path
        self.class_id = 0
        self.labels_collector = []
        self.labeled = False

    @abstractmethod
    def label_image(self):
        pass

    @abstractmethod
    def create_label_file(self):
        pass
