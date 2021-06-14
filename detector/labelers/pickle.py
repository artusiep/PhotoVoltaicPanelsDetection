import os
import pickle

from detector.labelers.abstract import RectangleLabeler


class PickleLabeler(RectangleLabeler):
    extension = 'pickle'

    def __format_labels(self) -> dict:
        return {
            'image_path': self.image_path,
            'abs_image_path': os.path.abspath(self.image_path),
            'rectangles': self.labels_collector
        }

    def label_image(self):
        self.labels_collector = self.rectangles
        self.labeled = True
        return self.rectangles

    def create_label_file(self):
        if not self.labeled:
            self.label_image()
        root, ext = os.path.splitext(self.image_path)
        file_name = f'{root}.{self.extension}'
        with open(file_name, 'wb') as label_file:
            data = self.__format_labels()
            pickle.dump(data, label_file)
        return file_name
