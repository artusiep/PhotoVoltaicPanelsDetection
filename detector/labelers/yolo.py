import os

from detector.labelers.abstract import RectangleLabeler
from detector.utils.utils import calculate_centers


class YoloRectangleLabeler(RectangleLabeler):
    extension = 'txt'

    def __get_yolo_string_line(self, x_central_normalized, y_central_normalized, width_normalized, height_normalized):
        yolo_string = f"{self.class_id} {x_central_normalized:.6f} {y_central_normalized:.6f} {width_normalized:.6f} {height_normalized:.6f}"
        return yolo_string

    def __get_yolo_labels(self):
        yolo_lines = [self.__get_yolo_string_line(label['x'], label['y'], label['width'], label['height']) for label in
                      self.labels_collector]
        return '\n'.join(yolo_lines)

    def label_image(self):
        for rectangle in self.rectangles:
            x_central, y_central, width, height = calculate_centers(rectangle)
            x_central_normalized = x_central / self.preprocessed_image.shape[1]
            width_normalized = width / self.preprocessed_image.shape[1]
            y_central_normalized = y_central / self.preprocessed_image.shape[0]
            height_normalized = height / self.preprocessed_image.shape[0]
            self.labels_collector.append({
                'x': x_central_normalized,
                'y': y_central_normalized,
                'width': width_normalized,
                'height': height_normalized
            })
        self.labeled = True
        return self.labels_collector

    def create_label_file(self):
        if not self.labeled:
            self.label_image()
        root, ext = os.path.splitext(self.label_path)
        file_name = f'{root}.{self.extension}'
        with open(file_name, "w+") as label_file:
            labels_formatted = self.__get_yolo_labels()
            label_file.write(labels_formatted)
        return file_name
