import base64
import json
import os

from labelme import LabelFile, __version__
from labelme.label_file import LabelFileError

from detector.labelers.abstract import RectangleLabeler
from detector.utils.utils import to_camel


class LabelMeLabeler(RectangleLabeler):
    extension = 'json'

    def __save(self, filename, shapes, image_path, image_height, image_width, image_data=None, tags=None):
        if image_data is not None:
            image_data = base64.b64encode(image_data).decode("utf-8")
            image_height, image_width = LabelFile._check_image_height_and_width(
                image_data, image_height, image_width
            )
        if tags is None:
            tags = {}
        data = dict(
            version=__version__,
            flags=None,
            shapes=shapes,
            imagePath=image_path,
            imageData=image_data,
            imageHeight=image_height,
            imageWidth=image_width,
        )
        for key, value in tags.items():
            assert key not in data
            data[to_camel(key)] = value
        try:
            with open(filename, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            raise LabelFileError(e)

    @staticmethod
    def __create_shape(rectangle):
        return {
            "label": "0",
            "points": rectangle.tolist(),
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }

    def label_image(self):
        self.labels_collector = self.rectangles
        self.labeled = True
        return self.rectangles

    def create_label_file(self):
        if not self.labeled:
            self.label_image()
        root, ext = os.path.splitext(self.label_path)
        file_name = f'{root}.{self.extension}'
        image_data = LabelFile().load_image_file(self.image_path)
        shapes = [self.__create_shape(rectangle) for rectangle in self.rectangles]
        self.__save(file_name, shapes, f'{root}.JPG', self.preprocessed_image.shape[0],
                    self.preprocessed_image.shape[1], image_data, self.tags)
        return file_name
