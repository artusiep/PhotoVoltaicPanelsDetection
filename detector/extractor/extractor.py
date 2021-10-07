import logging
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from cv2 import cv2

from detector.logger import init_logger
from detector.utils.display import display_image_in_actual_size
from detector.utils.utils import get_color_map_by_name

exiftool_path = 'exiftool'


class ThermalImageExtractorException(Exception):
    pass


class ThermalImageNotFound(ThermalImageExtractorException):
    def __init__(self, file_path):
        self.file_path = file_path


class ImageNotFound(ThermalImageExtractorException):
    def __init__(self, file_path):
        self.file_path = file_path


class ThermalImageExtractor:
    @classmethod
    def get_thermal_image_file_path(cls, file_path: str, color_maps: list, output_dir: str,
                                    display_photos: bool = False):
        if output_dir is None:
            output_dir = f'{tempfile.gettempdir()}/pvpd'
        try:
            return cls._extract_thermal_image(file_path, color_maps, output_dir, display_photos)[1]
        except ImageNotFound as e:
            logging.error(f"No image found under the path: {e.file_path}.")
        except ThermalImageNotFound as e:
            exiftool_report = subprocess.check_output([exiftool_path, e.file_path]).decode('utf-8')
            logging.error(f"No thermal data found in {e.file_path}. Exiftool report: \n{exiftool_report}")

    @staticmethod
    def _extract_thermal_image(file_path: str, color_maps: list,
                               output_dir: str = f'{tempfile.gettempdir()}/pvpd',
                               display_photos: bool = False):
        if not os.path.isfile(file_path):
            raise ImageNotFound(file_path)
        thermal_img_bytes = subprocess.check_output([exiftool_path, "-RawThermalImage", "-b", file_path],
                                                    stderr=subprocess.DEVNULL)
        if len(thermal_img_bytes) == 0:
            raise ThermalImageNotFound(file_path)
        img_np_arr = np.frombuffer(thermal_img_bytes, np.uint16)
        img_encode = cv2.imdecode(img_np_arr, cv2.IMREAD_UNCHANGED)

        thermal_normalized = (img_encode - np.amin(img_encode)) / (np.amax(img_encode) - np.amin(img_encode))

        filename = os.path.basename(file_path)
        images = []
        file_paths = []
        for color_map_name in color_maps:
            img_with_cm = ThermalImageExtractor.__apply_color_map(color_map_name, thermal_normalized)
            ready_to_save_img = ThermalImageExtractor.__format_image_to_save(img_with_cm)

            thermal_image_path = ThermalImageExtractor.__save_to_file(output_dir=output_dir,
                                                                      color_map_name=color_map_name,
                                                                      filename=filename,
                                                                      image_thermal=ready_to_save_img)
            file_paths.append(thermal_image_path)
            if display_photos:
                display_image_in_actual_size(ready_to_save_img)
            images.append(ready_to_save_img)
        return images, file_paths

    @staticmethod
    def __apply_color_map(color_map_name: str, img: np.ndarray):
        color_map = get_color_map_by_name(color_map_name)
        img_with_cm = color_map(img)
        return img_with_cm

    @staticmethod
    def __save_to_file(output_dir: str, color_map_name: str, filename: str, image_thermal: np.ndarray):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        file_path = f"{output_dir}/{color_map_name}-{filename}"
        cv2.imwrite(filename=file_path, img=image_thermal)
        logging.info(f"File extracted with '{color_map_name}' to {file_path}")
        return file_path

    @staticmethod
    def __format_image_to_save(image_thermal: np.ndarray) -> np.ndarray:
        image_thermal = image_thermal * 255
        image_integer = image_thermal.astype(np.uint8)
        return cv2.cvtColor(image_integer, cv2.COLOR_RGBA2BGR)


if __name__ == '__main__':
    init_logger()
    ThermalImageExtractor.get_thermal_image_file_path(file_path='../../data/raw/DJI_1_R (23).JPG',
                                                      color_maps=['jet', 'plasma'], output_dir='.', display_photos=True)
