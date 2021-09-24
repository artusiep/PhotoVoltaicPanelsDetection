import glob
from pathlib import Path

import PIL.Image
from labelme import LabelFile

PIL.Image.MAX_IMAGE_PIXELS = None


def fill_missing_labels(path):
    image_files = glob.glob(f'{path}/*.JPG')
    label_files_paths = glob.glob(f'{path}/*.json')
    label_files_names = [Path(label_file).stem for label_file in label_files_paths]

    for image_file_path in image_files:
        image_name = Path(image_file_path).stem

        if image_name not in label_files_names:
            print(f'Found missing label file for image {image_name}')
            image_data = LabelFile().load_image_file(image_file_path)
            LabelFile().save(f'{image_name}.json', [], f'{image_name}.JPG', 512, 640, image_data)


fill_missing_labels('data/thermal-panels')
