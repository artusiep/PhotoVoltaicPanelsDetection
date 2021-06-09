import glob
import os
from datetime import datetime

from configs.jet import TestJetConfig
from detector import detector
from detector.utils import save_img


def process(file_paths, config, silent, catalog, downscale=True):
    for index, file_path in enumerate(file_paths):
        new_file_path = file_path.replace(catalog, f'annotated/{catalog}/{current_time}')
        print(f"Annotation of img {file_path} started. Result will be saved to {new_file_path}. Run index {index}")
        try:
            annotated_img = detector.main(os.path.abspath(file_path), config=config, silent=silent,
                                          downscale_output=downscale)
        except Exception as e:
            print(f"Annotation of img {file_path} failed with {e}")
            continue
        save_img(annotated_img, new_file_path)


if __name__ == '__main__':
    current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    input_catalog = 'test'
    files = glob.glob(f"data/{input_catalog}/*.JPG")
    process(files, TestJetConfig(), False, input_catalog)
