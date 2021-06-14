#!/usr/bin/env python3

import argparse
import glob
import logging
import os

from detector.configs.abstract import Config
from detector.detector import Detector
from detector.labelers.abstract import RectangleLabeler
from detector.logger import init_logger
from detector.utils.utils import save_img, available_color_maps, get_color_map_by_name
from thermography.utils.cli import dir_path, extant_file

RAW = 'raw'
THERMAL = 'thermal'
image_types = (THERMAL, RAW)


def process(file_paths, config, output_dir, labelers, silent, downscale):
    logging.info(f"Annotation of images {file_paths} started. Result will be saved to {output_dir}.")
    for index, file_path in enumerate(file_paths):
        output_path = f"{output_dir}/{os.path.basename(file_path)}"
        logging.info(f"Annotation of img {file_path} started. Result will be saved to {output_path}. "
                     f"Run index {index} of {len(file_paths)}")
        try:
            annotated_img = Detector.main(os.path.abspath(file_path),
                                          config=config,
                                          labelers=labelers,
                                          silent=silent,
                                          downscale_output=downscale)
        except Exception as e:
            logging.error(f"Annotation of img {file_path} failed with: {e}")
            continue

        save_img(annotated_img, output_path)
    logging.info(f"Image annotation. Finished")


helps = {
    'config': "Config Class name. Based on config PV panels are detected from image. "
              "User can create his own config and place in detector/configs directory. "
              "Class name must be unique",
    'labelers': "Space separated list labeler class name. Labeler serialize detected PV panels to a predefined format. "
                "Bear in mind that not all label formats are lossless"
                "User can create his own config and place in detector/labelers directory.",
    'type': f"Type of input image. When set to '{RAW}' extraction of FLIR image is done using "
            f"color map from 'color-map' parameter",
    'color-map': "Color map to which extracted of FLIR image is saved. Used for thresholding in preprocessing. "
                 "Used color aps from matplotlib library. "
                 "Available map: https://matplotlib.org/stable/tutorials/colors/colormaps.html",
    'output-dir': "Directory to which extracted images are saved",
    'preprocessed-size': "During preprocessing size of image can change. "
                         "If set, image will have preprocessed image size",
    'show-step-images': "If set step images of detection will be shown using matplotlib",
    "files": "Space separated list of paths to file detection (and if needed extraction) is done.",
    "input-dir": "Every file from this directory will be processed. Only files with .JPG and .jpg are processed"
}


def parse_arguments():
    init_logger()
    parser = argparse.ArgumentParser(prog='PhotoVoltaic Panels Detector', description='')
    parser.add_argument('-c', '--config', choices=Config.get_all_subclass(), required=True, help=helps['config'])
    parser.add_argument('-l', '--labelers', choices=RectangleLabeler.get_all_subclass(), nargs='+',
                        help=helps['labelers'])
    parser.add_argument('-t', '--type', choices=image_types, default=THERMAL, help=helps['type'])
    parser.add_argument('-cm', '--color-map', choices=available_color_maps(), default=get_color_map_by_name('jet'),
                        nargs='+', metavar='', help=helps['color-map'])
    parser.add_argument('-o', '--output-dir', type=dir_path, required=True, help=helps['output-dir'])
    parser.add_argument('--preprocessed-size', action='store_true', help=helps['preprocessed-size'])
    parser.add_argument('--show-step-images', action='store_true', help=helps['show-step-images'])

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-f', '--files', type=extant_file, nargs='+', help=helps['files'])
    group.add_argument('-d', '--input-dir', type=dir_path, help=helps['input-dir'])

    args = parser.parse_args()

    if args.input_dir:
        files = glob.glob(f"{args.input_dir}/*.JPG")
        files = files.extend(glob.glob(f"{args.input_dir}/*.jpg"))
    else:
        files = args.files

    # if args.type == RAW:
    #     FlirImageExtractor(palettes=args.color_map)

    process(file_paths=files,
            config=Config.get_subclass_by_name(args.config),
            silent=not args.show_step_images,
            labelers=[RectangleLabeler.get_subclass_by_name(labeler) for labeler in args.labelers],
            output_dir=args.output_dir,
            downscale=not args.preprocessed_size)
    return args


if __name__ == '__main__':
    parse_arguments()
