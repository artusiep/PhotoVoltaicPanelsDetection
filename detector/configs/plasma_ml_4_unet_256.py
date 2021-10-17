from math import floor

import numpy as np
from cv2 import cv2

from detector.configs.abstract import Config
from detector.detection import EdgeDetectorParams, SegmentDetectorParams, SegmentClustererParams, \
    ClusterCleaningParams, IntersectionDetectorParams, RectangleDetectorParams
from detector.detection.preprocessing_ml import PreprocessingMlParams
from trainer.utils.consts import UNET_6_LAYERS, UNET_4_LAYERS


class PlasmaMl4Unet256Config(Config):
    __preprocessing_image_scaling = 1
    __edge_image_scaling = 3
    preprocessing_params = PreprocessingMlParams(
        model_name=UNET_4_LAYERS,
        weight_path='detector/configs/models/1_training_unet_4_layers_2021-10-17T23:28:47_gray/cp.ckpt',
        gray=True,
        model_image_size=(256, 256),
        start_neurons=16,
        gaussian_blur=3,
        model_output_threshold=64,
        min_area=(120 * __preprocessing_image_scaling) ** 2
    )
    edge_detector_params = EdgeDetectorParams(
        image_scaling=__edge_image_scaling,
        hysteresis_min_thresh=35,
        hysteresis_max_thresh=45,
        kernel_size=(3, 3),
        kernel_shape=cv2.MORPH_RECT,
        dilation_steps=4
    )
    segment_detector_params = SegmentDetectorParams(
        d_rho=1,
        d_theta=np.pi / 180,
        min_num_votes=110,
        min_line_length=max(floor(10 * (__edge_image_scaling - 2)), 20),
        max_line_gap=20 * __edge_image_scaling,
        extension_pixels=35 * __edge_image_scaling
    )
    segment_clusterer_params = SegmentClustererParams(
        num_init=10,
        num_clusters=2,
        swipe_clusters=True,
        cluster_type="gmm",
        use_angles=True,
        use_centers=True
    )
    cluster_cleaning_params = ClusterCleaningParams(
        max_angle_variation_mean=np.pi / 180 * 20,
        max_merging_angle=np.pi / 180 * 40,
        max_endpoint_distance=15
    )
    intersection_detector_params = IntersectionDetectorParams(
        angle_threshold=np.pi / 180 * 25
    )
    rectangle_detector_params = RectangleDetectorParams(
        aspect_ratio=1.5,
        aspect_ratio_relative_deviation=0.35,
        min_area=20 * 40
    )
