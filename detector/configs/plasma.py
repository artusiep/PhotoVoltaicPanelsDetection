from math import floor

import numpy as np
from cv2 import cv2

from detector.configs.abstract import Config
from thermography.detection import PreprocessingParams, EdgeDetectorParams, SegmentDetectorParams, \
    SegmentClustererParams, ClusterCleaningParams, IntersectionDetectorParams, RectangleDetectorParams


class PlasmaConfig(Config):
    __preprocessing_image_scaling = 1
    __edge_image_scaling = 3
    preprocessing_params = PreprocessingParams(
        gaussian_blur=3,
        image_scaling=__preprocessing_image_scaling,
        image_rotation=0,
        red_threshold=120,
        min_area=(120 * __preprocessing_image_scaling) ** 2
    )
    edge_detector_params = EdgeDetectorParams(
        hysteresis_min_thresh=35,
        hysteresis_max_thresh=45,
        kernel_size=(3, 3),
        kernel_shape=cv2.MORPH_RECT,
        dilation_steps=4
    )
    segment_detector_params = SegmentDetectorParams(
        d_rho=1,
        d_theta=np.pi / 180,
        min_num_votes=130,
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
        max_endpoint_distance=10 * (__edge_image_scaling)
    )
    intersection_detector_params = IntersectionDetectorParams(
        angle_threshold=np.pi / 180 * 25
    )
    rectangle_detector_params = RectangleDetectorParams(
        aspect_ratio=1.5,
        aspect_ratio_relative_deviation=0.35,
        min_area=floor(20 * (__edge_image_scaling)) * floor(40 * (__edge_image_scaling))
    )
