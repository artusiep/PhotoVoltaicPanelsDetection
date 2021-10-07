from math import floor

import numpy as np
from cv2 import cv2

from detector.configs.abstract import Config
from thermography.detection import PreprocessingParams, EdgeDetectorParams, SegmentDetectorParams, \
    SegmentClustererParams, ClusterCleaningParams, IntersectionDetectorParams, RectangleDetectorParams


class TestJetConfig(Config):
    __image_scaling = 6
    preprocessing_params = PreprocessingParams(
        gaussian_blur=9,
        image_scaling=__image_scaling,
        image_rotation=0,
        red_threshold=130,
        min_red_contour=125,
        min_area=(75 * __image_scaling) ** 2
    )
    edge_detector_params = EdgeDetectorParams(
        hysteresis_min_thresh=35,
        hysteresis_max_thresh=70,
        kernel_size=(5, 5),
        kernel_shape=cv2.MORPH_RECT,
        dilation_steps=3
    )
    segment_detector_params = SegmentDetectorParams(
        d_rho=1,
        d_theta=np.pi / 180,
        min_num_votes=50,
        min_line_length=max(floor(50 * (__image_scaling - 2)), 50),
        max_line_gap=50 * __image_scaling,
        extension_pixels=15 * __image_scaling
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
        max_merging_angle=np.pi / 180 * 10,
        max_endpoint_distance=25
    )
    intersection_detector_params = IntersectionDetectorParams(
        angle_threshold=np.pi / 180 * 25
    )
    rectangle_detector_params = RectangleDetectorParams(
        aspect_ratio=1.5,
        aspect_ratio_relative_deviation=0.35,
        min_area=20*40
    )
