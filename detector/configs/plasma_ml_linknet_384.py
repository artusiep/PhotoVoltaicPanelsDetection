from math import floor

import numpy as np
import cv2

from detector.configs.abstract import Config
from detector.detection import EdgeDetectorParams, SegmentDetectorParams, SegmentClustererParams, \
    ClusterCleaningParams, IntersectionDetectorParams, RectangleDetectorParams
from detector.detection.preprocessing_ml import PreprocessingMlParams


class PlasmaMlLinknet384Config(Config):
    """
    Download model weights: `gsutil -m cp gs://photo-voltaic-panels-detection/models/linknet_gray_0.9861.zip models && unzip models/linknet_gray_0.9861 -d models`
    """
    __preprocessing_image_scaling = 1
    __edge_image_scaling = 3
    preprocessing_params = PreprocessingMlParams(
        model_name='linknet',
        weight_path='detector/configs/models/linknet_gray_0.9861/cp.ckpt',
        gray=True,
        model_image_size=(384, 384),
        start_neurons=16,
        gaussian_blur=3,
        model_output_threshold=32,
        min_area=(120 * __preprocessing_image_scaling) ** 2
    )
    edge_detector_params = EdgeDetectorParams(
        image_scaling=__edge_image_scaling,
        hysteresis_min_thresh=35,
        hysteresis_max_thresh=45,
        kernel_size=(7, 7),
        kernel_shape=cv2.MORPH_CROSS,
        dilation_steps=4
    )
    segment_detector_params = SegmentDetectorParams(
        d_rho=1,
        d_theta=np.pi / 180,
        min_num_votes=60,
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
        max_merging_angle=np.pi / 180 * 30,
        max_endpoint_distance=10
    )
    intersection_detector_params = IntersectionDetectorParams(
        angle_threshold=np.pi / 180 * 25
    )
    rectangle_detector_params = RectangleDetectorParams(
        aspect_ratio=1.5,
        aspect_ratio_relative_deviation=0.35,
        min_area=20 * 30
    )
