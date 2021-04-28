import glob
import json
import os
from math import floor

import cv2
import numpy as np
import pytest

from main import preprocess_frame_funcitonal, detect_edges_funcitonal, detect_segments_functional, \
    cluster_segments_functional, detect_intersections_functional, detect_rectangles_functional
from thermography.detection import PreprocessingParams, EdgeDetectorParams, SegmentDetectorParams, \
    SegmentClustererParams, \
    ClusterCleaningParams, IntersectionDetectorParams, RectangleDetectorParams

image_path = "extractor/_thermal_jet.jpg"
# image_path = "data/raw/sample.JPG"
image_path = "data/thermal/TEMP_DJI_1_R (659).JPG"

img = cv2.imread(image_path, cv2.IMREAD_COLOR)


def get_absolute_path(file):
    return os.path.abspath(file)


def check_success(tweaking, *args):
    values = "-".join(*args)
    file_suffix = f"*---{values}"
    file_path = f"test_results/{tweaking}/{file_suffix}.txt"

    files = glob.glob(file_path)
    if len(files) == 1:
        print(get_absolute_path(files[0]))
        return True
    else:
        print(files)
    return False


def check_err(tweaking, *args):
    values = "-".join(*args)
    file_suffix = f"err---{values}"
    file_path = f"test_results/{tweaking}/{file_suffix}.txt"

    if glob.glob(file_path):
        print(get_absolute_path(file_path))
        return True


def write_error_maker(args, tweaking, error):
    values = "-".join(args)
    file_suffix = f"err---{values}"
    file_path = f"test_results/{tweaking}/{file_suffix}.txt"
    with open(file_path, "w") as file_object:
        json.dump({'error': error}, file_object)


@pytest.mark.parametrize("hysteresis_min_thresh", [x * 5 for x in range(1, 51)])
@pytest.mark.parametrize("hysteresis_max_thresh", [x * 5 for x in range(1, 51)])
@pytest.mark.parametrize("morph_type", [1, 2])
def test_edge_tweaking(hysteresis_min_thresh, hysteresis_max_thresh, morph_type):
    tweaking = "edge"
    human_morph_type = "MORPH_ELLIPSE" if morph_type == 2 else "MORPH_CROSS"
    args = [human_morph_type, str(hysteresis_min_thresh), str(hysteresis_max_thresh)]
    if hysteresis_min_thresh >= hysteresis_max_thresh:
        error_msg = "hysteresis_min_thresh is larger than hysteresis_max_thresh"
        write_error_maker(args, tweaking, error_msg)
        assert False

    if check_err(tweaking, args):
        assert False
    if check_success(tweaking, args):
        assert True
        return

    image_scaling = 6
    preprocessed, last_scaled_frame_rgb = preprocess_frame_funcitonal(
        img,
        PreprocessingParams(
            gaussian_blur=9,
            image_scaling=image_scaling,
            image_rotation=0,
            red_threshold=100,
            min_area=(100 * (image_scaling)) ** 2
        ))
    edge_image = detect_edges_funcitonal(
        preprocessed,
        EdgeDetectorParams(
            hysteresis_min_thresh=hysteresis_min_thresh,
            hysteresis_max_thresh=hysteresis_max_thresh,
            kernel_size=(3, 3),
            kernel_shape=morph_type,
            dilation_steps=4
        ))
    segment_image = detect_segments_functional(
        edge_image,
        SegmentDetectorParams(
            d_rho=1.0,
            d_theta=np.pi / 180,
            min_num_votes=75,
            min_line_length=max(floor(50 * (image_scaling - 2)), 50),
            max_line_gap=50 * image_scaling,
            extension_pixels=15 * image_scaling
        ))

    try:
        cluster_list = cluster_segments_functional(
            segment_image,
            params=SegmentClustererParams(
                num_init=10,
                num_clusters=2,
                swipe_clusters=True,
                cluster_type="gmm",
                use_angles=True,
                use_centers=True
            ),
            cleaning_params=ClusterCleaningParams(
                max_angle_variation_mean=np.pi / 180 * 20,
                max_merging_angle=np.pi / 180 * 10,
                max_endpoint_distance=10.0 * (image_scaling - 2)
            ))
    except Exception as e:
        write_error_maker(args, tweaking, str(e))
        assert False

    intersections = detect_intersections_functional(
        cluster_list,
        params=IntersectionDetectorParams(
            angle_threshold=np.pi / 180 * 25
        ))
    rectangles = detect_rectangles_functional(
        intersections,
        params=RectangleDetectorParams(
            aspect_ratio=1.5,
            aspect_ratio_relative_deviation=0.35,
            min_area=floor(20 * (image_scaling - 1)) * floor(40 * (image_scaling - 1))
        ))

    result = {
        "hysteresis_min_thresh": hysteresis_min_thresh,
        "hysteresis_max_thresh": hysteresis_max_thresh,
        "morph_type": human_morph_type,
        "rectangles": [x.tolist() for x in rectangles],
        "len(rectangles)": len(rectangles),
        "cluster_list": [x.tolist() for x in cluster_list]
    }

    values = "-".join(args)
    file_suffix = f"{len(rectangles)}---{values}"
    file_path = f"test_results/{tweaking}/{file_suffix}.txt"

    with open(file_path, "w") as file_object:
        json.dump(result, file_object)

    print(get_absolute_path(file_path))

    assert True
