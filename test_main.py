import glob
import glob
import json
import os

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


img = cv2.imread(image_path, cv2.IMREAD_COLOR)


def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))


def get_absolute_path(file):
    return os.path.abspath(file)


def check_success(gaussian_blur, image_scaling, red_threshold, min_area):
    file_suffix = f"*---{gaussian_blur}-{image_scaling}-{red_threshold}-{min_area}"
    file_path = f"test_results/2/{file_suffix}.txt"

    files = glob.glob(file_path)
    if len(files) == 1:
        print(get_absolute_path(files[0]))
        return True
    else:
        print(files)
    return False


def check_err(err_file_path):
    if glob.glob(err_file_path):
        print(get_absolute_path(err_file_path))
        return True


@pytest.mark.parametrize("gaussian_blur", [x for x in range(1, 10)])
@pytest.mark.parametrize("image_scaling", [x * 0.5 for x in range(1, 20)])
@pytest.mark.parametrize("red_threshold", [x * 10 for x in range(7, 15)])
@pytest.mark.parametrize("min_area", [x * 10 for x in range(10, 20)])
def test_parameters(gaussian_blur, image_scaling, red_threshold, min_area):
    err_file = f"err---{gaussian_blur}-{image_scaling}-{red_threshold}-{min_area}"
    err_file_path = f"test_results/2/{err_file}.txt"
    if check_err(err_file_path):
        assert False
    if check_success(gaussian_blur, image_scaling, red_threshold, min_area):
        assert True
        return

    preprocessed, last_scaled_frame_rgb = preprocess_frame_funcitonal(
        img,
        PreprocessingParams(
            gaussian_blur=gaussian_blur,
            image_scaling=image_scaling,
            image_rotation=0,
            red_threshold=red_threshold,
            min_area=min_area ** 2
        ))
    edge_image = detect_edges_funcitonal(
        preprocessed,
        EdgeDetectorParams(
            hysteresis_min_thresh=40,
            hysteresis_max_thresh=50,
            kernel_size=(3, 3),
            kernel_shape=cv2.MORPH_CROSS,
            dilation_steps=4
        ))
    segment_image = detect_segments_functional(
        edge_image,
        SegmentDetectorParams(
            d_rho=1.0,
            d_theta=np.pi / 180,
            min_num_votes=60,
            min_line_length=50,
            max_line_gap=150,
            extension_pixels=10
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
                max_endpoint_distance=10.0
            ))
    except Exception as e:
        err_file = f"err---{gaussian_blur}-{image_scaling}-{red_threshold}-{min_area}"
        err_file_path = f"test_results/2/{err_file}.txt"
        with open(err_file_path, "w") as file_object:
            json.dump({'error': str(e)}, file_object)
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
            min_area=20 * 40
        ))

    result = {
        "gaussian_blur": gaussian_blur,
        "image_scaling": image_scaling,
        "red_threshold": red_threshold,
        "min_area": min_area,
        "rectangles": [x.tolist() for x in rectangles],
        "len(rectangles)": len(rectangles),
        # "intersections": {key: value.tolist() for key, value in intersections.items()},
        "cluster_list": [x.tolist() for x in cluster_list]
    }

    file_suffix = f"{len(rectangles)}---{gaussian_blur}-{image_scaling}-{red_threshold}-{min_area}"
    file_path = f"test_results/2/{file_suffix}.txt"

    with open(file_path, "w") as file_object:
        json.dump(result, file_object)
    # json.dump(result, codecs.open(file_path, 'w', encoding='utf-8'), default=default,

    print(get_absolute_path(file_path))

    assert True
