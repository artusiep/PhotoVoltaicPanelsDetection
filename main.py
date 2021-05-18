import os
from typing import Tuple, Any

import cv2
import numpy as np
from matplotlib import pyplot as plt

from configs.abstract import Config
from configs.jet import JetConfig
from display import draw_rectangles, draw_segments
from thermography.detection import RectangleDetector, IntersectionDetector, \
    SegmentClusterer, SegmentDetector, EdgeDetector, FramePreprocessor, EdgeDetectorParams


def preprocess_frame_funcitonal(frame, params, imgs_show=False) -> Tuple[Any, Any]:
    """Preprocesses the frame stored at :attr:`self.last_input_frame` by scaling, rotating and computing the attention regions.
    See Also:
        Module :mod:`~thermography.detection.preprocessing` for more details.
    """
    frame_preprocessor = FramePreprocessor(input_image=frame, params=params)
    frame_preprocessor.preprocess()
    last_scaled_frame_rgb = frame_preprocessor.scaled_image_rgb
    last_scaled_frame = frame_preprocessor.scaled_image
    last_preprocessed_image = frame_preprocessor.preprocessed_image
    last_attention_image = frame_preprocessor.attention_image
    if imgs_show:
        plt.subplot(141), plt.imshow(last_scaled_frame_rgb)
        plt.title('last_scaled_frame_rgb'), plt.xticks([]), plt.yticks([])

        plt.subplot(142), plt.imshow(last_scaled_frame)
        plt.title('last_scaled_fram'), plt.xticks([]), plt.yticks([])

        plt.subplot(143), plt.imshow(last_preprocessed_image)
        plt.title('ast_preprocessed_image'), plt.xticks([]), plt.yticks([])

        plt.subplot(144), plt.imshow(last_attention_image)
        plt.title('last_attention_image'), plt.xticks([]), plt.yticks([])

        plt.show()

    return last_preprocessed_image, last_scaled_frame_rgb


def detect_edges_functional(frame, params: EdgeDetectorParams) -> Any:
    """Detects the edges in the :attr:`self.last_preprocessed_image` using the parameters in :attr:`self.edge_detection_parameters`.

    See Also:
        Module :mod:`~thermography.detection.edge_detection` for more details."""
    edge_detector = EdgeDetector(input_image=frame, params=params)
    edge_detector.detect()

    edge_image = edge_detector.edge_image
    return edge_image


def detect_segments_functional(frame, params) -> Any:
    """Detects the segments in the :attr:`self.last_edges_frame` using the parameters in :attr:`self.segment_detection_parameters`.
    See Also:
        Module :mod:`~thermography.detection.segment_detection` for more details."""
    segment_detector = SegmentDetector(input_image=frame, params=params)
    segment_detector.detect()
    return segment_detector.segments


def cluster_segments_functional(segments, params, cleaning_params) -> Any:
    """Clusters the segments in :attr:`self.last_segments` according to the parameters in :attr:`self.segment_clustering_parameters`.
    See Also:
        Module :mod:`~thermography.detection.segment_clustering` for more details."""
    segment_clusterer = SegmentClusterer(input_segments=segments, params=params)
    segment_clusterer.cluster_segments()
    mean_angles, mean_centers = segment_clusterer.compute_cluster_mean()
    segment_clusterer.clean_clusters(mean_angles=mean_angles, params=cleaning_params)
    return segment_clusterer.cluster_list


def detect_intersections_functional(cluster_list, params) -> Any:
    """Detects the intersections between the segments in :attr:`self.last_cluster_list` according to the parameters in :attr:`self.intersection_detection_parameters`.
    See Also:
        Module :mod:`~thermography.detection.intersection_detection` for more details."""
    intersection_detector = IntersectionDetector(input_segments=cluster_list, params=params)
    intersection_detector.detect()
    last_raw_intersections = intersection_detector.raw_intersections
    last_intersections = intersection_detector.cluster_cluster_intersections

    return last_intersections


def detect_rectangles_functional(intersections, params) -> Any:
    """Detects the rectangles defined through the intersections in :attr:`self.last_intersections` according to the parameters in :attr:`self.rectangle_detection_parameters`.
    See Also:
        Module :mod:`~thermography.detection.rectangle_detection` for more details."""
    rectangle_detector = RectangleDetector(input_intersections=intersections, params=params)
    rectangle_detector.detect()
    last_rectangles = rectangle_detector.rectangles
    return last_rectangles


def auto_canny(image):
    sigma = 0.33
    v = np.ma.median(np.ma.masked_equal(image, 0))
    print(v)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return lower, upper


def annotated_photos(rectangles: list, base_image: np.ndarray):
    """Draws the rectangles contained in the first parameter onto the base image passed as second parameter.

    This function displays the image using the third parameter as title.

    :param rectangles: List of rectangles.
    :param base_image: Base image over which to render the rectangles.
    :param windows_name: Title to give to the rendered image.
    """
    mean_color = np.mean(base_image, axis=(0, 1))
    mask = np.zeros_like(base_image)
    if mean_color[0] == mean_color[1] == mean_color[2]:
        mean_color = np.array([255, 255, 0])
    opposite_color = np.array([255, 255, 255]) - mean_color
    opposite_color = (int(opposite_color[0]), int(opposite_color[1]), int(opposite_color[2]))
    for rectangle in rectangles:
        xmin, ymin = rectangle.min(axis=0)
        xmax, ymax = rectangle.max(axis=0)
        width = int(xmax) - int(xmin)
        height = int(ymax) - int(ymin)

        xcentral = int(width / 2 + int(xmin))
        ycentral = int(height / 2 + int(ymin))

        cv2.polylines(base_image, np.int32([rectangle]), True, opposite_color, 5, cv2.LINE_AA)
        cv2.fillConvexPoly(mask, np.int32([rectangle]), (255, 0, 0), cv2.LINE_4)
        cv2.circle(mask, (xcentral, ycentral), radius=10, color=(0, 0, 0), thickness=-1)

    cv2.addWeighted(base_image, 1, mask, 0.5, 0, base_image)

    return base_image


def save_img(img, path):
    catalogues = "/".join(path.split('/')[:-1])
    try:
        os.makedirs(catalogues)
    except FileExistsError:
        pass
    except Exception as e:
        print("Failed to create path to save image {e}")
    result = cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    assert result


def main(image_path, config: Config, silent=True):
    img = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # distorted_image = img
    # if False:
    #     undistorted_image = cv2.undistort(src=distorted_image)
    # else:
    #     undistorted_image = distorted_image
    preprocessed, last_scaled_frame_rgb = preprocess_frame_funcitonal(img, config.preprocessing_params, not silent)
    edge_image = detect_edges_functional(preprocessed, config.edge_detector_params)
    segment_image = detect_segments_functional(edge_image, config.segment_detector_params)

    if not silent:
        plt.subplot(121), plt.imshow(preprocessed)
        plt.title('preprocessed'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(edge_image)
        plt.title('edge_image'), plt.xticks([]), plt.yticks([])
        plt.show()

    cluster_list = cluster_segments_functional(
        segment_image,
        params=config.segment_clusterer_params,
        cleaning_params=config.cluster_cleaning_params)

    intersections = detect_intersections_functional(cluster_list, params=config.intersection_detector_params)
    rectangles = detect_rectangles_functional(intersections, params=config.rectangle_detector_params)

    if not silent:
        draw_segments(cluster_list, last_scaled_frame_rgb, "Segments")
        draw_rectangles(rectangles, last_scaled_frame_rgb, "Rectangles")
    return annotated_photos(rectangles, last_scaled_frame_rgb)


if __name__ == '__main__':
    main('extractor/_thermal_jet..jpg', JetConfig(), silent=False)
