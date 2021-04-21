import cv2
import numpy as np
from matplotlib import pyplot as plt

from thermography.detection import FramePreprocessor
from thermography.detection import EdgeDetector, SegmentDetector, SegmentClusterer, IntersectionDetector, \
    RectangleDetector, EdgeDetectorParams

image_path = "data/thermal/TEMP_DJI_2_R (814).JPG"

img = cv2.imread(image_path, cv2.IMREAD_COLOR)

distorted_image = img
if False:
    undistorted_image = cv2.undistort(src=distorted_image)
else:
    undistorted_image = distorted_image


def preprocess_frame_funcitonal(frame, imgs_show=False) -> None:
    """Preprocesses the frame stored at :attr:`self.last_input_frame` by scaling, rotating and computing the attention regions.
    See Also:
        Module :mod:`~thermography.detection.preprocessing` for more details.
    """
    frame_preprocessor = FramePreprocessor(input_image=frame)
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

    return last_preprocessed_image


def detect_edges_funcitonal(img) -> None:
    """Detects the edges in the :attr:`self.last_preprocessed_image` using the parameters in :attr:`self.edge_detection_parameters`.

    See Also:
        Module :mod:`~thermography.detection.edge_detection` for more details."""
    edge_detector = EdgeDetector(input_image=img, params=EdgeDetectorParams(img))
    edge_detector.detect()

    edge_image = edge_detector.edge_image
    return edge_image


def detect_segments_funcitonal(img) -> None:
    """Detects the segments in the :attr:`self.last_edges_frame` using the parameters in :attr:`self.segment_detection_parameters`.
    See Also:
        Module :mod:`~thermography.detection.segment_detection` for more details."""
    segment_detector = SegmentDetector(input_image=img)
    segment_detector.detect()
    return segment_detector.segments


def cluster_segments_funcitonal(segments) -> None:
    """Clusters the segments in :attr:`self.last_segments` according to the parameters in :attr:`self.segment_clustering_parameters`.
    See Also:
        Module :mod:`~thermography.detection.segment_clustering` for more details."""
    segment_clusterer = SegmentClusterer(input_segments=segments)
    segment_clusterer.cluster_segments()
    mean_angles, mean_centers = segment_clusterer.compute_cluster_mean()
    segment_clusterer.clean_clusters(mean_angles=mean_angles)
    return segment_clusterer.cluster_list


def detect_intersections_funcitonal(cluster_list) -> None:
    """Detects the intersections between the segments in :attr:`self.last_cluster_list` according to the parameters in :attr:`self.intersection_detection_parameters`.
    See Also:
        Module :mod:`~thermography.detection.intersection_detection` for more details."""
    intersection_detector = IntersectionDetector(input_segments=cluster_list)
    intersection_detector.detect()
    last_raw_intersections = intersection_detector.raw_intersections
    last_intersections = intersection_detector.cluster_cluster_intersections

    return last_intersections


def detect_rectangles_funcitonal(intersections) -> None:
    """Detects the rectangles defined through the intersections in :attr:`self.last_intersections` according to the parameters in :attr:`self.rectangle_detection_parameters`.
    See Also:
        Module :mod:`~thermography.detection.rectangle_detection` for more details."""
    rectangle_detector = RectangleDetector(input_intersections=intersections)
    rectangle_detector.detect()
    last_rectangles = rectangle_detector.rectangles
    return last_rectangles


preprocessed = preprocess_frame_funcitonal(img)
edge_image = detect_edges_funcitonal(preprocessed)
plt.subplot(121), plt.imshow(preprocessed)
plt.title('preprocessed'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edge_image)
plt.title('edge_image'), plt.xticks([]), plt.yticks([])
plt.show()
segement_image = detect_segments_funcitonal(edge_image)
from display import draw_segments, draw_rectangles


cluster_list = cluster_segments_funcitonal(segement_image)

# draw_segments(cluster_list, preprocessed, "Segments")
intersections = detect_intersections_funcitonal(cluster_list)
rectanbles = detect_rectangles_funcitonal(intersections)

draw_rectangles(rectanbles, img, "Rectangles")
# plt.subplot(133), plt.imshow(segement_image)
# plt.title('segement_image'), plt.xticks([]), plt.yticks([])


# self.detect_edges()
# self.detect_segments()
cv2.waitKey()
