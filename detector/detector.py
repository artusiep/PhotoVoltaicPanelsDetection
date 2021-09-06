import logging
from typing import Tuple, Any, Type, List

import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt

from detector.configs.abstract import Config
from detector.labelers.abstract import RectangleLabeler
from detector.utils.display import draw_rectangles, draw_intersections, draw_segments, display_image_in_actual_size
from detector.utils.utils import read_bgr_img, rectangle_annotated_photos
from thermography.detection import RectangleDetector, IntersectionDetector, \
    SegmentClusterer, SegmentDetector, EdgeDetector, FramePreprocessor, EdgeDetectorParams
from thermography.utils import scale_image


class Detector:
    @staticmethod
    def preprocess_frame_functional(frame, params, silent) -> Tuple[Any, Any, Any]:
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
        mask = frame_preprocessor.mask
        if not silent:
            plt.subplot(231), plt.imshow(cv2.cvtColor(last_scaled_frame_rgb, cv2.COLOR_BGR2RGB))
            plt.title('last_scaled_frame_rgb'), plt.xticks([]), plt.yticks([])

            plt.subplot(222), plt.imshow(cv2.cvtColor(last_scaled_frame, cv2.COLOR_BGR2RGB))
            plt.title('last_scaled_fram'), plt.xticks([]), plt.yticks([])

            plt.subplot(223), plt.imshow(cv2.cvtColor(last_preprocessed_image, cv2.COLOR_BGR2RGB))
            plt.title('last_preprocessed_image'), plt.xticks([]), plt.yticks([])

            plt.subplot(224), plt.imshow(cv2.cvtColor(last_attention_image, cv2.COLOR_BGR2RGB))
            plt.title('last_attention_image'), plt.xticks([]), plt.yticks([])

            plt.show()

        return last_preprocessed_image, last_scaled_frame_rgb, mask

    @staticmethod
    def detect_edges_functional(frame, params: Type[EdgeDetectorParams], silent=False) -> Any:
        """Detects the edges in the :attr:`self.last_preprocessed_image` using the parameters in :attr:`self.edge_detection_parameters`.

        See Also:
            Module :mod:`~thermography.detection.edge_detection` for more details."""
        edge_detector = EdgeDetector(input_image=frame, params=params)
        edge_detector.detect()

        edge_image = edge_detector.edge_image

        if not silent:
            plt.subplot(121), plt.imshow(frame)
            plt.title('preprocessed'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(edge_image)
            plt.title('edge_image'), plt.xticks([]), plt.yticks([])
            plt.show()
        return edge_image

    @staticmethod
    def detect_segments_functional(frame, params) -> Any:
        """Detects the segments in the :attr:`self.last_edges_frame` using the parameters in :attr:`self.segment_detection_parameters`.
        See Also:
            Module :mod:`~thermography.detection.segment_detection` for more details."""
        segment_detector = SegmentDetector(input_image=frame, params=params)
        segment_detector.detect()
        return segment_detector.segments

    @staticmethod
    def cluster_segments_functional(segments, params, cleaning_params=None) -> Any:
        """Clusters the segments in :attr:`self.last_segments` according to the parameters in :attr:`self.segment_clustering_parameters`.
        See Also:
            Module :mod:`~thermography.detection.segment_clustering` for more details."""
        segment_clusterer = SegmentClusterer(input_segments=segments, params=params)
        segment_clusterer.cluster_segments()
        if cleaning_params:
            mean_angles, mean_centers = segment_clusterer.compute_cluster_mean()
            segment_clusterer.clean_clusters(mean_angles=mean_angles, params=cleaning_params)
        return segment_clusterer.cluster_list

    @staticmethod
    def detect_intersections_functional(cluster_list, params) -> Any:
        """Detects the intersections between the segments in :attr:`self.last_cluster_list` according to the parameters in :attr:`self.intersection_detection_parameters`.
        See Also:
            Module :mod:`~thermography.detection.intersection_detection` for more details."""
        intersection_detector = IntersectionDetector(input_segments=cluster_list, params=params)
        intersection_detector.detect()
        last_raw_intersections = intersection_detector.raw_intersections
        last_intersections = intersection_detector.cluster_cluster_intersections

        return last_intersections

    @staticmethod
    def detect_rectangles_functional(intersections, params) -> Any:
        """Detects the rectangles defined through the intersections in :attr:`self.last_intersections` according to the parameters in :attr:`self.rectangle_detection_parameters`.
        See Also:
            Module :mod:`~thermography.detection.rectangle_detection` for more details."""
        rectangle_detector = RectangleDetector(input_intersections=intersections, params=params)
        rectangle_detector.detect()
        last_rectangles = rectangle_detector.rectangles
        return last_rectangles

    @staticmethod
    def get_rectangles_labels(rectangles: List[np.ndarray], rectangle_labeler: Type[RectangleLabeler],
                              preprocessed_image: np.ndarray, edge_images: List[np.ndarray], label_path: str) -> Any:
        """Create label files using labeler based on detected rectangles."""
        labeler = rectangle_labeler(rectangles=rectangles, preprocessed_image=preprocessed_image, label_path=label_path,
                                    edge_images=edge_images)
        if label_path:
            label_file = labeler.create_label_file()
            logging.info(f"Created label file: {label_file}")
        else:
            logging.info(f"Cannot create label file. Parameter label_path not provided")

        return labeler.labels_collector

    @staticmethod
    def process_panel(contours, contour_id, preprocessed, last_scaled_frame_rgb, config, silent):
        mask_p = np.zeros_like(preprocessed)
        cv2.drawContours(mask_p, contours, contour_id, 255, cv2.FILLED)
        roi_image = cv2.bitwise_and(mask_p, preprocessed)

        edge_image = Detector.detect_edges_functional(roi_image, config.edge_detector_params, silent)
        segments = Detector.detect_segments_functional(edge_image, config.segment_detector_params)

        general_cluster_list = Detector.cluster_segments_functional(
            segments,
            params=config.segment_clusterer_params,
            cleaning_params=config.cluster_cleaning_params)

        intersections = Detector.detect_intersections_functional(general_cluster_list,
                                                                 params=config.intersection_detector_params)
        if not silent:
            display_image_in_actual_size(roi_image)
            display_image_in_actual_size(edge_image)
            draw_segments(general_cluster_list, last_scaled_frame_rgb, "Segments")
            draw_intersections(intersections, last_scaled_frame_rgb, "Intersections")

        return Detector.detect_rectangles_functional(intersections, params=config.rectangle_detector_params), edge_image

    @staticmethod
    def main(image_path, config: Config, labelers: List[Type[RectangleLabeler]] = None, labels_path: str = None,
             silent: bool = True, downscale_output: bool = True):
        img = read_bgr_img(image_path)

        # distorted_image = img
        # if False:
        #     undistorted_image = cv2.undistort(src=distorted_image)
        # else:
        #     undistorted_image = distorted_image
        preprocessed, last_scaled_frame_rgb, mask = Detector.preprocess_frame_functional(img,
                                                                                         config.preprocessing_params,
                                                                                         silent)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.INTERSECT_FULL)
        last_scaled_frame_rgb = scale_image(last_scaled_frame_rgb, config.edge_detector_params.image_scaling)

        rectangles = []
        edge_images = []

        for contour_id, _ in enumerate(contours):
            contour_rectangles, edge_image = Detector.process_panel(
                contours,
                contour_id,
                preprocessed,
                last_scaled_frame_rgb,
                config,
                silent)
            rectangles.extend(contour_rectangles)
            edge_images.append(edge_image)

        for labeler in labelers:
            Detector.get_rectangles_labels(rectangles, labeler, preprocessed, edge_images, labels_path)

        if not silent:
            draw_rectangles(rectangles, last_scaled_frame_rgb, "Rectangles")

        annotated_photo = rectangle_annotated_photos(rectangles, last_scaled_frame_rgb)
        if downscale_output:
            annotated_photo = cv2.resize(annotated_photo, (img.shape[1], img.shape[0]))
        return annotated_photo
