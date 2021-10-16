import logging
from datetime import datetime
from typing import Tuple, Any, Type, List

import numpy as np
from cv2 import cv2
from matplotlib import pyplot

from detector.configs.abstract import Config
from detector.detection import Preprocessor, PreprocessingParams, EdgeDetectorParams, SegmentDetector, \
    SegmentDetectorParams, EdgeDetector, SegmentClusterer, ClusterCleaningParams, SegmentClustererParams, \
    IntersectionDetector, IntersectionDetectorParams, RectangleDetector, RectangleDetectorParams, scale_image
from detector.detection.preprocessing_ml import PreprocessorMl, PreprocessingMlParams
from detector.labelers.abstract import RectangleLabeler
from detector.utils.display import draw_rectangles, draw_intersections, draw_segments, display_image_in_actual_size
from detector.utils.utils import read_bgr_img, rectangle_annotated_photos


class Detector:
    @staticmethod
    def preprocess_image_functional(
            image: np.ndarray,
            params: PreprocessingParams,
            silent: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Preprocesses the image :param:`image` by scaling, rotating and computing the attention regions.

        See Also:
            Module :mod:`~detection.preprocessing` for more details.
        """
        preprocessor = Preprocessor(input_image=image, params=params)
        preprocessor.preprocess()
        scaled_image_rgb = preprocessor.scaled_image_rgb
        scaled_image = preprocessor.scaled_image
        preprocessed_image = preprocessor.preprocessed_image
        attention_image = preprocessor.attention_image
        mask = preprocessor.mask
        if not silent:
            pyplot.subplot(231), pyplot.imshow(cv2.cvtColor(scaled_image_rgb, cv2.COLOR_BGR2RGB))
            pyplot.title('scaled_image_rgb'), pyplot.xticks([]), pyplot.yticks([])

            pyplot.subplot(222), pyplot.imshow(cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB))
            pyplot.title('scaled_image'), pyplot.xticks([]), pyplot.yticks([])

            pyplot.subplot(223), pyplot.imshow(cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB))
            pyplot.title('preprocessed_image'), pyplot.xticks([]), pyplot.yticks([])

            pyplot.subplot(224), pyplot.imshow(cv2.cvtColor(attention_image, cv2.COLOR_BGR2RGB))
            pyplot.title('attention_image'), pyplot.xticks([]), pyplot.yticks([])

            pyplot.show()

        return preprocessed_image, scaled_image_rgb, mask

    @staticmethod
    def preprocess_ml_image_functional(
            image: np.ndarray,
            params: PreprocessingMlParams,
            silent: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Preprocesses the image :param:`image` by scaling, rotating and computing the attention regions.

        See Also:
            Module :mod:`~detection.preprocessing` for more details.
        """
        preprocessor = PreprocessorMl(input_image=image, params=params)
        preprocessor.preprocess()
        scaled_image_rgb = preprocessor.scaled_image_rgb
        scaled_image = preprocessor.scaled_image
        preprocessed_image = preprocessor.preprocessed_image
        attention_image = preprocessor.attention_image
        mask = preprocessor.mask
        if not silent:
            pyplot.subplot(231), pyplot.imshow(cv2.cvtColor(scaled_image_rgb, cv2.COLOR_BGR2RGB))
            pyplot.title('scaled_image_rgb'), pyplot.xticks([]), pyplot.yticks([])

            pyplot.subplot(222), pyplot.imshow(cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB))
            pyplot.title('scaled_image'), pyplot.xticks([]), pyplot.yticks([])

            pyplot.subplot(223), pyplot.imshow(cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB))
            pyplot.title('preprocessed_image'), pyplot.xticks([]), pyplot.yticks([])

            pyplot.subplot(224), pyplot.imshow(cv2.cvtColor(attention_image, cv2.COLOR_BGR2RGB))
            pyplot.title('attention_image'), pyplot.xticks([]), pyplot.yticks([])

            pyplot.show()

        return preprocessed_image, scaled_image_rgb, mask

    @staticmethod
    def detect_edges_functional(preprocessed_image: np.ndarray, params: EdgeDetectorParams, silent=False) -> np.ndarray:
        """Detects the edges in the :param:`preprocessed_image` using the parameters in :param:`params`.

        See Also:
            Module :mod:`~detection.edge_detection` for more details.
        """
        edge_detector = EdgeDetector(input_image=preprocessed_image, params=params)
        edge_detector.detect()

        edge_image = edge_detector.edge_image

        if not silent:
            from matplotlib import pyplot
            pyplot.subplot(121), pyplot.imshow(preprocessed_image)
            pyplot.title('preprocessed'), pyplot.xticks([]), pyplot.yticks([])
            pyplot.subplot(122), pyplot.imshow(edge_image)
            pyplot.title('edge_image'), pyplot.xticks([]), pyplot.yticks([])
            pyplot.show()
        return edge_image

    @staticmethod
    def detect_segments_functional(edge_image: np.ndarray, params: SegmentDetectorParams) -> Any:
        """
        Detects the segments in the :param:edge_image` using the parameters in :param:`params`.

        See Also:
            Module :mod:`~detection.segment_detection` for more details.
        """
        segment_detector = SegmentDetector(input_image=edge_image, params=params)
        segment_detector.detect()
        return segment_detector.segments

    @staticmethod
    def cluster_segments_functional(segments, params: SegmentClustererParams,
                                    cleaning_params: ClusterCleaningParams) -> Any:
        """Clusters the segments in :param:`segments` according to the parameters in :param:`param`.
        After that aggregate and clean cluster using :param:`cleaning_params`
        See Also:
            Module :mod:`~detection.segment_clustering` for more details.
        """
        segment_clusterer = SegmentClusterer(input_segments=segments, params=params)
        segment_clusterer.cluster_segments()
        if cleaning_params:
            mean_angles, mean_centers = segment_clusterer.compute_cluster_mean()
            segment_clusterer.clean_clusters(mean_angles=mean_angles, params=cleaning_params)
        return segment_clusterer.cluster_list

    @staticmethod
    def detect_intersections_functional(cluster_list, params: IntersectionDetectorParams) -> Any:
        """Detects the intersections between the segments in :param:`cluster_list` according to the parameters
            in :param:`params`.
        See Also:
            Module :mod:`~detection.intersection_detection` for more details.
        """
        intersection_detector = IntersectionDetector(input_segments=cluster_list, params=params)
        intersection_detector.detect()
        last_raw_intersections = intersection_detector.raw_intersections
        last_intersections = intersection_detector.cluster_cluster_intersections

        return last_intersections

    @staticmethod
    def detect_rectangles_functional(intersections, params: RectangleDetectorParams) -> Any:
        """Detects the rectangles defined through the intersections in :param:`intersections` according to the
        parameters in :param:`params`.
        See Also:
            Module :mod:`~detection.rectangle_detection` for more details.
        """
        rectangle_detector = RectangleDetector(input_intersections=intersections, params=params)
        rectangle_detector.detect()
        rectangles = rectangle_detector.rectangles
        return rectangles

    @staticmethod
    def get_rectangles_labels(rectangles: List[np.ndarray], rectangle_labeler: Type[RectangleLabeler],
                              thermal_image: np.ndarray, image_path: str,
                              preprocessed_image: np.ndarray, label_path: str, tags: dict) -> Any:
        """Create label files using labeler based on detected rectangles."""
        labeler = rectangle_labeler(
            rectangles=rectangles,
            thermal_image=thermal_image,
            image_path=image_path,
            preprocessed_image=preprocessed_image,
            label_path=label_path,
            tags=tags
        )
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
        # We would like to have normalized rectangle coordinates accordingly to source image
        normalized_segments = segments
        general_cluster_list = Detector.cluster_segments_functional(
            normalized_segments,
            params=config.segment_clusterer_params,
            cleaning_params=config.cluster_cleaning_params)

        intersections = Detector.detect_intersections_functional(general_cluster_list,
                                                                 params=config.intersection_detector_params)
        if not silent:
            display_image_in_actual_size(roi_image)
            display_image_in_actual_size(edge_image)
            draw_segments(general_cluster_list, scale_image(last_scaled_frame_rgb, 3), "Segments")
            draw_intersections(intersections,  scale_image(last_scaled_frame_rgb, 3), "Intersections")

        rectangles = Detector.detect_rectangles_functional(intersections, params=config.rectangle_detector_params)
        return rectangles, edge_image

    @staticmethod
    def main(image_path: str, config: Config, labelers: List[Type[RectangleLabeler]] = None, labels_path: str = None,
             silent: bool = True):
        start_time = datetime.now()
        thermal_image = read_bgr_img(image_path)

        # distorted_image = img
        # if False:
        #     undistorted_image = cv2.undistort(src=distorted_image)
        # else:
        #     undistorted_image = distorted_image
        if isinstance(config.preprocessing_params, PreprocessingParams):
            preprocessing_func = Detector.preprocess_image_functional
        else:
            preprocessing_func = Detector.preprocess_ml_image_functional

        preprocessed_image, scaled_image_rgb, mask = preprocessing_func(thermal_image,
                                                                        config.preprocessing_params,
                                                                        silent)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.INTERSECT_FULL)

        rectangles = []

        for contour_id, _ in enumerate(contours):
            try:
                contour_rectangles, edge_image = Detector.process_panel(
                    contours,
                    contour_id,
                    preprocessed_image,
                    scaled_image_rgb,
                    config,
                    silent)
                rectangles.extend(contour_rectangles)
            except Exception as e:
                logging.error(f"Failed to process panel for contour_id {contour_id} due to {e}")

        end_time = datetime.now()

        tags = {
            'start_time': start_time,
            'end_time': end_time,
            'detection_duration': end_time - start_time
        }
        for labeler in labelers:
            Detector.get_rectangles_labels(
                rectangles,
                labeler,
                thermal_image,
                image_path,
                preprocessed_image,
                labels_path,
                tags
            )

        if not silent:
            draw_rectangles(rectangles,  scale_image(scaled_image_rgb,3), "rectangles")

        annotated_photo = rectangle_annotated_photos(rectangles, scaled_image_rgb)
        return annotated_photo, labels_path
