import logging
from typing import Tuple, Any, Type, List

from cv2 import cv2
from matplotlib import pyplot as plt

from detector.configs.abstract import Config
from detector.configs.test_jet import TestJetConfig
from detector.labelers.abstract import RectangleLabeler
from detector.labelers.yolo import YoloRectangleLabeler
from detector.utils.display import draw_rectangles, draw_intersections, draw_segments, display_image_in_actual_size
from detector.utils.utils import read_bgr_img, rectangle_annotated_photos
from thermography.detection import RectangleDetector, IntersectionDetector, \
    SegmentClusterer, SegmentDetector, EdgeDetector, FramePreprocessor, EdgeDetectorParams, SegmentClustererParams


class Detector:
    @staticmethod
    def preprocess_frame_functional(frame, params, silent) -> Tuple[Any, Any]:
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
        if not silent:
            plt.subplot(221), plt.imshow(cv2.cvtColor(last_scaled_frame_rgb, cv2.COLOR_BGR2RGB))
            plt.title('last_scaled_frame_rgb'), plt.xticks([]), plt.yticks([])

            plt.subplot(222), plt.imshow(cv2.cvtColor(last_scaled_frame, cv2.COLOR_BGR2RGB))
            plt.title('last_scaled_fram'), plt.xticks([]), plt.yticks([])

            plt.subplot(223), plt.imshow(cv2.cvtColor(last_preprocessed_image, cv2.COLOR_BGR2RGB))
            plt.title('last_preprocessed_image'), plt.xticks([]), plt.yticks([])

            plt.subplot(224), plt.imshow(cv2.cvtColor(last_attention_image, cv2.COLOR_BGR2RGB))
            plt.title('last_attention_image'), plt.xticks([]), plt.yticks([])

            plt.show()

        return last_preprocessed_image, last_scaled_frame_rgb

    @staticmethod
    def detect_edges_functional(frame, params: EdgeDetectorParams, silent=False) -> Any:
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
    def cluster_segments_by_contours(preprocessed, cluster_list, cluster_cleaning_params):
        contours, hierarchy = cv2.findContours(preprocessed, cv2.RETR_TREE, cv2.INTERSECT_FULL)
        if len(contours) <= 1:
            logging.info("No need to cluster by contours")
        moments = [cv2.moments(contour) for contour in contours]
        centroids = [(int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])) for M in moments]

        if len(centroids) >= 4:
            logging.warning(
                f"Barely possible that there is {len(centroids)} centroids. It's better to leave segments clusters as they are")
            return cluster_list

        column_clusterer = SegmentClustererParams(
            num_init=10,
            num_clusters=len(centroids),
            swipe_clusters=True,
            use_angles=False,
            centroids=centroids
        )
        if len(cluster_list[0]) > len(cluster_list[1]):
            columns_cluster_list = Detector.cluster_segments_functional(
                cluster_list[0],
                params=column_clusterer,
                cleaning_params=cluster_cleaning_params
            )
            del cluster_list[0]
            cluster_list.extend(columns_cluster_list)
        else:
            columns_cluster_list = Detector.cluster_segments_functional(
                cluster_list[1],
                params=column_clusterer,
                cleaning_params=cluster_cleaning_params
            )
            del cluster_list[1]
            cluster_list.extend(columns_cluster_list)
        return cluster_list

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
    def get_rectangles_labels(rectangles, rectangle_labeler: Type[RectangleLabeler], preprocessed_image,
                              image_path) -> Any:
        """Create label files using labeler based on detected rectangles."""
        labeler = rectangle_labeler(rectangles=rectangles, preprocessed_image=preprocessed_image, image_path=image_path)
        label_file = labeler.create_label_file()
        logging.info(f"Created label file: {label_file}")

        return labeler.labels_collector

    @staticmethod
    def main(image_path, config: Config, labelers: List[Type[RectangleLabeler]] = None, silent: bool = True,
             downscale_output: bool = True):
        img = read_bgr_img(image_path)

        # distorted_image = img
        # if False:
        #     undistorted_image = cv2.undistort(src=distorted_image)
        # else:
        #     undistorted_image = distorted_image
        preprocessed, last_scaled_frame_rgb = Detector.preprocess_frame_functional(img, config.preprocessing_params,
                                                                                   silent)
        edge_image = Detector.detect_edges_functional(preprocessed, config.edge_detector_params, silent)
        segments = Detector.detect_segments_functional(edge_image, config.segment_detector_params)

        if not silent:
            display_image_in_actual_size(preprocessed)
            display_image_in_actual_size(edge_image)

        general_cluster_list = Detector.cluster_segments_functional(
            segments,
            params=config.segment_clusterer_params,
            cleaning_params=config.cluster_cleaning_params)

        contours_clustered_cluster_list = Detector.cluster_segments_by_contours(preprocessed, general_cluster_list,
                                                                                config.cluster_cleaning_params)

        intersections = Detector.detect_intersections_functional(contours_clustered_cluster_list,
                                                                 params=config.intersection_detector_params)
        rectangles = Detector.detect_rectangles_functional(intersections, params=config.rectangle_detector_params)
        for labeler in labelers:
            Detector.get_rectangles_labels(rectangles, labeler, preprocessed, image_path)

        if not silent:
            draw_segments(general_cluster_list, last_scaled_frame_rgb, "Segments", render_indices=True)
            draw_intersections(intersections, last_scaled_frame_rgb, "Intersections")
            draw_rectangles(rectangles, last_scaled_frame_rgb, "Rectangles")
        annotated_photo = rectangle_annotated_photos(rectangles, last_scaled_frame_rgb)
        if downscale_output:
            annotated_photo = cv2.resize(annotated_photo, (img.shape[1], img.shape[0]))
        return annotated_photo


if __name__ == '__main__':
    Detector.main('../data/thermal/TEMP_DJI_8_R (286).JPG', TestJetConfig(), labeler=[YoloRectangleLabeler],
                  silent=False)