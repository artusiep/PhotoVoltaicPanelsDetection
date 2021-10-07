from typing import Type

from detector.detection import PreprocessingParams, EdgeDetectorParams, SegmentDetectorParams, SegmentClustererParams, \
    ClusterCleaningParams, IntersectionDetectorParams, RectangleDetectorParams
from detector.utils.pvpd_base_class import PVPDBaseClass


class Config(PVPDBaseClass):
    preprocessing_params: Type[PreprocessingParams] = None
    edge_detector_params: Type[EdgeDetectorParams] = None
    segment_detector_params: Type[SegmentDetectorParams] = None
    segment_clusterer_params: Type[SegmentClustererParams] = None
    cluster_cleaning_params: Type[ClusterCleaningParams] = None
    intersection_detector_params: Type[IntersectionDetectorParams] = None
    rectangle_detector_params: Type[RectangleDetectorParams] = None
