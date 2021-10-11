from detector.detection import PreprocessingParams, EdgeDetectorParams, SegmentDetectorParams, SegmentClustererParams, \
    ClusterCleaningParams, IntersectionDetectorParams, RectangleDetectorParams
from detector.utils.pvpd_base_class import PVPDBaseClass


class Config(PVPDBaseClass):
    preprocessing_params: PreprocessingParams = None
    edge_detector_params: EdgeDetectorParams = None
    segment_detector_params: SegmentDetectorParams = None
    segment_clusterer_params: SegmentClustererParams = None
    cluster_cleaning_params: ClusterCleaningParams = None
    intersection_detector_params: IntersectionDetectorParams = None
    rectangle_detector_params: RectangleDetectorParams = None
