from abc import ABC


class Config(ABC):
    preprocessing_params = None
    edge_detector_params = None
    segment_detector_params = None
    segment_clusterer_params = None
    cluster_cleaning_params = None
    intersection_detector_params = None
    rectangle_detector_params = None
