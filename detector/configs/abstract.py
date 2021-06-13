from detector.utils.pvpd_base_class import PVPDBaseClass


class Config(PVPDBaseClass):
    preprocessing_params = None
    edge_detector_params = None
    segment_detector_params = None
    segment_clusterer_params = None
    cluster_cleaning_params = None
    intersection_detector_params = None
    rectangle_detector_params = None
