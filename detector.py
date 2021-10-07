from detector.configs.plasma import PlasmaConfig
from detector.detector import Detector
from detector.labelers.pickle import PickleLabeler
from detector.labelers.tiff import TiffImageLabeler
from detector.labelers.yolo import YoloRectangleLabeler
from detector.logger import init_logger

if __name__ == '__main__':
    init_logger()
    Detector.main('data/plasma/6.JPG', PlasmaConfig(), labelers=[TiffImageLabeler, YoloRectangleLabeler, PickleLabeler],
                  silent=False)
