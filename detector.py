from detector.configs.plasma import PlasmaConfig
from detector.detector import Detector
from detector.labelers.yolo import YoloRectangleLabeler
from detector.logger import init_logger

if __name__ == '__main__':
    init_logger()
    Detector.main('experiments/data/thermal/plasma-DJI_5_R (709).JPG', PlasmaConfig(), labelers=[YoloRectangleLabeler],
                  silent=False, labels_path="")