from ...src.vehicle.lp_detector.lp_detector import LPDetector
from ...src.vehicle.lp_recognition.lp_recognition import LPRecognition

class LoadModel():
    def __init__(self, cfg_base):
        self.lp_detector = LPDetector(cfg_base)
        self.lp_recognition = LPRecognition(cfg_base)