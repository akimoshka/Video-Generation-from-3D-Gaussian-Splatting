# src/detector.py
from typing import List, Dict
import numpy as np

class Detector:
    def __init__(self, model_name: str = "yolov8s.pt", conf: float = 0.35):
        self.model_name = model_name
        self.conf = conf
        # TODO: load actual YOLO model

    def detect_image(self, image_bgr: np.ndarray) -> List[Dict]:
        """
        Returns list of:
        { "label": str, "conf": float, "bbox": [x1,y1,x2,y2] }
        """
        # TODO: implement
        return []

def localize_objects_3d(detections, meta) -> list:
    """
    From detections + per-frame meta (camera pose, intrinsics, depth)
    compute 3D world positions.

    Returns list of:
    { "label": str, "world": [x,y,z], "score": float }
    """
    # TODO: implement; for now:
    return []
