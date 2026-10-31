# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""YOLOVO oriented bounding box predictor."""

from ultralytics.models.yolo.obb import OBBPredictor
from ultralytics.models.yolovo.predict import YOLOVOPredictor


class YOLOVOOBBPredictor(YOLOVOPredictor, OBBPredictor):
    """OBB predictor with YOLOVO prompt lifecycle."""

    pass
