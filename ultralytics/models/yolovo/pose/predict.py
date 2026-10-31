# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""YOLOVO pose estimation predictor."""

from ultralytics.models.yolo.pose import PosePredictor
from ultralytics.models.yolovo.predict import YOLOVOPredictor


class YOLOVOPosePredictor(YOLOVOPredictor, PosePredictor):
    """Pose predictor with YOLOVO prompt lifecycle."""

    pass
