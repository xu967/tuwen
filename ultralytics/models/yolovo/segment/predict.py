# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""YOLOVO instance segmentation predictor."""

from ultralytics.models.yolo.segment import SegmentationPredictor
from ultralytics.models.yolovo.predict import YOLOVOPredictor


class YOLOVOSegPredictor(YOLOVOPredictor, SegmentationPredictor):
    """Segmentation predictor with YOLOVO prompt lifecycle."""

    pass
