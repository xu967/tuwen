# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""YOLOVO oriented bounding box components."""

from .predict import YOLOVOOBBPredictor
from .train import YOLOVOOBBTrainer
from .val import YOLOVOOBBValidator

__all__ = "YOLOVOOBBTrainer", "YOLOVOOBBValidator", "YOLOVOOBBPredictor"
