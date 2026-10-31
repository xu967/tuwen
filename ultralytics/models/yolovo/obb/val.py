# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""YOLOVO oriented bounding box validator."""

from ultralytics.models.yolo.obb import OBBValidator
from ultralytics.models.yolovo.val import YOLOVOValidator


class YOLOVOOBBValidator(YOLOVOValidator, OBBValidator):
    """OBB validator with YOLOVO text-cache lifecycle."""

    pass
