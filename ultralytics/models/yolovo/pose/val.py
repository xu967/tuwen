# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""YOLOVO pose estimation validator."""

from ultralytics.models.yolo.pose import PoseValidator
from ultralytics.models.yolovo.val import YOLOVOValidator


class YOLOVOPoseValidator(YOLOVOValidator, PoseValidator):
    """Pose validator with YOLOVO text-cache lifecycle."""

    pass
