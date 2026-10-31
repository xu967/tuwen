# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

__version__ = "8.3.163"

import os

# Set ENV variables (place before imports)
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU utilization during training

# Import standard models
from ultralytics.models import NAS, RTDETR, SAM, YOLO, YOLOE, FastSAM, YOLOWorld

# Import YOLOVO (conditional)
try:
    from ultralytics.models.yolovo.model import YOLOVO
    _YOLOVO_AVAILABLE = True
except ImportError:
    _YOLOVO_AVAILABLE = False
    YOLOVO = None  # Set to None if not available

# Import DepthGen (conditional)
try:
    from ultralytics.nn.mm import DepthGen
    _DEPTHGEN_AVAILABLE = True
except Exception:
    _DEPTHGEN_AVAILABLE = False
    DepthGen = None

# Import DEMGen (conditional)
try:
    from ultralytics.nn.mm import DEMGen
    _DEMGEN_AVAILABLE = True
except Exception:
    _DEMGEN_AVAILABLE = False
    DEMGen = None

# Import EdgeGen (conditional)
try:
    from ultralytics.nn.mm import EdgeGen
    _EDGEGEN_AVAILABLE = True
except Exception:
    _EDGEGEN_AVAILABLE = False
    EdgeGen = None
from ultralytics.utils import ASSETS, SETTINGS
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

# Import MultiModal Sampler Tool (conditional)
try:
    from ultralytics.tools import MultiModalSampler, sample_from_yaml, quick_sample
    _MMSAMPLER_AVAILABLE = True
except ImportError:
    _MMSAMPLER_AVAILABLE = False
    MultiModalSampler = None
    sample_from_yaml = None
    quick_sample = None

settings = SETTINGS

# Dynamic __all__ based on availability
__all__ = [
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOWorld",
    "YOLOE",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
]

if _YOLOVO_AVAILABLE:
    __all__.append("YOLOVO")
if _DEPTHGEN_AVAILABLE:
    __all__.append("DepthGen")
if _DEMGEN_AVAILABLE:
    __all__.append("DEMGen")
if _EDGEGEN_AVAILABLE:
    __all__.append("EdgeGen")
if _MMSAMPLER_AVAILABLE:
    __all__.extend(["MultiModalSampler", "sample_from_yaml", "quick_sample"])

__all__ = tuple(__all__)

# Error Reporter — capture unhandled exceptions and generate structured reports
try:
    from pathlib import Path as _Path
    from ultralytics.utils.error_reporter import ErrorReporter as _ErrorReporter
    _error_reporter = _ErrorReporter(project_root=str(_Path(__file__).resolve().parents[1]))
except Exception:
    pass
