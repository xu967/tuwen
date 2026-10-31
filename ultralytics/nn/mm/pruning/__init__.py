# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Zero-registration pruning engine for YOLOVO.

Dependency discovery is fully automatic (TorchDispatchMode operator tracing +
per-channel index-map propagation + slot/unit partitioning). Any module that
parse_model can build and the model can forward is either pruned correctly,
skipped conservatively with a reported reason, or refused with an explanation.

Structured (physical channel removal), unstructured and 2:4 semi-structured
sparsity share the same lifecycle (prune -> finetrain -> val/export). Pruning
criteria are pluggable via criteria.register_importance.
"""

from .api import run_sparse_prune, run_structured_prune
from .criteria import IMPORTANCE_REGISTRY, register_importance
from .engine_v2 import ZeroRegPruneEngine
from .executors import StructuredExecutor, UnstructuredExecutor
from .spec import adapt_model_to_state_dict, factory_self_check, rebuild_pruned_model
from .verify import PruneVerificationError, verify_pruned_model

__all__ = [
    "ZeroRegPruneEngine",
    "StructuredExecutor",
    "UnstructuredExecutor",
    "IMPORTANCE_REGISTRY",
    "register_importance",
    "run_structured_prune",
    "run_sparse_prune",
    "rebuild_pruned_model",
    "adapt_model_to_state_dict",
    "factory_self_check",
    "verify_pruned_model",
    "PruneVerificationError",
]
