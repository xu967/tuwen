# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Module fencing analysis for the zero-registration pruning engine.

Fencing is the defense line against operators the dispatcher cannot see
(pybind CUDA extensions such as mamba's selective_scan_cuda) and against
parameters that no traced op ever touched (training-only branches that even
the train-mode pass could not reach). Fenced modules have every parameter
tainted; this module turns the raw fencing data into a reviewable report.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FenceEntry:
    module: str
    reason: str
    params: list = field(default_factory=list)
    n_elements: int = 0


def summarize_fencing(trace_result, registry) -> list[FenceEntry]:
    """Aggregate fenced modules with the parameters and element counts they lock."""
    entries: dict[str, FenceEntry] = {}
    for mod, why in trace_result.fenced_modules.items():
        entries[mod] = FenceEntry(module=mod or "<top>", reason=why)

    for name in list(trace_result.all_params) + list(trace_result.all_buffers):
        owner = trace_result.param_owner.get(name, "")
        for mod, entry in entries.items():
            if mod == "" and owner == "" or (mod and (owner == mod or owner.startswith(mod + "."))):
                entry.params.append(name)
                dims = registry.param_dims.get(name)
                if isinstance(dims, tuple):
                    entry.n_elements += sum(len(d) for d in dims if d is not None)
                break

    return sorted(entries.values(), key=lambda e: e.module)


def summarize_conservative(propagator) -> list[dict]:
    """Deduplicate the conservative-decision log per (module, op, reason)."""
    seen: dict[tuple, dict] = {}
    for item in propagator.conservative_log:
        key = (item["module"], item["op"], item["reason"])
        entry = seen.setdefault(key, {**item, "count": 0})
        entry["count"] += 1
    return sorted(seen.values(), key=lambda e: (-e["count"], e["module"]))
