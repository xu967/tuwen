#!/usr/bin/env python3
"""
YOLO11 → YOLOVO pretrained weight converter.

Converts YOLO11 detection weights to YOLOVO open-vocabulary detection weights
by mapping layer keys with shape verification. Supports all scales (n/s/m/l/x)
and all YOLOVO backbone variants.

Usage:
    python tools/convert_yolo11_to_yolovo.py \
        --source yolo11n.pt \
        --config ultralytics/cfg/models/yolovo/extraction/yolovo11n-c2psa-sefn.yaml \
        --output weights/yolovo11n-c2psa-sefn.pt

Layer mapping rules:
    ┌─────────────────────────┬─────────────────────────┬──────────────────────────┐
    │ Source (YOLO11)         │ Target (YOLOVO)         │ Strategy                 │
    ├─────────────────────────┼─────────────────────────┼──────────────────────────┤
    │ model.{0-9}.*           │ model.{0-9}.*           │ Direct copy              │
    │ model.10.*              │ model.10.*              │ Shape-matched copy       │
    │ model.{11-22}.*         │ model.{15-26}.*         │ Offset by +4             │
    │ model.23.cv2.*          │ model.30.cv2.*          │ Direct copy              │
    │ model.23.dfl.*          │ model.30.dfl.*          │ Direct copy              │
    │ model.23.cv3.{i}.0.0.*  │ model.30.cv_obj.{i}.0.* │ cv3 DWConv → cv_obj DWConv│
    │ model.23.cv3.{i}.0.0.*  │ model.30.cv_embed.{i}.0.*│ cv3 DWConv → cv_embed DWConv│
    │ -                       │ model.{11-14}.* (VCTE)   │ Random init (no source)  │
    │ -                       │ model.{27-29}.* (Mod)    │ Random init (no source)  │
    │ -                       │ model.30.cv_obj.*.1-4    │ Random init (no source)  │
    │ -                       │ model.30.cv_embed.*.1-4  │ Random init (no source)  │
    │ -                       │ model.30.mod_*           │ Random init (no source)  │
    │ -                       │ model.30.logit_*         │ Random init (no source)  │
    └─────────────────────────┴─────────────────────────┴──────────────────────────┘
"""

import argparse
import copy
import re
import sys
from pathlib import Path

import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Standard YOLO11 scale constants
# ---------------------------------------------------------------------------
STANDARD_SCALES = {
    "n": [0.50, 0.25, 1024],
    "s": [0.50, 0.50, 1024],
    "m": [0.50, 1.00, 512],
    "l": [1.00, 1.00, 512],
    "x": [1.00, 1.50, 512],
}


# ---------------------------------------------------------------------------
# Key remapping strategies
# ---------------------------------------------------------------------------

def _is_backbone_direct(src_key: str) -> str | None:
    """model.{0-9}.* stays at model.{0-9}.*"""
    return src_key


def _remap_neck_offset(tgt_key: str) -> str | None:
    """model.{15-26}.* → model.{11-22}.* (offset by −4 into YOLO11 neck)."""
    m = re.match(r"^model\.(1[5-9]|2[0-6])(\..*)$", tgt_key)
    if m:
        old_idx = int(m.group(1)) - 4
        return f"model.{old_idx}{m.group(2)}"
    return None


def _remap_detect_cv2(tgt_key: str) -> str | None:
    """model.30.cv2.* → model.23.cv2.*"""
    if tgt_key.startswith("model.30.cv2."):
        return "model.23.cv2." + tgt_key[len("model.30.cv2."):]
    return None


def _remap_detect_dfl(tgt_key: str) -> str | None:
    """model.30.dfl.* → model.23.dfl.*"""
    if tgt_key.startswith("model.30.dfl."):
        return "model.23.dfl." + tgt_key[len("model.30.dfl."):]
    return None


def _remap_cv3_to_cv_obj(tgt_key: str) -> str | None:
    """model.30.cv_obj.{i}.0.* → model.23.cv3.{i}.0.0.* (first DWConv).

    cv3 internal structure (nested Sequential):
        [0] = Sequential(DWConv(neck_ch→neck_ch), Conv(neck_ch→c3))
        [1] = Sequential(DWConv(c3→c3), Conv(c3→c3))
        [2] = Conv2d(c3→nc)

    cv_obj internal structure (flat Sequential):
        [0] = DWConv(neck_ch→neck_ch)   ← maps from cv3.[i].0.0
        [1] = Conv(neck_ch→c3')         ← intermediate → c3' differs (skip)
        [2] = DWConv(c3'→c3')           ← c3' differs from c3 (skip)
        [3] = Conv(c3'→c3')             ← c3' differs from c3 (skip)
        [4] = Conv2d(c3'→1)            ← output differs (skip)

    Only the first depthwise conv (neck_ch→neck_ch) is guaranteed to match.
    """
    m = re.match(r"^model\.30\.cv_obj\.(\d)\.0(\..*)$", tgt_key)
    if m:
        return f"model.23.cv3.{m.group(1)}.0.0{m.group(2)}"
    return None


def _remap_cv3_to_cv_embed(tgt_key: str) -> str | None:
    """model.30.cv_embed.{i}.0.* → model.23.cv3.{i}.0.0.* (first DWConv).

    Same structural mapping as cv_obj — only the first depthwise conv matches.
    """
    m = re.match(r"^model\.30\.cv_embed\.(\d)\.0(\..*)$", tgt_key)
    if m:
        return f"model.23.cv3.{m.group(1)}.0.0{m.group(2)}"
    return None


# Ordered list of remap strategies (first match wins)
REMAP_STRATEGIES = [
    _is_backbone_direct,
    _remap_neck_offset,
    _remap_detect_cv2,
    _remap_detect_dfl,
    _remap_cv3_to_cv_obj,
    _remap_cv3_to_cv_embed,
]


# ---------------------------------------------------------------------------
# Key remapping
# ---------------------------------------------------------------------------

def find_source_key(tgt_key: str, src_sd: dict, tgt_shape: tuple) -> str | None:
    """Try all remap strategies to find a shape-matching source key.

    Args:
        tgt_key: target state_dict key in YOLOVO model.
        src_sd: source YOLO11 state_dict.
        tgt_shape: expected tensor shape in the target.

    Returns:
        Source key string, or None if no match found.
    """
    for strategy in REMAP_STRATEGIES:
        src_key = strategy(tgt_key)
        if src_key is None:
            continue
        if src_key not in src_sd:
            continue
        if src_sd[src_key].shape == tgt_shape:
            return src_key
    return None


def build_remap_table(src_sd: dict, tgt_sd: dict) -> dict:
    """Build {tgt_key: src_key} mapping table for all transferable weights."""
    remap = {}
    for tgt_key in tgt_sd:
        src_key = find_source_key(tgt_key, src_sd, tgt_sd[tgt_key].shape)
        if src_key is not None:
            remap[tgt_key] = src_key
    return remap


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _count_params(sd: dict) -> int:
    return sum(v.numel() for v in sd.values())


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_yolo11_state_dict(weights_path: str) -> dict:
    """Load YOLO11 checkpoint and return state_dict + metadata."""
    import ultralytics.nn.tasks as t

    torch.serialization.add_safe_globals([t.DetectionModel])

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    model = ckpt.get("model", ckpt)
    if hasattr(model, "state_dict"):
        sd = model.float().state_dict()
    else:
        sd = model

    return {k: v for k, v in sd.items()}  # strip metadata keys like num_batches_tracked


def build_yolovo_state_dict(config_path: str, scale: str) -> dict:
    """Build YOLOVO model from config and return state_dict + model.

    Loads the YAML, injects scale, and passes as a dict so the constructor
    picks up the correct scale.
    """
    import yaml
    from ultralytics.nn.tasks import YOLOVOModel

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Inject standard scales and explicitly set the target scale
    if "scales" not in cfg:
        cfg["scales"] = {}
    for s, vals in STANDARD_SCALES.items():
        if s not in cfg["scales"]:
            cfg["scales"][s] = vals
    cfg["scale"] = scale

    model = YOLOVOModel(cfg, ch=3, nc=80)
    sd = model.state_dict()
    return {k: v for k, v in sd.items()}, model


# ---------------------------------------------------------------------------
# Scale detection
# ---------------------------------------------------------------------------

def detect_scale(source_path: str) -> str:
    """Extract scale from filename: yolo11<n/s/m/l/x>.pt → 'n'/'s'/..."""
    basename = Path(source_path).stem.lower()
    for scale in ["n", "s", "m", "l", "x"]:
        if basename.endswith(f"11{scale}") or basename.endswith(f"11{scale}-cls"):
            return scale
    raise ValueError(
        f"Cannot detect scale from filename '{source_path}'. "
        f"Expected pattern: yolo11{{n,s,m,l,x}}.pt"
    )


def is_config_exists(config_path: str) -> bool:
    return Path(config_path).is_file()


# ---------------------------------------------------------------------------
# Detect head cv3 → cv_obj/cv_embed fine-grained mapping
# ---------------------------------------------------------------------------

def build_cv3_remap_table(src_sd: dict, tgt_sd: dict) -> dict:
    """Additional fine-grained remapping for cv3 intermediates.

    Tries to map remaining unmatched model.30.cv_obj.* and model.30.cv_embed.*
    keys from model.23.cv3.* by structural decomposition (shape permitting).

    Returns {tgt_key: src_key} for any newly matched keys.
    """
    extra = {}

    # cv3 structure per scale i:
    #   cv3.i.0.0.* = DWConv(neck_ch→neck_ch)  →  cv_obj.i.0.* / cv_embed.i.0.*
    #   cv3.i.0.1.* = Conv(neck_ch→c3)         →  (usually shape mismatch)
    #   cv3.i.1.0.* = DWConv(c3→c3)            →  (usually shape mismatch)
    #   cv3.i.1.1.* = Conv(c3→c3)              →  (usually shape mismatch)
    #   cv3.i.2.*   = Conv2d(c3→nc)            →  (always shape mismatch)

    cv3_to_tgt_map = [
        # (cv3_sub_path_template,  tgt_template)
        ("model.23.cv3.{i}.0.0.{suffix}", "model.30.cv_obj.{i}.0.{suffix}"),
        ("model.23.cv3.{i}.0.0.{suffix}", "model.30.cv_embed.{i}.0.{suffix}"),
    ]

    for cv3_tmpl, tgt_tmpl in cv3_to_tgt_map:
        # Scan source for all matching cv3 keys
        cv3_prefix = "model.23.cv3."
        for src_key in src_sd:
            if not src_key.startswith(cv3_prefix):
                continue
            # Parse: model.23.cv3.{i}.0.0.{suffix}
            m = re.match(r"^model\.23\.cv3\.(\d+)\.0\.0\.(.+)$", src_key)
            if not m:
                continue
            scale_idx, suffix = m.group(1), m.group(2)
            tgt_key = f"model.30.cv_obj.{scale_idx}.0.{suffix}"
            if tgt_key in tgt_sd and src_sd[src_key].shape == tgt_sd[tgt_key].shape:
                extra[tgt_key] = src_key
            tgt_key_emb = f"model.30.cv_embed.{scale_idx}.0.{suffix}"
            if tgt_key_emb in tgt_sd and src_sd[src_key].shape == tgt_sd[tgt_key_emb].shape:
                extra[tgt_key_emb] = src_key

    return extra


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(source_path: str, config_path: str, output_path: str) -> dict:
    """Run the full conversion pipeline.

    Returns:
        Statistics dict: {copied, skipped_new, total_copied_params, total_new_params, ...}
    """
    # -- Step 0: detect scale --
    scale = detect_scale(source_path)
    print(f"Detected scale: {scale}")

    # -- Step 1: load source weights --
    print(f"Loading source weights: {source_path}")
    src_sd = load_yolo11_state_dict(source_path)
    print(f"  Source parameters: {_count_params(src_sd):,}")

    # -- Step 2: build target model --
    print(f"Building YOLOVO from: {config_path}")
    tgt_sd, tgt_model = build_yolovo_state_dict(config_path, scale)
    print(f"  Target parameters: {_count_params(tgt_sd):,}")

    # -- Step 3: build key remap table --
    remap = build_remap_table(src_sd, tgt_sd)

    # -- Step 4: fine-grained cv3 remap --
    cv3_remap = build_cv3_remap_table(src_sd, tgt_sd)
    # Override/add to main remap
    remap.update(cv3_remap)

    # -- Step 5: perform weight copy --
    result_sd = {}
    copied_keys = []
    skipped_keys = []

    for tgt_key in tgt_sd:
        if tgt_key in remap:
            result_sd[tgt_key] = src_sd[remap[tgt_key]].clone()
            copied_keys.append(tgt_key)
        else:
            result_sd[tgt_key] = tgt_sd[tgt_key].clone()  # keep random init
            skipped_keys.append(tgt_key)

    # -- Step 6: validate & report --
    # Categorize for reporting
    backbone_copied = sum(1 for k in copied_keys if _is_backbone_or_neck(k))
    neck_copied = sum(1 for k in copied_keys if _is_neck(k))
    detect_copied = sum(1 for k in copied_keys if k.startswith("model.30.") and "cv2" in k)
    dfl_copied = sum(1 for k in copied_keys if "dfl" in k)
    cv_obj_copied = sum(1 for k in copied_keys if "cv_obj" in k)
    cv_embed_copied = sum(1 for k in copied_keys if "cv_embed" in k)
    new_text = sum(1 for k in skipped_keys if any(
        k.startswith(f"model.{i}.") for i in range(11, 15)))
    new_mod = sum(1 for k in skipped_keys if any(
        k.startswith(f"model.{i}.") for i in range(27, 30)))
    new_detect = sum(1 for k in skipped_keys if k.startswith("model.30."))
    new_other = len(skipped_keys) - new_text - new_mod - new_detect

    copied_params = sum(tgt_sd[k].numel() for k in copied_keys)
    skipped_params = sum(tgt_sd[k].numel() for k in skipped_keys)
    total_params = copied_params + skipped_params

    print()
    print("─" * 60)
    print("  Conversion Report")
    print("─" * 60)
    print(f"  Total target keys:              {len(tgt_sd)}")
    print(f"  Copied from source:             {len(copied_keys)}")
    print(f"  Random-initialized (new layers):{len(skipped_keys)}")
    print()
    print(f"  Copied parameters:   {copied_params:>12,}  ({copied_params / total_params * 100:.1f}%)")
    print(f"  New parameters:      {skipped_params:>12,}  ({skipped_params / total_params * 100:.1f}%)")
    print(f"  ─────────────────────────────────────")
    print(f"  Total:               {total_params:>12,}")
    print()
    if detect_copied:
        print(f"  Detect cv2 keys copied:          {detect_copied}")
    if dfl_copied:
        print(f"  DFL keys copied:                 {dfl_copied}")
    if cv_obj_copied:
        print(f"  cv_obj DWConv from cv3:          {cv_obj_copied}")
    if cv_embed_copied:
        print(f"  cv_embed DWConv from cv3:        {cv_embed_copied}")
    print(f"  Text pipeline keys (new):        {new_text}")
    print(f"  NeckModulator keys (new):        {new_mod}")
    print(f"  YOLOVODetect new keys:           {new_detect}")
    if new_other:
        print(f"  Other new keys:                  {new_other}")

    # -- Step 7: save --
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tgt_model.load_state_dict(result_sd, strict=True)
    ckpt = {
        "model": tgt_model,
        "date": torch.tensor(0.0),
        "train_args": {
            "source_weights": str(Path(source_path).resolve()),
            "target_config": str(Path(config_path).resolve()),
            "scale": scale,
            "copied_params": copied_params,
            "total_params": total_params,
            "copied_keys": len(copied_keys),
            "skipped_keys": len(skipped_keys),
        },
    }
    torch.save(ckpt, output_path)
    print(f"\nSaved converted weights to: {output_path}")

    return {
        "copied": len(copied_keys),
        "skipped": len(skipped_keys),
        "copied_params": copied_params,
        "skipped_params": skipped_params,
    }


def _is_backbone_or_neck(key: str) -> bool:
    for i in range(10):
        if key.startswith(f"model.{i}."):
            return True
    for i in range(15, 27):
        if key.startswith(f"model.{i}."):
            return True
    return False


def _is_neck(key: str) -> bool:
    for i in range(15, 27):
        if key.startswith(f"model.{i}."):
            return True
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLO11 pretrained weights to YOLOVO format"
    )
    parser.add_argument(
        "--source", required=True,
        help="Path to YOLO11 weights (e.g. yolo11n.pt)",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to YOLOVO YAML config (e.g.ultralytics/cfg/models/yolovo/yolovo11.yaml)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for converted weights",
    )
    parser.add_argument(
        "--scale",
        help="Scale override (auto-detected from filename by default)",
    )
    args = parser.parse_args()

    if not Path(args.source).is_file():
        print(f"ERROR: source file not found: {args.source}")
        sys.exit(1)
    if not Path(args.config).is_file():
        print(f"ERROR: config file not found: {args.config}")
        sys.exit(1)

    convert(args.source, args.config, args.output)


if __name__ == "__main__":
    main()
