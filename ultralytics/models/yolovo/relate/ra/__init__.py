"""RA (RelateAnything) structure modules ported for the YOLOVO relate head.

pooling.SoftSpatialPool   cross-attention region pooling
transformer.RelationTransformer  pair context (self + cross layers)
deformable.DeformableRelRead deformable scene read with null slots
geometry.*                box / scene / pair geometry encoders
"""
from .pooling import SoftSpatialPool
from .transformer import RelationTransformer
from .deformable import DeformableRelRead
from .geometry import BoxPromptEncoder, ScenePosEnc, RelGeomEncoder

__all__ = ["SoftSpatialPool", "RelationTransformer", "DeformableRelRead",
           "BoxPromptEncoder", "ScenePosEnc", "RelGeomEncoder"]
