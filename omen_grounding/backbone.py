from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, Tuple

import torch

from .scene_types import SemanticSceneGraph
from .types import GroundedTextDocument


@dataclass(frozen=True)
class GroundingProposal:
    proposal_id: str
    layer: str
    proposal_type: str
    segment_index: int
    surface_form: str
    confidence: float
    authority: str = "proposal"
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class GroundingLayerCarrier:
    layer_name: str
    segment_index: int = -1
    confidence: float = 0.0
    proposal_count: int = 0
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class GroundingBackboneOutputs:
    scene: Optional[SemanticSceneGraph] = None
    l1_typed_perception: Tuple[GroundingLayerCarrier, ...] = field(default_factory=tuple)
    l2_structural_grounding: Tuple[GroundingLayerCarrier, ...] = field(default_factory=tuple)
    l3_linguistic_grounding: Tuple[GroundingLayerCarrier, ...] = field(default_factory=tuple)
    l4_scene_proposals: Tuple[GroundingProposal, ...] = field(default_factory=tuple)
    l5_interlingua_proposals: Tuple[GroundingProposal, ...] = field(default_factory=tuple)
    metadata: Dict[str, float] = field(default_factory=dict)
    tensors: Dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class GroundingSupervisionTargets:
    route_target: int = 0
    structural_targets: Optional[torch.Tensor] = None
    linguistic_target: float = 0.0
    scene_target_counts: Optional[torch.Tensor] = None
    interlingua_target_counts: Optional[torch.Tensor] = None


@dataclass
class GroundingLossBreakdown:
    route_loss: Optional[torch.Tensor] = None
    struct_loss: Optional[torch.Tensor] = None
    ling_loss: Optional[torch.Tensor] = None
    scene_loss: Optional[torch.Tensor] = None
    inter_loss: Optional[torch.Tensor] = None
    total_loss: Optional[torch.Tensor] = None

    def as_metadata(self) -> Dict[str, float]:
        metadata: Dict[str, float] = {}
        for key in (
            "route_loss",
            "struct_loss",
            "ling_loss",
            "scene_loss",
            "inter_loss",
            "total_loss",
        ):
            value = getattr(self, key)
            if value is None:
                continue
            metadata[f"grounding_{key}"] = float(value.detach().cpu().item())
        return metadata


class SemanticGroundingBackbone(Protocol):
    """Interface for learned or external semantic scene builders."""

    def build_scene_graph(
        self,
        document: GroundedTextDocument,
    ) -> Optional[SemanticSceneGraph]:
        ...

    def forward_document(
        self,
        document: GroundedTextDocument,
    ) -> GroundingBackboneOutputs:
        ...

    def build_supervision_targets(
        self,
        document: GroundedTextDocument,
        *,
        teacher_scene: Optional[SemanticSceneGraph] = None,
    ) -> GroundingSupervisionTargets:
        ...

    def compute_grounding_losses(
        self,
        document: GroundedTextDocument,
        *,
        teacher_scene: Optional[SemanticSceneGraph] = None,
        targets: Optional[GroundingSupervisionTargets] = None,
    ) -> GroundingLossBreakdown:
        ...
