from __future__ import annotations

from typing import Optional, Protocol

from .types import GroundedTextDocument
from .scene_types import SemanticSceneGraph


class SemanticGroundingBackbone(Protocol):
    """Interface for learned or external semantic scene builders."""

    def build_scene_graph(
        self,
        document: GroundedTextDocument,
    ) -> Optional[SemanticSceneGraph]:
        ...
