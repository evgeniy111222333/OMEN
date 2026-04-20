from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .backbone import SemanticGroundingBackbone
from .interlingua import build_canonical_interlingua
from .interlingua_types import CanonicalInterlingua
from .ontology_growth import GroundingOntologyGrowthResult, build_grounding_ontology_growth
from .semantic_scene import build_semantic_scene_graph
from .symbolic_compiler import SymbolicCompilationResult, compile_canonical_interlingua
from .verifier_stack import GroundingVerifierStackResult, run_grounding_verifier_stack
from .verification import GroundingVerificationReport, verify_symbolic_hypotheses
from .world_state_writeback import GroundingWorldStateWriteback, build_grounding_world_state_writeback
from .text_semantics import ground_text_document
from .types import GroundedTextDocument
from .scene_types import SemanticSceneGraph


@dataclass
class TextGroundingPipelineResult:
    document: GroundedTextDocument
    scene: SemanticSceneGraph
    interlingua: CanonicalInterlingua
    compiled: SymbolicCompilationResult
    verification: GroundingVerificationReport
    verifier_stack: GroundingVerifierStackResult
    world_state: GroundingWorldStateWriteback
    ontology: GroundingOntologyGrowthResult


def ground_text_to_symbolic(
    text: str,
    *,
    language: str = "text",
    max_segments: int = 24,
    backbone: Optional[SemanticGroundingBackbone] = None,
    memory_records: Optional[Sequence[object]] = None,
) -> TextGroundingPipelineResult:
    document = ground_text_document(
        text,
        language=language,
        max_segments=max_segments,
    )
    scene = build_semantic_scene_graph(document, backbone=backbone)
    interlingua = build_canonical_interlingua(scene)
    compiled = compile_canonical_interlingua(interlingua, document=document)
    verification = verify_symbolic_hypotheses(
        compiled,
        document=document,
        interlingua=interlingua,
        scene=scene,
    )
    world_state = build_grounding_world_state_writeback(compiled, verification)
    ontology = build_grounding_ontology_growth(document, interlingua)
    verifier_stack = run_grounding_verifier_stack(
        scene=scene,
        verification_records=verification.records,
        world_state_records=world_state.records,
        ontology_concepts=ontology.concepts,
        memory_records=memory_records,
    )
    return TextGroundingPipelineResult(
        document=document,
        scene=scene,
        interlingua=interlingua,
        compiled=compiled,
        verification=verification,
        verifier_stack=verifier_stack,
        world_state=world_state,
        ontology=ontology,
    )
