from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .backbone import (
    GroundingBackboneOutputs,
    GroundingLayerCarrier,
    GroundingLossBreakdown,
    GroundingProposal,
    GroundingSupervisionTargets,
    SemanticGroundingBackbone,
)
from .heuristic_backbone import HeuristicFallbackSemanticBackbone
from .scene_types import (
    SemanticClaim,
    SemanticCoreferenceLink,
    SemanticEntity,
    SemanticEvent,
    SemanticGoal,
    SemanticSceneGraph,
    SemanticState,
)
from .semantic_context import build_semantic_context_objects
from .types import GroundedTextDocument

_ROUTE_LABELS: Tuple[str, ...] = (
    "unknown",
    "natural_text",
    "structured_text",
    "mixed",
    "code",
)
_ROUTE_TO_INDEX = {label: idx for idx, label in enumerate(_ROUTE_LABELS)}
_PROPOSAL_TYPES: Tuple[str, ...] = ("entity", "event", "goal", "claim", "canonical")
_PROPOSAL_TO_INDEX = {label: idx for idx, label in enumerate(_PROPOSAL_TYPES)}


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _candidate_segment_index(candidate: object) -> int:
    if hasattr(candidate, "source_segment"):
        return int(getattr(candidate, "source_segment", 0) or 0)
    source_segments = tuple(getattr(candidate, "source_segments", ()) or ())
    if source_segments:
        return int(source_segments[0])
    return 0


def _candidate_surface(candidate: object) -> str:
    for attr in (
        "canonical_name",
        "event_type",
        "goal_value",
        "goal_name",
        "predicate",
        "subject_name",
        "object_name",
    ):
        value = getattr(candidate, attr, None)
        if isinstance(value, str) and value:
            return value
    return ""


class LearnedSemanticGroundingBackbone(nn.Module):
    """Trainable L1-L5 proposal backbone bootstrapped from bounded seed proposals."""

    def __init__(
        self,
        *,
        d_model: int = 64,
        max_bytes: int = 128,
        entity_threshold: float = 0.44,
        proposal_threshold: float = 0.46,
        seed_teacher: Optional[SemanticGroundingBackbone] = None,
    ) -> None:
        super().__init__()
        self.max_bytes = max(16, int(max_bytes))
        self.entity_threshold = float(entity_threshold)
        self.proposal_threshold = float(proposal_threshold)
        self.seed_teacher = seed_teacher or HeuristicFallbackSemanticBackbone()

        self.byte_embedding = nn.Embedding(257, d_model)
        self.segment_feature_proj = nn.Linear(15, d_model)
        self.candidate_feature_proj = nn.Linear(10, d_model)
        self.proposal_type_embedding = nn.Embedding(len(_PROPOSAL_TYPES), d_model)
        self.segment_norm = nn.LayerNorm(d_model)
        self.route_head = nn.Linear(d_model, len(_ROUTE_LABELS))
        self.struct_head = nn.Linear(d_model, 1)
        self.ling_head = nn.Linear(d_model, 1)
        self.scene_count_head = nn.Linear(d_model, 4)
        self.inter_count_head = nn.Linear(d_model, 2)
        self.proposal_heads = nn.ModuleDict(
            {name: nn.Linear(d_model, 1) for name in ("entity", "event", "goal", "claim")}
        )
        self._last_outputs: Optional[GroundingBackboneOutputs] = None

        self._reset_bootstrap_heads()

    def _reset_bootstrap_heads(self) -> None:
        for layer in (
            self.route_head,
            self.struct_head,
            self.ling_head,
            self.scene_count_head,
            self.inter_count_head,
            *self.proposal_heads.values(),
        ):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _device(self) -> torch.device:
        return self.byte_embedding.weight.device

    def _segment_feature_tensor(
        self,
        document: GroundedTextDocument,
        segment: object,
    ) -> torch.Tensor:
        routing = getattr(segment, "routing", None) or getattr(document, "routing", None)
        modality = str(getattr(routing, "modality", "unknown") or "unknown")
        features = torch.tensor(
            [
                min(len(str(getattr(segment, "text", "") or "")) / 256.0, 1.0),
                min(len(tuple(getattr(segment, "tokens", ()) or ())) / 64.0, 1.0),
                min(len(tuple(getattr(segment, "structural_units", ()) or ())) / 8.0, 1.0),
                min(len(tuple(getattr(segment, "states", ()) or ())) / 8.0, 1.0),
                min(len(tuple(getattr(segment, "relations", ()) or ())) / 8.0, 1.0),
                min(len(tuple(getattr(segment, "goals", ()) or ())) / 8.0, 1.0),
                min(len(tuple(getattr(segment, "entities", ()) or ())) / 8.0, 1.0),
                min(len(tuple(getattr(segment, "events", ()) or ())) / 8.0, 1.0),
                1.0 if bool(getattr(segment, "counterexample", False)) else 0.0,
                _safe_float(getattr(routing, "confidence", 0.0)),
                _safe_float(getattr(routing, "ambiguity", 0.0)),
                1.0 if modality == "natural_text" else 0.0,
                1.0 if modality == "structured_text" else 0.0,
                1.0 if modality == "mixed" else 0.0,
                _safe_float(getattr(document, "metadata", {}).get("grounding_multilingual", 0.0)),
            ],
            dtype=torch.float32,
            device=self._device(),
        )
        return features

    def _encode_segment(self, document: GroundedTextDocument, segment: object) -> torch.Tensor:
        raw_text = str(getattr(segment, "text", "") or "")
        encoded = list(raw_text.encode("utf-8")[: self.max_bytes])
        if not encoded:
            encoded = [0]
        byte_ids = torch.tensor([value + 1 for value in encoded], dtype=torch.long, device=self._device())
        byte_repr = self.byte_embedding(byte_ids).mean(dim=0)
        feature_repr = self.segment_feature_proj(self._segment_feature_tensor(document, segment))
        return self.segment_norm(byte_repr + feature_repr)

    def _seed_scene(
        self,
        document: GroundedTextDocument,
        *,
        teacher_scene: Optional[SemanticSceneGraph] = None,
    ) -> SemanticSceneGraph:
        if teacher_scene is not None:
            return teacher_scene
        try:
            seeded = self.seed_teacher.build_scene_graph(document)
        except Exception:
            seeded = None
        if seeded is not None:
            return seeded
        return SemanticSceneGraph(language=document.language, source_text=document.source_text, metadata=dict(document.metadata))

    def _proposal_hidden(
        self,
        *,
        proposal_type: str,
        segment_repr: torch.Tensor,
        document_repr: torch.Tensor,
        feature_values: Sequence[float],
    ) -> torch.Tensor:
        type_id = _PROPOSAL_TO_INDEX[proposal_type]
        type_repr = self.proposal_type_embedding(
            torch.tensor(type_id, dtype=torch.long, device=self._device())
        )
        features = torch.tensor(feature_values, dtype=torch.float32, device=self._device())
        candidate_repr = self.candidate_feature_proj(features)
        return self.segment_norm(segment_repr + (0.35 * document_repr) + type_repr + candidate_repr)

    def _proposal_score(
        self,
        candidate: object,
        *,
        proposal_type: str,
        segment_repr: torch.Tensor,
        document_repr: torch.Tensor,
        total_segments: int,
    ) -> Tuple[float, torch.Tensor]:
        status = str(getattr(candidate, "status", "") or "")
        epistemic_status = str(getattr(candidate, "epistemic_status", "asserted") or "asserted")
        polarity = str(getattr(candidate, "polarity", "positive") or "positive")
        feature_values = (
            _safe_float(getattr(candidate, "confidence", 0.5), 0.5),
            1.0 if status == "supported" else 0.0,
            1.0 if status == "proposal" else 0.0,
            1.0 if epistemic_status != "asserted" else 0.0,
            1.0 if polarity != "positive" else 0.0,
            1.0 if getattr(candidate, "speaker_entity_id", None) else 0.0,
            1.0 if getattr(candidate, "modality", "") else 0.0,
            1.0 if getattr(candidate, "condition", "") else 0.0,
            1.0 if getattr(candidate, "explanation", "") else 0.0,
            min(_candidate_segment_index(candidate) / max(total_segments, 1), 1.0),
        )
        hidden = self._proposal_hidden(
            proposal_type=proposal_type,
            segment_repr=segment_repr,
            document_repr=document_repr,
            feature_values=feature_values,
        )
        logit = self.proposal_heads[proposal_type](hidden).squeeze(-1)
        seed_confidence = _safe_float(getattr(candidate, "confidence", 0.5), 0.5)
        score = 0.65 * seed_confidence + 0.35 * float(torch.sigmoid(logit).detach().cpu().item())
        return min(max(score, 0.0), 1.0), logit

    def _retain_minimum(
        self,
        candidates: Sequence[Tuple[object, float]],
    ) -> List[object]:
        retained = [candidate for candidate, score in candidates if score >= self.proposal_threshold]
        if retained or not candidates:
            return retained
        best_candidate = max(candidates, key=lambda item: item[1])[0]
        return [best_candidate]

    def _build_scene_from_seed(
        self,
        document: GroundedTextDocument,
        seed_scene: SemanticSceneGraph,
        *,
        segment_reprs: Dict[int, torch.Tensor],
        document_repr: torch.Tensor,
    ) -> GroundingBackboneOutputs:
        total_segments = max(len(segment_reprs), 1)

        entity_scores: Dict[str, float] = {}
        event_scores: Dict[str, float] = {}
        goal_scores: Dict[str, float] = {}
        claim_scores: Dict[str, float] = {}
        l4_proposals: List[GroundingProposal] = []

        for entity in seed_scene.entities:
            segment_index = _candidate_segment_index(entity)
            segment_repr = segment_reprs.get(segment_index, document_repr)
            score, _logit = self._proposal_score(
                entity,
                proposal_type="entity",
                segment_repr=segment_repr,
                document_repr=document_repr,
                total_segments=total_segments,
            )
            entity_scores[entity.entity_id] = score
            l4_proposals.append(
                GroundingProposal(
                    proposal_id=f"entity:{entity.entity_id}",
                    layer="L4",
                    proposal_type="entity",
                    segment_index=segment_index,
                    surface_form=entity.canonical_name,
                    confidence=score,
                    authority="learned_proposal",
                    metadata={"bootstrap_teacher": 1.0},
                )
            )

        scored_events: List[Tuple[SemanticEvent, float]] = []
        for event in seed_scene.events:
            segment_index = _candidate_segment_index(event)
            segment_repr = segment_reprs.get(segment_index, document_repr)
            score, _logit = self._proposal_score(
                event,
                proposal_type="event",
                segment_repr=segment_repr,
                document_repr=document_repr,
                total_segments=total_segments,
            )
            event_scores[event.event_id] = score
            scored_events.append((event, score))
            l4_proposals.append(
                GroundingProposal(
                    proposal_id=f"event:{event.event_id}",
                    layer="L4",
                    proposal_type="event",
                    segment_index=segment_index,
                    surface_form=event.event_type,
                    confidence=score,
                    authority="learned_proposal",
                    metadata={"bootstrap_teacher": 1.0},
                )
            )

        scored_goals: List[Tuple[SemanticGoal, float]] = []
        for goal in seed_scene.goals:
            segment_index = _candidate_segment_index(goal)
            segment_repr = segment_reprs.get(segment_index, document_repr)
            score, _logit = self._proposal_score(
                goal,
                proposal_type="goal",
                segment_repr=segment_repr,
                document_repr=document_repr,
                total_segments=total_segments,
            )
            goal_scores[goal.goal_id] = score
            scored_goals.append((goal, score))
            l4_proposals.append(
                GroundingProposal(
                    proposal_id=f"goal:{goal.goal_id}",
                    layer="L4",
                    proposal_type="goal",
                    segment_index=segment_index,
                    surface_form=goal.goal_value,
                    confidence=score,
                    authority="learned_proposal",
                    metadata={"bootstrap_teacher": 1.0},
                )
            )

        scored_claims: List[Tuple[SemanticClaim, float]] = []
        for claim in seed_scene.claims:
            segment_index = _candidate_segment_index(claim)
            segment_repr = segment_reprs.get(segment_index, document_repr)
            score, _logit = self._proposal_score(
                claim,
                proposal_type="claim",
                segment_repr=segment_repr,
                document_repr=document_repr,
                total_segments=total_segments,
            )
            claim_scores[claim.claim_id] = score
            scored_claims.append((claim, score))
            l4_proposals.append(
                GroundingProposal(
                    proposal_id=f"claim:{claim.claim_id}",
                    layer="L4",
                    proposal_type="claim",
                    segment_index=segment_index,
                    surface_form=claim.predicate or claim.object_value or "",
                    confidence=score,
                    authority="learned_proposal",
                    metadata={"bootstrap_teacher": 1.0},
                )
            )

        retained_events = self._retain_minimum(scored_events)
        retained_goals = self._retain_minimum(scored_goals)
        retained_event_ids = {event.event_id for event in retained_events}
        retained_goal_ids = {goal.goal_id for goal in retained_goals}
        retained_claims = [
            claim
            for claim, score in scored_claims
            if score >= self.proposal_threshold
            or claim.event_id in retained_event_ids
            or claim.goal_id in retained_goal_ids
        ]
        if not retained_claims and scored_claims:
            retained_claims = [max(scored_claims, key=lambda item: item[1])[0]]

        referenced_entity_ids: Set[str] = set()
        for event in retained_events:
            for entity_id in (
                event.subject_entity_id,
                event.object_entity_id,
                event.agent_entity_id,
                event.patient_entity_id,
            ):
                if entity_id:
                    referenced_entity_ids.add(entity_id)
        for goal in retained_goals:
            if goal.target_entity_id:
                referenced_entity_ids.add(goal.target_entity_id)
        for claim in retained_claims:
            for entity_id in (
                claim.subject_entity_id,
                claim.object_entity_id,
                claim.speaker_entity_id,
            ):
                if entity_id:
                    referenced_entity_ids.add(entity_id)

        retained_entities = [
            entity
            for entity in seed_scene.entities
            if entity_scores.get(entity.entity_id, 0.0) >= self.entity_threshold
            or entity.entity_id in referenced_entity_ids
        ]
        if not retained_entities and seed_scene.entities:
            retained_entities = [
                max(seed_scene.entities, key=lambda item: entity_scores.get(item.entity_id, 0.0))
            ]
        retained_entity_ids = {entity.entity_id for entity in retained_entities}

        retained_states = [
            state
            for state in seed_scene.states
            if state.key_entity_id in retained_entity_ids
            or not retained_entity_ids
        ]
        retained_coreference_links = [
            link
            for link in seed_scene.coreference_links
            if link.source_entity_id in retained_entity_ids and link.target_entity_id in retained_entity_ids
        ]
        mentions, discourse_relations, temporal_markers, explanations = build_semantic_context_objects(
            document,
            tuple(retained_entities),
        )

        l5_proposals: List[GroundingProposal] = []
        for event in retained_events:
            l5_proposals.append(
                GroundingProposal(
                    proposal_id=f"canonical:event:{event.event_id}",
                    layer="L5",
                    proposal_type="canonical",
                    segment_index=int(event.source_segment),
                    surface_form=event.event_type,
                    confidence=event_scores.get(event.event_id, 0.0),
                    authority="learned_proposal",
                    metadata={"canonical_relation": 1.0},
                )
            )
        for goal in retained_goals:
            l5_proposals.append(
                GroundingProposal(
                    proposal_id=f"canonical:goal:{goal.goal_id}",
                    layer="L5",
                    proposal_type="canonical",
                    segment_index=int(goal.source_segment),
                    surface_form=goal.goal_name,
                    confidence=goal_scores.get(goal.goal_id, 0.0),
                    authority="learned_proposal",
                    metadata={"canonical_goal": 1.0},
                )
            )
        for claim in retained_claims:
            if claim.predicate:
                l5_proposals.append(
                    GroundingProposal(
                        proposal_id=f"canonical:claim:{claim.claim_id}",
                        layer="L5",
                        proposal_type="canonical",
                        segment_index=int(claim.source_segment),
                        surface_form=claim.predicate,
                        confidence=claim_scores.get(claim.claim_id, 0.0),
                        authority="learned_proposal",
                        metadata={"canonical_claim": 1.0},
                    )
                )

        metadata = dict(seed_scene.metadata)
        for key, value in list(seed_scene.metadata.items()):
            if key.startswith("scene_fallback_"):
                metadata[key.replace("scene_fallback_", "scene_bootstrap_teacher_")] = float(value)
                metadata[key] = 0.0
        metadata.update(
            {
                "scene_entities": float(len(retained_entities)),
                "scene_states": float(len(retained_states)),
                "scene_events": float(len(retained_events)),
                "scene_goals": float(len(retained_goals)),
                "scene_claims": float(len(retained_claims)),
                "scene_mentions": float(len(mentions)),
                "scene_discourse_relations": float(len(discourse_relations)),
                "scene_temporal_markers": float(len(temporal_markers)),
                "scene_explanations": float(len(explanations)),
                "scene_coreference_links": float(len(retained_coreference_links)),
                "scene_mean_entity_confidence": float(
                    sum(entity_scores.get(entity.entity_id, entity.confidence) for entity in retained_entities)
                    / max(len(retained_entities), 1)
                ),
                "scene_mean_event_confidence": float(
                    sum(event_scores.get(event.event_id, event.confidence) for event in retained_events)
                    / max(len(retained_events), 1)
                ),
                "scene_negative_events": float(sum(1 for event in retained_events if event.polarity != "positive")),
                "scene_event_modalities": float(sum(1 for event in retained_events if event.modality)),
                "scene_event_conditions": float(sum(1 for event in retained_events if event.condition)),
                "scene_event_explanations": float(sum(1 for event in retained_events if event.explanation)),
                "scene_event_temporal_anchors": float(sum(1 for event in retained_events if event.temporal)),
                "scene_claim_attributed": float(sum(1 for claim in retained_claims if claim.speaker_entity_id)),
                "scene_claim_nonasserted": float(
                    sum(1 for claim in retained_claims if str(claim.epistemic_status) != "asserted")
                ),
                "scene_fallback_backbone_active": 0.0,
                "scene_fallback_low_authority": 0.0,
                "scene_backbone_replaceable": 0.0,
                "scene_learned_backbone_active": 1.0,
                "scene_trainable_backbone_active": 1.0,
                "scene_bootstrap_teacher_active": 1.0,
                "scene_bootstrap_teacher_heuristic": 1.0,
                "scene_bootstrap_teacher_retained_ratio": float(
                    (len(retained_events) + len(retained_goals) + len(retained_claims))
                    / max(len(seed_scene.events) + len(seed_scene.goals) + len(seed_scene.claims), 1)
                ),
                "scene_l1_typed_segments": float(len(segment_reprs)),
                "scene_l2_structural_segments": float(len(segment_reprs)),
                "scene_l3_linguistic_segments": float(len(segment_reprs)),
                "scene_l4_learned_proposals": float(len(l4_proposals)),
                "scene_l5_canonical_proposals": float(len(l5_proposals)),
            }
        )
        scene = SemanticSceneGraph(
            language=document.language,
            source_text=document.source_text,
            entities=tuple(retained_entities),
            states=tuple(retained_states),
            events=tuple(retained_events),
            goals=tuple(retained_goals),
            claims=tuple(retained_claims),
            mentions=mentions,
            discourse_relations=discourse_relations,
            temporal_markers=temporal_markers,
            explanations=explanations,
            coreference_links=tuple(retained_coreference_links),
            metadata=metadata,
        )
        return GroundingBackboneOutputs(
            scene=scene,
            l4_scene_proposals=tuple(l4_proposals),
            l5_interlingua_proposals=tuple(l5_proposals),
            metadata=metadata,
        )

    def _forward_document_impl(
        self,
        document: GroundedTextDocument,
        *,
        teacher_scene: Optional[SemanticSceneGraph] = None,
    ) -> GroundingBackboneOutputs:
        segments = tuple(getattr(document, "segments", ()) or ())
        segment_reprs: Dict[int, torch.Tensor] = {
            int(getattr(segment, "index", idx)): self._encode_segment(document, segment)
            for idx, segment in enumerate(segments)
        }
        if segment_reprs:
            stacked_segment_reprs = torch.stack(tuple(segment_reprs.values()), dim=0)
            document_repr = stacked_segment_reprs.mean(dim=0)
        else:
            stacked_segment_reprs = torch.zeros((0, self.byte_embedding.embedding_dim), device=self._device())
            document_repr = torch.zeros((self.byte_embedding.embedding_dim,), device=self._device())

        route_logits = self.route_head(document_repr)
        struct_logits = (
            self.struct_head(stacked_segment_reprs).squeeze(-1)
            if stacked_segment_reprs.numel() > 0
            else torch.zeros((0,), device=self._device())
        )
        ling_logits = (
            self.ling_head(stacked_segment_reprs).squeeze(-1)
            if stacked_segment_reprs.numel() > 0
            else torch.zeros((0,), device=self._device())
        )
        scene_count_pred = F.softplus(self.scene_count_head(document_repr))
        inter_count_pred = F.softplus(self.inter_count_head(document_repr))
        route_probs = torch.softmax(route_logits, dim=0)

        l1_carriers: List[GroundingLayerCarrier] = []
        l2_carriers: List[GroundingLayerCarrier] = []
        l3_carriers: List[GroundingLayerCarrier] = []
        for idx, segment in enumerate(segments):
            segment_index = int(getattr(segment, "index", idx))
            struct_score = (
                float(torch.sigmoid(struct_logits[idx]).detach().cpu().item())
                if idx < int(struct_logits.numel())
                else 0.0
            )
            ling_score = (
                float(torch.sigmoid(ling_logits[idx]).detach().cpu().item())
                if idx < int(ling_logits.numel())
                else 0.0
            )
            l1_carriers.append(
                GroundingLayerCarrier(
                    layer_name="L1",
                    segment_index=segment_index,
                    confidence=float(route_probs.max().detach().cpu().item()),
                    proposal_count=len(tuple(getattr(segment, "tokens", ()) or ())),
                    metadata={
                        "route_unknown": float(route_probs[0].detach().cpu().item()),
                        "route_natural_text": float(route_probs[1].detach().cpu().item()),
                        "route_structured_text": float(route_probs[2].detach().cpu().item()),
                        "route_mixed": float(route_probs[3].detach().cpu().item()),
                        "route_code": float(route_probs[4].detach().cpu().item()),
                    },
                )
            )
            l2_carriers.append(
                GroundingLayerCarrier(
                    layer_name="L2",
                    segment_index=segment_index,
                    confidence=struct_score,
                    proposal_count=len(tuple(getattr(segment, "structural_units", ()) or ())),
                    metadata={"counterexample_segment": 1.0 if bool(getattr(segment, "counterexample", False)) else 0.0},
                )
            )
            l3_carriers.append(
                GroundingLayerCarrier(
                    layer_name="L3",
                    segment_index=segment_index,
                    confidence=ling_score,
                    proposal_count=len(tuple(getattr(segment, "relations", ()) or ()))
                    + len(tuple(getattr(segment, "goals", ()) or ())),
                    metadata={"multilingual_document": _safe_float(document.metadata.get("grounding_multilingual", 0.0))},
                )
            )

        seed_scene = self._seed_scene(document, teacher_scene=teacher_scene)
        outputs = self._build_scene_from_seed(
            document,
            seed_scene,
            segment_reprs=segment_reprs,
            document_repr=document_repr,
        )
        outputs.l1_typed_perception = tuple(l1_carriers)
        outputs.l2_structural_grounding = tuple(l2_carriers)
        outputs.l3_linguistic_grounding = tuple(l3_carriers)
        outputs.metadata.update(
            {
                "grounding_learned_route_entropy": float(
                    (-torch.sum(route_probs * torch.log(route_probs.clamp_min(1e-8)))).detach().cpu().item()
                ),
                "grounding_learned_scene_pred_entities": float(scene_count_pred[0].detach().cpu().item()),
                "grounding_learned_scene_pred_events": float(scene_count_pred[1].detach().cpu().item()),
                "grounding_learned_scene_pred_goals": float(scene_count_pred[2].detach().cpu().item()),
                "grounding_learned_scene_pred_claims": float(scene_count_pred[3].detach().cpu().item()),
                "grounding_learned_inter_pred_relations": float(inter_count_pred[0].detach().cpu().item()),
                "grounding_learned_inter_pred_goal_state": float(inter_count_pred[1].detach().cpu().item()),
                "grounding_learned_struct_mean": float(
                    torch.sigmoid(struct_logits).mean().detach().cpu().item() if struct_logits.numel() else 0.0
                ),
                "grounding_learned_ling_mean": float(
                    torch.sigmoid(ling_logits).mean().detach().cpu().item() if ling_logits.numel() else 0.0
                ),
            }
        )
        outputs.tensors.update(
            {
                "route_logits": route_logits,
                "struct_logits": struct_logits,
                "ling_logits": ling_logits,
                "scene_count_pred": scene_count_pred,
                "inter_count_pred": inter_count_pred,
            }
        )
        self._last_outputs = outputs
        return outputs

    def forward_document(self, document: GroundedTextDocument) -> GroundingBackboneOutputs:
        return self._forward_document_impl(document)

    def build_scene_graph(self, document: GroundedTextDocument) -> Optional[SemanticSceneGraph]:
        with torch.no_grad():
            outputs = self._forward_document_impl(document)
        return outputs.scene

    def build_supervision_targets(
        self,
        document: GroundedTextDocument,
        *,
        teacher_scene: Optional[SemanticSceneGraph] = None,
    ) -> GroundingSupervisionTargets:
        segments = tuple(getattr(document, "segments", ()) or ())
        structural_targets = torch.tensor(
            [
                1.0
                if (
                    tuple(getattr(segment, "structural_units", ()) or ())
                    or tuple(getattr(segment, "states", ()) or ())
                    or tuple(getattr(segment, "relations", ()) or ())
                    or tuple(getattr(segment, "goals", ()) or ())
                )
                else 0.0
                for segment in segments
            ],
            dtype=torch.float32,
            device=self._device(),
        )
        routed_modality = str(getattr(getattr(document, "routing", None), "modality", "unknown") or "unknown")
        seed_scene = self._seed_scene(document, teacher_scene=teacher_scene)
        scene_target_counts = torch.tensor(
            [
                float(len(seed_scene.entities)),
                float(len(seed_scene.events)),
                float(len(seed_scene.goals)),
                float(len(seed_scene.claims)),
            ],
            dtype=torch.float32,
            device=self._device(),
        )
        interlingua_target_counts = torch.tensor(
            [
                float(
                    max(
                        len(seed_scene.events),
                        sum(1 for claim in seed_scene.claims if bool(getattr(claim, "predicate", ""))),
                    )
                ),
                float(len(seed_scene.goals) + len(seed_scene.states)),
            ],
            dtype=torch.float32,
            device=self._device(),
        )
        return GroundingSupervisionTargets(
            route_target=_ROUTE_TO_INDEX.get(routed_modality, 0),
            structural_targets=structural_targets,
            linguistic_target=_safe_float(document.metadata.get("grounding_multilingual", 0.0)),
            scene_target_counts=scene_target_counts,
            interlingua_target_counts=interlingua_target_counts,
        )

    def compute_grounding_losses(
        self,
        document: GroundedTextDocument,
        *,
        teacher_scene: Optional[SemanticSceneGraph] = None,
        targets: Optional[GroundingSupervisionTargets] = None,
    ) -> GroundingLossBreakdown:
        outputs = self._forward_document_impl(document, teacher_scene=teacher_scene)
        targets = targets or self.build_supervision_targets(document, teacher_scene=teacher_scene)

        route_logits = outputs.tensors["route_logits"].unsqueeze(0)
        route_target = torch.tensor([int(targets.route_target)], dtype=torch.long, device=self._device())
        route_loss = F.cross_entropy(route_logits, route_target)

        struct_logits = outputs.tensors["struct_logits"]
        if struct_logits.numel() and targets.structural_targets is not None and targets.structural_targets.numel():
            struct_targets = targets.structural_targets.to(device=self._device(), dtype=torch.float32)
            struct_loss = F.binary_cross_entropy_with_logits(struct_logits, struct_targets)
        else:
            struct_loss = torch.zeros((), device=self._device())

        ling_logits = outputs.tensors["ling_logits"]
        if ling_logits.numel():
            ling_targets = torch.full_like(ling_logits, float(targets.linguistic_target))
            ling_loss = F.binary_cross_entropy_with_logits(ling_logits, ling_targets)
        else:
            ling_loss = torch.zeros((), device=self._device())

        scene_target_counts = (
            targets.scene_target_counts.to(device=self._device(), dtype=torch.float32)
            if targets.scene_target_counts is not None
            else torch.zeros((4,), dtype=torch.float32, device=self._device())
        )
        inter_target_counts = (
            targets.interlingua_target_counts.to(device=self._device(), dtype=torch.float32)
            if targets.interlingua_target_counts is not None
            else torch.zeros((2,), dtype=torch.float32, device=self._device())
        )
        scene_loss = F.mse_loss(outputs.tensors["scene_count_pred"], scene_target_counts)
        inter_loss = F.mse_loss(outputs.tensors["inter_count_pred"], inter_target_counts)
        total_loss = route_loss + struct_loss + ling_loss + scene_loss + inter_loss

        loss_breakdown = GroundingLossBreakdown(
            route_loss=route_loss,
            struct_loss=struct_loss,
            ling_loss=ling_loss,
            scene_loss=scene_loss,
            inter_loss=inter_loss,
            total_loss=total_loss,
        )
        outputs.metadata.update(loss_breakdown.as_metadata())
        self._last_outputs = outputs
        return loss_breakdown


_DEFAULT_LEARNED_BACKBONE: Optional[LearnedSemanticGroundingBackbone] = None


def get_default_learned_grounding_backbone() -> LearnedSemanticGroundingBackbone:
    global _DEFAULT_LEARNED_BACKBONE
    if _DEFAULT_LEARNED_BACKBONE is None:
        _DEFAULT_LEARNED_BACKBONE = LearnedSemanticGroundingBackbone()
    return _DEFAULT_LEARNED_BACKBONE
