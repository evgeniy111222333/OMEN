from __future__ import annotations

import zlib
from typing import Any, Dict, FrozenSet, Iterable, List, Set, Tuple

from .scene_types import SemanticSceneGraph


GROUND_MENTION_PRED = 932
GROUND_DISCOURSE_PRED = 933
GROUND_TEMPORAL_PRED = 934
GROUND_EXPLANATION_PRED = 935
GROUND_COREFERENCE_PRED = 936


def _stable_hash(text: str) -> int:
    return int(zlib.adler32(text.encode("utf-8")) & 0x7FFFFFFF)


def _symbol(kind: str, value: str, *, base: int) -> int:
    return int(base + (_stable_hash(f"{kind}:{value}") % 100_000))


def _entity_symbol(value: str) -> int:
    return _symbol("entity", value, base=2_500_000)


def _lex_symbol(value: str) -> int:
    return _symbol("lex", value, base=2_700_000)


def _dedupe_facts(facts: Iterable[Any]) -> Tuple[Any, ...]:
    out: List[Any] = []
    seen: Set[Any] = set()
    for fact in facts:
        if fact in seen:
            continue
        seen.add(fact)
        out.append(fact)
    return tuple(out)


def compile_scene_context_symbolic_atoms(
    scene: SemanticSceneGraph,
) -> Tuple[FrozenSet[Any], Dict[str, float]]:
    from omen_prolog import HornAtom

    facts: List[Any] = []

    for mention in scene.mentions:
        facts.append(HornAtom(GROUND_MENTION_PRED, (_entity_symbol(mention.entity_id), _lex_symbol(mention.surface_form))))
    for relation in scene.discourse_relations:
        facts.append(
            HornAtom(
                GROUND_DISCOURSE_PRED,
                (_lex_symbol(relation.relation_type), _lex_symbol(relation.marker), int(relation.target_segment)),
            )
        )
    for marker in scene.temporal_markers:
        facts.append(
            HornAtom(
                GROUND_TEMPORAL_PRED,
                (_lex_symbol(marker.marker_type), _lex_symbol(marker.marker_value), int(marker.source_segment)),
            )
        )
    for explanation in scene.explanations:
        facts.append(
            HornAtom(
                GROUND_EXPLANATION_PRED,
                (_lex_symbol(explanation.explanation_type), int(explanation.source_segment)),
            )
        )
    for link in scene.coreference_links:
        facts.append(
            HornAtom(
                GROUND_COREFERENCE_PRED,
                (
                    _entity_symbol(link.source_entity_id),
                    _entity_symbol(link.target_entity_id),
                    _lex_symbol(link.relation_type),
                ),
            )
        )

    deduped = frozenset(_dedupe_facts(facts))
    stats = {
        "scene_context_facts": float(len(deduped)),
        "scene_context_mention_facts": float(sum(1 for fact in deduped if getattr(fact, "pred", None) == GROUND_MENTION_PRED)),
        "scene_context_discourse_facts": float(sum(1 for fact in deduped if getattr(fact, "pred", None) == GROUND_DISCOURSE_PRED)),
        "scene_context_temporal_facts": float(sum(1 for fact in deduped if getattr(fact, "pred", None) == GROUND_TEMPORAL_PRED)),
        "scene_context_explanation_facts": float(sum(1 for fact in deduped if getattr(fact, "pred", None) == GROUND_EXPLANATION_PRED)),
        "scene_context_coreference_facts": float(sum(1 for fact in deduped if getattr(fact, "pred", None) == GROUND_COREFERENCE_PRED)),
    }
    return deduped, stats
