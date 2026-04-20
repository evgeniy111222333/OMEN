from __future__ import annotations

import zlib
from typing import Any, Dict, FrozenSet, Iterable, List, Sequence, Set, Tuple

from .scene_types import SemanticSceneGraph


GROUND_ENTITY_PRED = 910
GROUND_ENTITY_TYPE_PRED = 911
GROUND_STATE_PRED = 912
GROUND_RELATION_PRED = 913
GROUND_GOAL_PRED = 914
GROUND_CLAIM_KIND_PRED = 915
GROUND_EVENT_POLARITY_PRED = 916
GROUND_CLAIM_EPISTEMIC_PRED = 917
GROUND_CLAIM_SPEAKER_PRED = 918


def _stable_hash(text: str) -> int:
    return int(zlib.adler32(text.encode("utf-8")) & 0x7FFFFFFF)


def _symbol(kind: str, value: str, *, base: int) -> int:
    return int(base + (_stable_hash(f"{kind}:{value}") % 100_000))


def _entity_symbol(value: str) -> int:
    return _symbol("ent", value, base=1_100_000)


def _lex_symbol(value: str) -> int:
    return _symbol("lex", value, base=1_300_000)


def _claim_symbol(value: str) -> int:
    return _symbol("claim", value, base=1_500_000)


def _dedupe_facts(facts: Iterable[Any]) -> Tuple[Any, ...]:
    out: List[Any] = []
    seen: Set[Any] = set()
    for fact in facts:
        if fact in seen:
            continue
        seen.add(fact)
        out.append(fact)
    return tuple(out)


def compile_scene_symbolic_atoms(
    scene: SemanticSceneGraph,
) -> Tuple[FrozenSet[Any], FrozenSet[Any], Dict[str, float]]:
    from omen_prolog import HornAtom

    facts: List[Any] = []
    target_facts: List[Any] = []

    for entity in scene.entities:
        ent_sym = _entity_symbol(entity.entity_id)
        type_sym = _lex_symbol(entity.semantic_type)
        facts.append(HornAtom(GROUND_ENTITY_PRED, (ent_sym,)))
        facts.append(HornAtom(GROUND_ENTITY_TYPE_PRED, (ent_sym, type_sym)))

    for state in scene.states:
        ent_sym = _entity_symbol(state.key_entity_id)
        key_sym = _lex_symbol(state.key_name)
        value_sym = _lex_symbol(state.value)
        facts.append(HornAtom(GROUND_STATE_PRED, (ent_sym, key_sym, value_sym)))

    for event in scene.events:
        if event.subject_entity_id is None or event.object_entity_id is None:
            continue
        subj_sym = _entity_symbol(event.subject_entity_id)
        rel_sym = _lex_symbol(event.event_type)
        obj_sym = _entity_symbol(event.object_entity_id)
        polarity_sym = _lex_symbol(event.polarity)
        relation_fact = HornAtom(GROUND_RELATION_PRED, (subj_sym, rel_sym, obj_sym))
        facts.append(relation_fact)
        facts.append(HornAtom(GROUND_EVENT_POLARITY_PRED, (subj_sym, rel_sym, polarity_sym)))
        if event.polarity != "negative":
            target_facts.append(relation_fact)

    for goal in scene.goals:
        goal_sym = _lex_symbol(goal.goal_name)
        target_sym = _entity_symbol(goal.target_entity_id or goal.goal_value)
        goal_fact = HornAtom(GROUND_GOAL_PRED, (goal_sym, target_sym))
        facts.append(goal_fact)
        target_facts.append(goal_fact)

    for claim in scene.claims:
        claim_sym = _claim_symbol(claim.claim_id)
        kind_sym = _lex_symbol(claim.claim_kind)
        facts.append(HornAtom(GROUND_CLAIM_KIND_PRED, (claim_sym, kind_sym)))
        epistemic_sym = _lex_symbol(str(getattr(claim, "epistemic_status", "asserted") or "asserted"))
        facts.append(HornAtom(GROUND_CLAIM_EPISTEMIC_PRED, (claim_sym, epistemic_sym)))
        speaker_name = str(getattr(claim, "speaker_name", "") or "")
        if speaker_name:
            facts.append(HornAtom(GROUND_CLAIM_SPEAKER_PRED, (claim_sym, _lex_symbol(speaker_name))))

    deduped_facts = frozenset(_dedupe_facts(facts))
    deduped_targets = frozenset(_dedupe_facts(target_facts))
    stats = {
        "grounding_symbolic_facts": float(len(deduped_facts)),
        "grounding_symbolic_targets": float(len(deduped_targets)),
    }
    return deduped_facts, deduped_targets, stats
