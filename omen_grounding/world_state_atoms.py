from __future__ import annotations

import zlib
from typing import Any, Dict, FrozenSet, Iterable, List, Sequence, Set, Tuple

from .world_state_writeback import GroundingWorldStateRecord


GROUND_WORLD_ACTIVE_RELATION_PRED = 917
GROUND_WORLD_ACTIVE_STATE_PRED = 918
GROUND_WORLD_ACTIVE_GOAL_PRED = 919
GROUND_WORLD_HYPOTHETICAL_RELATION_PRED = 920
GROUND_WORLD_HYPOTHETICAL_STATE_PRED = 921
GROUND_WORLD_HYPOTHETICAL_GOAL_PRED = 922
GROUND_WORLD_CONTRADICTED_RELATION_PRED = 923
GROUND_WORLD_CONTRADICTED_STATE_PRED = 924
GROUND_WORLD_CONTRADICTED_GOAL_PRED = 925


def _stable_hash(text: str) -> int:
    return int(zlib.adler32(text.encode("utf-8")) & 0x7FFFFFFF)


def _symbol(kind: str, value: str, *, base: int) -> int:
    return int(base + (_stable_hash(f"{kind}:{value}") % 100_000))


def _entity_symbol(value: str) -> int:
    return _symbol("ent", value, base=1_700_000)


def _lex_symbol(value: str) -> int:
    return _symbol("lex", value, base=1_900_000)


def _dedupe_facts(facts: Iterable[Any]) -> Tuple[Any, ...]:
    out: List[Any] = []
    seen: Set[Any] = set()
    for fact in facts:
        if fact in seen:
            continue
        seen.add(fact)
        out.append(fact)
    return tuple(out)


def _predicate_for_record(record: GroundingWorldStateRecord) -> int | None:
    status = str(record.world_status or "").strip().lower()
    record_type = str(record.record_type or "").strip().lower()
    mapping = {
        ("active", "relation"): GROUND_WORLD_ACTIVE_RELATION_PRED,
        ("active", "state"): GROUND_WORLD_ACTIVE_STATE_PRED,
        ("active", "goal"): GROUND_WORLD_ACTIVE_GOAL_PRED,
        ("hypothetical", "relation"): GROUND_WORLD_HYPOTHETICAL_RELATION_PRED,
        ("hypothetical", "state"): GROUND_WORLD_HYPOTHETICAL_STATE_PRED,
        ("hypothetical", "goal"): GROUND_WORLD_HYPOTHETICAL_GOAL_PRED,
        ("contradicted", "relation"): GROUND_WORLD_CONTRADICTED_RELATION_PRED,
        ("contradicted", "state"): GROUND_WORLD_CONTRADICTED_STATE_PRED,
        ("contradicted", "goal"): GROUND_WORLD_CONTRADICTED_GOAL_PRED,
    }
    return mapping.get((status, record_type))


def _fact_args_for_record(record: GroundingWorldStateRecord) -> Tuple[int, ...]:
    symbols = tuple(str(item) for item in record.symbols)
    record_type = str(record.record_type or "").strip().lower()
    if record_type == "relation" and len(symbols) >= 3:
        return (_entity_symbol(symbols[0]), _lex_symbol(symbols[1]), _entity_symbol(symbols[2]))
    if record_type == "state" and len(symbols) >= 2:
        return (_entity_symbol(symbols[0]), _lex_symbol(symbols[1]))
    if record_type == "goal" and len(symbols) >= 2:
        target = symbols[2] if len(symbols) >= 3 else symbols[1]
        return (_lex_symbol(symbols[0]), _entity_symbol(target))
    return tuple()


def compile_world_state_symbolic_atoms(
    records: Sequence[GroundingWorldStateRecord],
) -> Tuple[FrozenSet[Any], FrozenSet[Any], FrozenSet[Any], Dict[str, float]]:
    from omen_prolog import HornAtom

    active_facts: List[Any] = []
    hypothetical_facts: List[Any] = []
    contradicted_facts: List[Any] = []

    for record in records:
        predicate = _predicate_for_record(record)
        args = _fact_args_for_record(record)
        if predicate is None or not args:
            continue
        atom = HornAtom(predicate, args)
        if record.world_status == "active":
            active_facts.append(atom)
        elif record.world_status == "hypothetical":
            hypothetical_facts.append(atom)
        elif record.world_status == "contradicted":
            contradicted_facts.append(atom)

    active = frozenset(_dedupe_facts(active_facts))
    hypothetical = frozenset(_dedupe_facts(hypothetical_facts))
    contradicted = frozenset(_dedupe_facts(contradicted_facts))
    stats = {
        "grounding_world_state_active_facts": float(len(active)),
        "grounding_world_state_hypothetical_facts": float(len(hypothetical)),
        "grounding_world_state_contradicted_facts": float(len(contradicted)),
    }
    return active, hypothetical, contradicted, stats
