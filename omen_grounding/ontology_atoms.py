from __future__ import annotations

import zlib
from typing import Any, Dict, FrozenSet, Iterable, List, Sequence, Set, Tuple

from .ontology_growth import GroundingOntologyConcept


GROUND_ONTOLOGY_CONCEPT_PRED = 929
GROUND_ONTOLOGY_MEMBER_PRED = 930
GROUND_ONTOLOGY_SIGNATURE_PRED = 931


def _stable_hash(text: str) -> int:
    return int(zlib.adler32(text.encode("utf-8")) & 0x7FFFFFFF)


def _symbol(kind: str, value: str, *, base: int) -> int:
    return int(base + (_stable_hash(f"{kind}:{value}") % 100_000))


def _concept_symbol(value: str) -> int:
    return _symbol("concept", value, base=2_100_000)


def _lex_symbol(value: str) -> int:
    return _symbol("lex", value, base=2_300_000)


def _dedupe_facts(facts: Iterable[Any]) -> Tuple[Any, ...]:
    out: List[Any] = []
    seen: Set[Any] = set()
    for fact in facts:
        if fact in seen:
            continue
        seen.add(fact)
        out.append(fact)
    return tuple(out)


def compile_ontology_symbolic_atoms(
    concepts: Sequence[GroundingOntologyConcept],
) -> Tuple[FrozenSet[Any], Dict[str, float]]:
    from omen_prolog import HornAtom

    facts: List[Any] = []
    for concept in concepts:
        if str(concept.world_status or "").strip().lower() not in {"active", "supported"}:
            continue
        concept_sym = _concept_symbol(concept.concept_key)
        facts.append(HornAtom(GROUND_ONTOLOGY_CONCEPT_PRED, (concept_sym,)))
        for member in concept.member_terms:
            facts.append(HornAtom(GROUND_ONTOLOGY_MEMBER_PRED, (concept_sym, _lex_symbol(member))))
        for term in concept.signature_terms:
            facts.append(HornAtom(GROUND_ONTOLOGY_SIGNATURE_PRED, (concept_sym, _lex_symbol(term))))

    deduped = frozenset(_dedupe_facts(facts))
    stats = {
        "grounding_ontology_facts": float(len(deduped)),
        "grounding_ontology_active_concepts": float(
            sum(1 for concept in concepts if str(concept.world_status or "").strip().lower() in {"active", "supported"})
        ),
    }
    return deduped, stats
