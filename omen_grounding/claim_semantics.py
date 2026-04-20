from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

from .text_semantics import normalize_symbol_text
from .types import GroundedStructuralUnit

_QUESTION_RE = re.compile(
    r"(?:\?\s*$)|(?:^(?:who|what|when|where|why|how|can|could|should|is|are|do|does|did)\b)|"
    r"(?:^(?:хто|що|коли|де|чому|як|чи)\b)",
    re.IGNORECASE | re.UNICODE,
)
_HEDGE_RE = re.compile(
    r"\b(?:maybe|perhaps|probably|possibly|likely|apparently|seems|might|may|"
    r"можливо|мабуть|ймовірно|схоже|наче)\b",
    re.IGNORECASE | re.UNICODE,
)
_RULE_LABEL_RE = re.compile(
    r"^\s*(?:rule|policy|principle|\u043f\u0440\u0430\u0432\u0438\u043b\u043e)\b(?:\s*[:=-]\s*|\s+)",
    re.IGNORECASE | re.UNICODE,
)
_OBLIGATION_RE = re.compile(
    r"\b(?:must|should|need(?:s)?(?:\s+to)?|required|ought|"
    r"\u043f\u043e\u0432\u0438\u043d\u0435\u043d|\u043f\u043e\u0432\u0438\u043d\u043d\u0430|"
    r"\u043f\u043e\u0442\u0440\u0456\u0431\u043d\u043e|\u0442\u0440\u0435\u0431\u0430|"
    r"\u043c\u0430\u0454|\u043c\u0430\u044e\u0442\u044c)\b",
    re.IGNORECASE | re.UNICODE,
)
_GENERIC_RE = re.compile(
    r"\b(?:all|every|each|classes?|types?|kinds?|objects?\s+type|"
    r"generic|abstract|result|discussion|conclusion|"
    r"\u0432\u0441\u0456|\u0443\u0441\u0456|\u043a\u043e\u0436\u0435\u043d|"
    r"\u043a\u043e\u0436\u043d\u0430|\u043a\u043e\u0436\u043d\u0456|"
    r"\u043a\u043b\u0430\u0441(?:\u0438|\u0443)?|"
    r"\u0442\u0438\u043f(?:\u0438|\u0443)?|"
    r"\u043e\u0431'\u0454\u043a\u0442\u0438?\s+\u0442\u0438\u043f\u0443)\b",
    re.IGNORECASE | re.UNICODE,
)


@dataclass(frozen=True)
class ClaimSemanticProfile:
    claim_source: str = "document"
    epistemic_status: str = "asserted"
    speaker_name: str = ""
    semantic_mode: str = "instance"
    quantifier_mode: str = "instance"


def _speaker_name_from_unit(unit: GroundedStructuralUnit) -> str:
    for key, value in tuple(getattr(unit, "fields", ()) or ()):
        if str(key or "") != "speaker":
            continue
        normalized = normalize_symbol_text(value)
        if normalized:
            return normalized
        raw = str(value or "").strip()
        if raw:
            return raw
    raw_text = str(getattr(unit, "text", "") or "").strip()
    match = re.match(r"^(user|assistant|speaker|q|a)\s*[:>-]", raw_text, flags=re.IGNORECASE | re.UNICODE)
    if match is None:
        return ""
    normalized = normalize_symbol_text(match.group(1))
    return normalized or match.group(1).strip().casefold()


def infer_claim_semantics(
    text: str,
    *,
    structural_units: Sequence[GroundedStructuralUnit] = (),
) -> ClaimSemanticProfile:
    units = tuple(structural_units or ())
    unit_types = {str(getattr(unit, "unit_type", "") or "") for unit in units}
    speaker_name = next(
        (name for name in (_speaker_name_from_unit(unit) for unit in units) if name),
        "",
    )
    normalized_text = str(text or "").strip()
    if _QUESTION_RE.search(normalized_text):
        epistemic_status = "questioned"
    elif _HEDGE_RE.search(normalized_text):
        epistemic_status = "hedged"
    elif "citation_region" in unit_types:
        epistemic_status = "cited"
    else:
        epistemic_status = "asserted"
    if "citation_region" in unit_types:
        claim_source = "citation_region"
    elif "speaker_turn" in unit_types:
        claim_source = "speaker_turn"
    else:
        claim_source = "document"
    if _RULE_LABEL_RE.search(normalized_text):
        semantic_mode = "rule"
        quantifier_mode = "generic_all"
    elif _OBLIGATION_RE.search(normalized_text):
        semantic_mode = "obligation"
        quantifier_mode = "directive"
    elif "citation_region" in unit_types or _GENERIC_RE.search(normalized_text):
        semantic_mode = "generic"
        quantifier_mode = "generic_all"
    else:
        semantic_mode = "instance"
        quantifier_mode = "instance"
    return ClaimSemanticProfile(
        claim_source=claim_source,
        epistemic_status=epistemic_status,
        speaker_name=speaker_name,
        semantic_mode=semantic_mode,
        quantifier_mode=quantifier_mode,
    )


def is_nonasserted_epistemic_status(value: Optional[str]) -> bool:
    return str(value or "asserted") in {"cited", "questioned", "hedged"}
