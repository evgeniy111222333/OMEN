from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .backbone import SemanticGroundingBackbone
from .claim_semantics import infer_claim_semantics
from .heuristic_policy import append_heuristic_evidence
from .semantic_context import build_semantic_context_objects
from .scene_types import (
    SemanticClaim,
    SemanticCoreferenceLink,
    SemanticEntity,
    SemanticEvent,
    SemanticGoal,
    SemanticSceneGraph,
    SemanticState,
)
from .text_semantics import (
    extract_structured_pairs,
    is_counterexample_text,
    normalize_symbol_text,
)
from .types import (
    GroundedEntityHint,
    GroundedEventHint,
    GroundedGoalHint,
    GroundedRelationHint,
    GroundedTextDocument,
)


_QUOTE_RE = re.compile(r'["“”«»„]([^"“”«»„]+)["“”«»„]', re.UNICODE)
_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b", re.UNICODE)

_GOAL_KEYS = frozenset(
    {
        "goal",
        "target",
        "desired",
        "expect",
        "expected",
        "aim",
        "objective",
        "мета",
        "ціль",
        "завдання",
    }
)

_FOCUS_WRAPPER_TOKENS = frozenset(
    {
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "object",
        "objects",
        "type",
        "types",
        "class",
        "classes",
        "entity",
        "entities",
        "обєкт",
        "обєкти",
        "об'єкт",
        "об'єкти",
        "тип",
        "типу",
        "клас",
        "класу",
        "факт",
        "правило",
        "крок",
        "step",
        "rule",
        "fact",
        "then",
        "later",
        "after",
        "before",
        "because",
        "if",
        "when",
        "потім",
        "далі",
        "після",
        "перед",
        "бо",
        "тому",
        "що",
        "якщо",
        "коли",
    }
)

_PRONOUN_TOKENS = frozenset(
    {
        "it",
        "they",
        "them",
        "he",
        "she",
        "him",
        "her",
        "this",
        "that",
        "these",
        "those",
        "це",
        "він",
        "вона",
        "вони",
        "його",
        "її",
        "їх",
        "цей",
        "ця",
        "ці",
        "те",
    }
)

_PERSON_LIKE_TOKENS = frozenset(
    {
        "user",
        "dispatcher",
        "operator",
        "assistant",
        "agent",
        "bob",
        "alice",
        "користувач",
        "диспетчер",
        "оператор",
        "асистент",
        "агент",
        "боб",
        "аліса",
    }
)

_ANAPHOR_TYPES = {"anaphor"}
_ANAPHOR_TOKENS = set(_PRONOUN_TOKENS)

_GOAL_MARKERS: Tuple[Tuple[str, float], ...] = (
    ("expected result", 0.66),
    ("expected", 0.62),
    ("goal", 0.60),
    ("target", 0.60),
    ("desired", 0.58),
    ("objective", 0.58),
    ("need", 0.56),
    ("must", 0.54),
    ("should", 0.54),
    ("want", 0.52),
    ("aim", 0.52),
    ("мета", 0.62),
    ("ціль", 0.62),
    ("завдання", 0.60),
    ("очікується", 0.60),
    ("очікуваний результат", 0.66),
    ("потрібно", 0.58),
    ("треба", 0.58),
    ("має", 0.54),
    ("повинен", 0.58),
    ("повинна", 0.58),
)

_RELATION_MARKERS: Tuple[Tuple[str, str, float], ...] = (
    ("results in", "causes", 0.74),
    ("leads to", "causes", 0.74),
    ("because", "causes", 0.68),
    ("triggers", "causes", 0.72),
    ("trigger", "causes", 0.70),
    ("causes", "causes", 0.72),
    ("cause", "causes", 0.70),
    ("contains", "contains", 0.68),
    ("becomes", "becomes", 0.68),
    ("become", "becomes", 0.66),
    ("generates", "generates", 0.78),
    ("generate", "generates", 0.76),
    ("creates", "generates", 0.76),
    ("create", "generates", 0.74),
    ("opens", "opens", 0.74),
    ("open", "opens", 0.72),
    ("generating", "generates", 0.74),
    ("creating", "generates", 0.74),
    ("opening", "opens", 0.72),
    ("uses", "uses", 0.64),
    ("use", "uses", 0.62),
    ("has", "has", 0.58),
    ("have", "has", 0.58),
    ("were", "is", 0.54),
    ("was", "is", 0.54),
    ("are", "is", 0.54),
    ("is", "is", 0.54),
    ("призводить до", "causes", 0.76),
    ("веде до", "causes", 0.74),
    ("бо", "causes", 0.66),
    ("тому що", "causes", 0.68),
    ("через", "causes", 0.62),
    ("спричиняє", "causes", 0.76),
    ("викликає", "causes", 0.74),
    ("містить", "contains", 0.68),
    ("стає", "becomes", 0.66),
    ("перетворюється на", "becomes", 0.70),
    ("генерує", "generates", 0.80),
    ("генерують", "generates", 0.80),
    ("генерувати", "generates", 0.76),
    ("створює", "generates", 0.78),
    ("створюють", "generates", 0.78),
    ("створювати", "generates", 0.76),
    ("відчиняє", "opens", 0.78),
    ("відчиняють", "opens", 0.78),
    ("відчиняти", "opens", 0.74),
    ("відкриває", "opens", 0.78),
    ("відкривають", "opens", 0.78),
    ("відкривати", "opens", 0.74),
    ("використовує", "uses", 0.68),
    ("використовують", "uses", 0.68),
    ("використовувати", "uses", 0.64),
    ("має", "has", 0.60),
    ("мають", "has", 0.60),
    ("є", "is", 0.58),
)

_EVENT_VERBS = _RELATION_MARKERS

_MODAL_MARKERS: Tuple[Tuple[str, str], ...] = (
    ("must", "must"),
    ("should", "should"),
    ("need to", "need"),
    ("needs", "need"),
    ("required", "must"),
    ("expected", "expected"),
    ("треба", "need"),
    ("потрібно", "need"),
    ("повинен", "must"),
    ("повинна", "must"),
    ("має", "must"),
    ("очікується", "expected"),
)

_CONDITION_MARKERS: Tuple[str, ...] = (
    "if",
    "when",
    "unless",
    "provided",
    "якщо",
    "коли",
    "за умови",
)

_EXPLANATION_MARKERS: Tuple[str, ...] = (
    "because",
    "since",
    "due to",
    "бо",
    "тому що",
    "оскільки",
    "через",
)

_TEMPORAL_MARKERS: Tuple[Tuple[str, str], ...] = (
    ("before", "before"),
    ("after", "after"),
    ("then", "sequence"),
    ("later", "sequence"),
    ("minute", "duration"),
    ("hour", "duration"),
    ("перед", "before"),
    ("після", "after"),
    ("потім", "sequence"),
    ("далі", "sequence"),
    ("через", "duration"),
    ("хвилин", "duration"),
    ("годин", "duration"),
)

_RELATION_PATTERNS: Tuple[Tuple[re.Pattern[str], str, float], ...]
_EVENT_PATTERNS: Tuple[Tuple[re.Pattern[str], str, float], ...]
_GOAL_PATTERNS: Tuple[Tuple[re.Pattern[str], float], ...]


def _marker_regex(marker: str) -> re.Pattern[str]:
    escaped = re.escape(marker)
    if any(ch.isalnum() for ch in marker):
        return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE | re.UNICODE)
    return re.compile(escaped, re.IGNORECASE | re.UNICODE)


_RELATION_PATTERNS = tuple(
    (_marker_regex(marker), canonical, confidence)
    for marker, canonical, confidence in sorted(_RELATION_MARKERS, key=lambda item: len(item[0]), reverse=True)
)

_EVENT_PATTERNS = tuple(
    (
        re.compile(
            rf"(?P<subject>.+?)\s+(?P<verb>{re.escape(marker)})\s+(?P<object>.+)",
            re.IGNORECASE | re.UNICODE,
        ),
        canonical,
        confidence,
    )
    for marker, canonical, confidence in sorted(_EVENT_VERBS, key=lambda item: len(item[0]), reverse=True)
)

_GOAL_PATTERNS = tuple(
    (_marker_regex(marker), confidence)
    for marker, confidence in sorted(_GOAL_MARKERS, key=lambda item: len(item[0]), reverse=True)
)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalized(value: Optional[str]) -> str:
    return normalize_symbol_text(value) or ""


def _tokenize_words(segment: str, *, limit: Optional[int] = None) -> List[str]:
    tokens: List[str] = []
    seen: Set[str] = set()
    for raw in re.findall(r"[^\W\d_](?:[\w'’`.-]*[^\W_])?", str(segment or ""), re.UNICODE):
        normalized = normalize_symbol_text(raw)
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        tokens.append(normalized)
        if limit is not None and len(tokens) >= int(limit):
            break
    return tokens


def _strip_wrapping_quotes(text: str) -> str:
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] in "\"«“”„" and stripped[-1] in "\"»“”„":
        return stripped[1:-1].strip()
    return stripped


def _quoted_focus(fragment: str, *, left: bool) -> Optional[str]:
    matches = [normalize_symbol_text(match) for match in _QUOTE_RE.findall(str(fragment or ""))]
    matches = [match for match in matches if match]
    if not matches:
        return None
    return matches[-1] if left else matches[0]


def _focus_symbol(fragment: str, *, left: bool) -> Optional[str]:
    quoted = _quoted_focus(fragment, left=left)
    if quoted is not None:
        return quoted
    tokens = _tokenize_words(fragment)
    filtered = [token for token in tokens if token not in _FOCUS_WRAPPER_TOKENS]
    tokens = filtered or tokens
    if not tokens:
        return None
    phrase = "_".join(tokens[-3:] if left else tokens[:3])
    return normalize_symbol_text(phrase)


def _first_marker_focus(segment: str, markers: Sequence[str]) -> str:
    normalized = str(segment or "")
    for marker in markers:
        match = _marker_regex(marker).search(normalized)
        if match is None:
            continue
        focus = _focus_symbol(normalized[match.end() :], left=False)
        if focus:
            return focus
    return ""


def _extract_modality(segment: str) -> str:
    normalized = str(segment or "").casefold()
    for marker, canonical in _MODAL_MARKERS:
        if marker.casefold() in normalized:
            return canonical
    return ""


def _extract_condition(segment: str) -> str:
    return _first_marker_focus(segment, _CONDITION_MARKERS)


def _extract_explanation(segment: str) -> str:
    return _first_marker_focus(segment, _EXPLANATION_MARKERS)


def _extract_temporal(segment: str) -> str:
    normalized = str(segment or "")
    timestamp = _TIME_RE.search(normalized)
    if timestamp is not None:
        return normalize_symbol_text(timestamp.group(0)) or ""
    lowered = normalized.casefold()
    for marker, _marker_type in _TEMPORAL_MARKERS:
        if marker.casefold() in lowered:
            return normalize_symbol_text(marker) or ""
    return ""


def _segment_evidence_refs(segment) -> Tuple[str, ...]:
    refs: List[str] = []
    for unit in tuple(getattr(segment, "structural_units", ()) or ()):
        unit_type = str(getattr(unit, "unit_type", "") or "")
        refs.append(f"structural_unit:{getattr(unit, 'unit_id', '')}")
        refs.append(f"unit_type:{unit_type}")
        refs.extend(
            str(reference)
            for reference in tuple(getattr(unit, "references", ()) or ())
            if str(reference).strip()
        )
    return tuple(dict.fromkeys(item for item in refs if item))


def _relation_candidates_from_verbs(segment: str, *, span=None) -> List[GroundedRelationHint]:
    relations: List[GroundedRelationHint] = []
    seen: Set[Tuple[str, str, str]] = set()
    for pattern, canonical, confidence in _EVENT_PATTERNS:
        for match in pattern.finditer(str(segment or "")):
            left = _focus_symbol(match.group("subject"), left=True)
            right = _focus_symbol(match.group("object"), left=False)
            if left is None or right is None:
                continue
            relation = (left, canonical, right)
            if relation in seen:
                continue
            seen.add(relation)
            relations.append(
                GroundedRelationHint(
                    left=left,
                    relation=canonical,
                    right=right,
                    confidence=confidence,
                    source="heuristic_backbone_verb_clause",
                    status="hint",
                    span=span,
                )
            )
    return relations


def extract_relation_hints(segment: str, *, span=None) -> List[GroundedRelationHint]:
    relations: List[GroundedRelationHint] = []
    seen: Set[Tuple[str, str, str]] = set()
    normalized = str(segment or "")
    for pattern, canonical, confidence in _RELATION_PATTERNS:
        for match in pattern.finditer(normalized):
            left = _focus_symbol(normalized[: match.start()], left=True)
            right = _focus_symbol(normalized[match.end() :], left=False)
            if left is None or right is None:
                continue
            relation = (left, canonical, right)
            if relation in seen:
                continue
            seen.add(relation)
            relations.append(
                GroundedRelationHint(
                    left=left,
                    relation=canonical,
                    right=right,
                    confidence=confidence,
                    source="heuristic_backbone_relation_marker",
                    status="hint",
                    span=span,
                )
            )
    for relation in _relation_candidates_from_verbs(normalized, span=span):
        key = (relation.left, relation.relation, relation.right)
        if key in seen:
            continue
        seen.add(key)
        relations.append(relation)
    chain_parts = [
        _focus_symbol(part, left=False)
        for part in re.split(r"\s*(?:->|=>|then|потім|далі)\s*", normalized, flags=re.IGNORECASE | re.UNICODE)
        if part.strip()
    ]
    chain_parts = [part for part in chain_parts if part]
    if len(chain_parts) >= 2:
        for left, right in zip(chain_parts, chain_parts[1:]):
            relation = (left, "transition", right)
            if relation in seen:
                continue
            seen.add(relation)
            relations.append(
                GroundedRelationHint(
                    left=left,
                    relation="transition",
                    right=right,
                    confidence=0.62,
                    source="heuristic_backbone_transition_chain",
                    status="hint",
                    span=span,
                )
            )
    return relations


def extract_goal_hints(
    segment: str,
    *,
    structured_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    span=None,
) -> List[GroundedGoalHint]:
    goals: List[GroundedGoalHint] = []
    seen: Set[Tuple[str, str]] = set()
    normalized = str(segment or "")
    for key, value in list(structured_pairs or extract_structured_pairs(normalized)):
        if key not in _GOAL_KEYS:
            continue
        pair = (key, value)
        if pair in seen:
            continue
        seen.add(pair)
        goals.append(
            GroundedGoalHint(
                goal_name=key,
                goal_value=value,
                confidence=0.66,
                source="heuristic_backbone_structured_goal",
                status="hint",
                span=span,
            )
        )
    for pattern, confidence in _GOAL_PATTERNS:
        match = pattern.search(normalized)
        if match is None:
            continue
        goal_value = _focus_symbol(normalized[match.end() :], left=False)
        goal_name = normalize_symbol_text(match.group(0))
        if goal_name is None or goal_value is None:
            continue
        pair = (goal_name, goal_value)
        if pair in seen:
            continue
        seen.add(pair)
        goals.append(
            GroundedGoalHint(
                goal_name=goal_name,
                goal_value=goal_value,
                confidence=confidence,
                source="heuristic_backbone_goal_marker",
                status="hint",
                span=span,
            )
        )
    return goals


def _infer_entity_type(name: str) -> str:
    normalized = normalize_symbol_text(name) or ""
    if not normalized:
        return "entity"
    if normalized in _PRONOUN_TOKENS:
        return "anaphor"
    if normalized in _PERSON_LIKE_TOKENS:
        return "agent"
    raw = _strip_wrapping_quotes(str(name or "")).strip()
    if _TIME_RE.fullmatch(raw) or (normalized.count("_") >= 1 and normalized.replace("_", "").isdigit()):
        return "time"
    return "entity"


def extract_entity_hints(
    segment: str,
    *,
    structured_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    relations: Optional[Sequence[GroundedRelationHint]] = None,
    goals: Optional[Sequence[GroundedGoalHint]] = None,
    span=None,
) -> List[GroundedEntityHint]:
    entities: List[GroundedEntityHint] = []
    seen: Set[str] = set()
    relation_hints = list(relations or extract_relation_hints(segment, span=span))
    goal_hints = list(goals or extract_goal_hints(segment, structured_pairs=structured_pairs, span=span))
    for key, value in list(structured_pairs or extract_structured_pairs(segment)):
        for name in (key, value):
            normalized = normalize_symbol_text(name)
            if normalized is None or normalized in seen:
                continue
            seen.add(normalized)
            entities.append(
                GroundedEntityHint(
                    name=normalized,
                    semantic_type=_infer_entity_type(normalized),
                    confidence=0.58,
                    source="heuristic_backbone_structured_entity",
                    status="hint",
                    span=span,
                )
            )
    for relation in relation_hints:
        for name in (relation.left, relation.right):
            normalized = normalize_symbol_text(name)
            if normalized is None or normalized in seen:
                continue
            seen.add(normalized)
            entities.append(
                GroundedEntityHint(
                    name=normalized,
                    semantic_type=_infer_entity_type(normalized),
                    confidence=max(0.56, float(relation.confidence) * 0.92),
                    source="heuristic_backbone_relation_entity",
                    status="hint",
                    span=span,
                )
            )
    for goal in goal_hints:
        normalized = normalize_symbol_text(goal.goal_value)
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        entities.append(
            GroundedEntityHint(
                name=normalized,
                semantic_type=_infer_entity_type(normalized),
                confidence=max(0.56, float(goal.confidence) * 0.92),
                source="heuristic_backbone_goal_entity",
                status="hint",
                span=span,
            )
        )
    for quoted in _QUOTE_RE.findall(str(segment or "")):
        normalized = normalize_symbol_text(quoted)
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        entities.append(
            GroundedEntityHint(
                name=normalized,
                semantic_type=_infer_entity_type(normalized),
                confidence=0.62,
                source="heuristic_backbone_quoted_span",
                status="hint",
                span=span,
            )
        )
    return entities


def extract_event_hints(
    segment: str,
    *,
    relations: Optional[Sequence[GroundedRelationHint]] = None,
    span=None,
) -> List[GroundedEventHint]:
    relation_hints = list(relations or extract_relation_hints(segment, span=span))
    modality = _extract_modality(segment)
    condition = _extract_condition(segment)
    explanation = _extract_explanation(segment)
    temporal = _extract_temporal(segment)
    counterexample = is_counterexample_text(segment)
    events: List[GroundedEventHint] = []
    seen: Set[Tuple[str, str, str, str, str, str, str]] = set()
    for relation in relation_hints:
        key = (
            relation.left,
            relation.relation,
            relation.right,
            modality,
            condition,
            explanation,
            temporal,
        )
        if key in seen:
            continue
        seen.add(key)
        confidence = min(
            0.92,
            float(relation.confidence)
            + (0.04 if modality else 0.0)
            + (0.04 if condition else 0.0)
            + (0.05 if explanation else 0.0)
            + (0.03 if temporal else 0.0),
        )
        events.append(
            GroundedEventHint(
                subject=relation.left,
                predicate=relation.relation,
                object_name=relation.right,
                agent=relation.left,
                patient=relation.right,
                modality=modality,
                condition=condition,
                explanation=explanation,
                temporal=temporal,
                confidence=max(0.56, confidence - (0.08 if counterexample else 0.0)),
                source=relation.source,
                status="hint",
                span=span,
            )
        )
    return events


class _EntityAccumulator:
    def __init__(self, canonical_name: str, *, semantic_type: str = "entity") -> None:
        self.canonical_name = canonical_name
        self.semantic_type = semantic_type or "entity"
        self.aliases: Set[str] = {canonical_name}
        self.source_segments: Set[int] = set()
        self.source_spans = []
        self.confidences: List[float] = []
        self.status = "candidate"

    def add(
        self,
        *,
        alias: Optional[str],
        semantic_type: Optional[str],
        source_segment: int,
        source_span,
        confidence: float,
        status: str,
    ) -> None:
        if alias:
            self.aliases.add(alias)
        if semantic_type and semantic_type not in _ANAPHOR_TYPES:
            self.semantic_type = semantic_type
        self.source_segments.add(int(source_segment))
        if source_span is not None:
            self.source_spans.append(source_span)
        self.confidences.append(float(confidence))
        if self.status == "candidate" and status:
            self.status = "hint"

    def finalize(self, entity_id: str) -> SemanticEntity:
        mean_conf = sum(self.confidences) / max(len(self.confidences), 1)
        return SemanticEntity(
            entity_id=entity_id,
            canonical_name=self.canonical_name,
            semantic_type=self.semantic_type,
            aliases=tuple(sorted(self.aliases)),
            source_segments=tuple(sorted(self.source_segments)),
            source_spans=tuple(self.source_spans),
            confidence=float(mean_conf),
            status=self.status,
        )


def _is_anaphor(name: Optional[str], semantic_type: Optional[str] = None) -> bool:
    normalized = _normalized(name)
    return bool(normalized and (semantic_type in _ANAPHOR_TYPES or normalized in _ANAPHOR_TOKENS))


class HeuristicFallbackSemanticBackbone(SemanticGroundingBackbone):
    """Replaceable low-authority natural-language fallback scene builder."""

    def build_scene_graph(
        self,
        document: GroundedTextDocument,
    ) -> Optional[SemanticSceneGraph]:
        entity_acc: Dict[str, _EntityAccumulator] = {}
        entity_ids: Dict[str, str] = {}
        alias_index: Dict[str, str] = {}
        states: List[SemanticState] = []
        events: List[SemanticEvent] = []
        goals: List[SemanticGoal] = []
        claims: List[SemanticClaim] = []
        coreference_links: List[SemanticCoreferenceLink] = []
        last_salient_entity_id: Optional[str] = None
        relation_proposals = 0.0
        goal_proposals = 0.0
        entity_proposals = 0.0
        event_proposals = 0.0

        def ensure_entity(
            name: str,
            *,
            semantic_type: str = "entity",
            source_segment: int,
            source_span,
            confidence: float,
            status: str,
            alias: Optional[str] = None,
        ) -> str:
            normalized = _normalized(name)
            if not normalized:
                normalized = _normalized(alias) or f"anon:{len(entity_ids)}"
            entity_id = alias_index.get(normalized)
            if entity_id is None:
                entity_id = f"ent:{len(entity_ids)}:{normalized}"
                entity_ids[normalized] = entity_id
                entity_acc[entity_id] = _EntityAccumulator(normalized, semantic_type=semantic_type)
                alias_index[normalized] = entity_id
            accumulator = entity_acc[entity_id]
            accumulator.add(
                alias=alias or normalized,
                semantic_type=semantic_type,
                source_segment=source_segment,
                source_span=source_span,
                confidence=confidence,
                status=status,
            )
            alias_index[_normalized(alias or normalized)] = entity_id
            return entity_id

        def resolve_entity(
            name: Optional[str],
            *,
            semantic_type: str = "entity",
            source_segment: int,
            source_span,
            confidence: float,
            status: str,
        ) -> Optional[str]:
            nonlocal last_salient_entity_id
            normalized = _normalized(name)
            if not normalized:
                return None
            if _is_anaphor(normalized, semantic_type):
                if last_salient_entity_id is None:
                    return None
                coreference_links.append(
                    SemanticCoreferenceLink(
                        link_id=f"coref:{source_segment}:{normalized}:{len(coreference_links)}",
                        source_entity_id=last_salient_entity_id,
                        target_entity_id=last_salient_entity_id,
                        source_segment=source_segment,
                        target_segment=source_segment,
                        relation_type="anaphor_resolution",
                        source_span=source_span,
                        confidence=max(0.54, min(float(confidence), 0.82)),
                        status="hint",
                    )
                )
                return last_salient_entity_id
            entity_id = ensure_entity(
                normalized,
                semantic_type=semantic_type,
                source_segment=source_segment,
                source_span=source_span,
                confidence=confidence,
                status=status,
                alias=name,
            )
            if semantic_type not in _ANAPHOR_TYPES:
                last_salient_entity_id = entity_id
            return entity_id

        for segment in document.segments:
            seg_idx = int(segment.index)
            segment_entity_ids: List[str] = []
            segment_salient_entity_id: Optional[str] = None
            claim_profile = infer_claim_semantics(
                segment.text,
                structural_units=tuple(getattr(segment, "structural_units", ()) or ()),
            )
            segment_claim_evidence = append_heuristic_evidence(
                _segment_evidence_refs(segment),
                source="heuristic_backbone",
                role="fallback_extraction",
            )
            segment_speaker_id: Optional[str] = None
            segment_speaker_name = claim_profile.speaker_name or None
            if segment_speaker_name:
                segment_speaker_id = ensure_entity(
                    segment_speaker_name,
                    semantic_type="speaker",
                    source_segment=seg_idx,
                    source_span=segment.span,
                    confidence=max(0.58, float(getattr(segment, "confidence", 0.0) or 0.0)),
                    status="supported",
                    alias=segment_speaker_name,
                )
            structured_pairs = tuple((state.key, state.value) for state in segment.states)
            segment_routing = getattr(segment, "routing", None)
            allow_nl_semantics = (
                segment_routing is None
                or str(getattr(segment_routing, "modality", "unknown")) in {"natural_text", "mixed"}
            )
            relation_hints = (
                extract_relation_hints(segment.text, span=segment.span)
                if allow_nl_semantics
                else []
            )
            goal_hints = extract_goal_hints(
                segment.text,
                structured_pairs=structured_pairs,
                span=segment.span,
            )
            entity_hints = extract_entity_hints(
                segment.text,
                structured_pairs=structured_pairs,
                relations=relation_hints,
                goals=goal_hints,
                span=segment.span,
            )
            event_hints = (
                extract_event_hints(
                    segment.text,
                    relations=relation_hints,
                    span=segment.span,
                )
                if allow_nl_semantics
                else []
            )
            relation_proposals += float(len(relation_hints))
            goal_proposals += float(len(goal_hints))
            entity_proposals += float(len(entity_hints))
            event_proposals += float(len(event_hints))

            for entity in entity_hints:
                entity_id = resolve_entity(
                    entity.name,
                    semantic_type=entity.semantic_type,
                    source_segment=seg_idx,
                    source_span=entity.span or segment.span,
                    confidence=entity.confidence,
                    status=entity.status,
                )
                if entity_id is not None:
                    segment_entity_ids.append(entity_id)
                    if segment_salient_entity_id is None and not _is_anaphor(entity.name, entity.semantic_type):
                        segment_salient_entity_id = entity_id

            for idx, state in enumerate(segment.states):
                entity_id = resolve_entity(
                    state.key,
                    semantic_type="entity",
                    source_segment=seg_idx,
                    source_span=state.span or segment.span,
                    confidence=max(0.62, state.confidence),
                    status=state.status,
                )
                if entity_id is None:
                    continue
                segment_entity_ids.append(entity_id)
                if segment_salient_entity_id is None:
                    segment_salient_entity_id = entity_id
                states.append(
                    SemanticState(
                        state_id=f"state:{seg_idx}:{idx}",
                        key_entity_id=entity_id,
                        key_name=state.key,
                        value=state.value,
                        source_segment=seg_idx,
                        source_span=state.span or segment.span,
                        confidence=max(0.62, state.confidence),
                        status="proposal",
                        evidence_refs=segment_claim_evidence,
                    )
                )
                claims.append(
                    SemanticClaim(
                        claim_id=f"claim:state:{seg_idx}:{idx}",
                        claim_kind="state",
                        source_segment=seg_idx,
                        source_span=state.span or segment.span,
                        confidence=max(0.62, state.confidence),
                        status="proposal",
                        subject_entity_id=entity_id,
                        predicate="state",
                        object_value=state.value,
                        proposition_id=f"state:{seg_idx}:{idx}",
                        speaker_entity_id=segment_speaker_id,
                        speaker_name=segment_speaker_name,
                        epistemic_status=claim_profile.epistemic_status,
                        claim_source="fallback_extraction",
                        semantic_mode=claim_profile.semantic_mode,
                        quantifier_mode=claim_profile.quantifier_mode,
                        evidence_refs=segment_claim_evidence,
                    )
                )

            seen_event_triples: Set[Tuple[str, str, str]] = set()
            for idx, event in enumerate(event_hints):
                subj_id = resolve_entity(
                    event.subject,
                    semantic_type="entity",
                    source_segment=seg_idx,
                    source_span=event.span or segment.span,
                    confidence=event.confidence,
                    status=event.status,
                )
                obj_id = (
                    resolve_entity(
                        event.object_name,
                        semantic_type="entity",
                        source_segment=seg_idx,
                        source_span=event.span or segment.span,
                        confidence=event.confidence,
                        status=event.status,
                    )
                    if event.object_name
                    else None
                )
                agent_id = (
                    resolve_entity(
                        event.agent or event.subject,
                        semantic_type="agent",
                        source_segment=seg_idx,
                        source_span=event.span or segment.span,
                        confidence=max(0.54, event.confidence * 0.95),
                        status=event.status,
                    )
                    if event.agent or event.subject
                    else None
                )
                patient_id = (
                    resolve_entity(
                        event.patient or event.object_name,
                        semantic_type="entity",
                        source_segment=seg_idx,
                        source_span=event.span or segment.span,
                        confidence=max(0.54, event.confidence * 0.95),
                        status=event.status,
                    )
                    if event.patient or event.object_name
                    else None
                )
                event_id = f"event:{seg_idx}:{idx}"
                events.append(
                    SemanticEvent(
                        event_id=event_id,
                        event_type=event.predicate,
                        subject_entity_id=subj_id,
                        object_entity_id=obj_id,
                        agent_entity_id=agent_id or subj_id,
                        patient_entity_id=patient_id or obj_id,
                        subject_name=event.subject,
                        object_name=event.object_name,
                        modality=event.modality,
                        condition=event.condition,
                        explanation=event.explanation,
                        temporal=event.temporal,
                        source_segment=seg_idx,
                        source_span=event.span or segment.span,
                        confidence=event.confidence,
                        polarity="negative" if segment.counterexample else "positive",
                        status="proposal",
                        evidence_refs=segment_claim_evidence,
                        metadata={
                            "counterexample_segment": 1.0 if segment.counterexample else 0.0,
                            "fallback_backbone": 1.0,
                            "fallback_semantic_authority": 0.0,
                            "has_modality": 1.0 if event.modality else 0.0,
                            "has_condition": 1.0 if event.condition else 0.0,
                            "has_explanation": 1.0 if event.explanation else 0.0,
                            "has_temporal": 1.0 if event.temporal else 0.0,
                        },
                    )
                )
                if subj_id is not None:
                    segment_entity_ids.append(subj_id)
                if obj_id is not None:
                    segment_entity_ids.append(obj_id)
                if agent_id is not None:
                    segment_entity_ids.append(agent_id)
                if patient_id is not None:
                    segment_entity_ids.append(patient_id)
                if agent_id is not None:
                    segment_salient_entity_id = agent_id
                elif subj_id is not None:
                    segment_salient_entity_id = subj_id
                triple = (_normalized(event.subject), event.predicate, _normalized(event.object_name))
                seen_event_triples.add(triple)
                claims.append(
                    SemanticClaim(
                        claim_id=f"claim:event:{seg_idx}:{idx}",
                        claim_kind="relation",
                        source_segment=seg_idx,
                        source_span=event.span or segment.span,
                        confidence=event.confidence,
                        status="proposal",
                        subject_entity_id=subj_id,
                        predicate=event.predicate,
                        object_entity_id=obj_id,
                        object_value=event.object_name,
                        proposition_id=event_id,
                        event_id=event_id,
                        speaker_entity_id=segment_speaker_id,
                        speaker_name=segment_speaker_name,
                        epistemic_status=claim_profile.epistemic_status,
                        claim_source="fallback_extraction",
                        semantic_mode=claim_profile.semantic_mode,
                        quantifier_mode=claim_profile.quantifier_mode,
                        evidence_refs=segment_claim_evidence,
                    )
                )

            for idx, relation in enumerate(relation_hints):
                triple = (_normalized(relation.left), relation.relation, _normalized(relation.right))
                if triple in seen_event_triples:
                    continue
                subj_id = resolve_entity(
                    relation.left,
                    semantic_type="entity",
                    source_segment=seg_idx,
                    source_span=relation.span or segment.span,
                    confidence=relation.confidence,
                    status=relation.status,
                )
                obj_id = resolve_entity(
                    relation.right,
                    semantic_type="entity",
                    source_segment=seg_idx,
                    source_span=relation.span or segment.span,
                    confidence=relation.confidence,
                    status=relation.status,
                )
                event_id = f"event:fallback:{seg_idx}:{idx}"
                events.append(
                    SemanticEvent(
                        event_id=event_id,
                        event_type=relation.relation,
                        subject_entity_id=subj_id,
                        object_entity_id=obj_id,
                        agent_entity_id=subj_id,
                        patient_entity_id=obj_id,
                        subject_name=relation.left,
                        object_name=relation.right,
                        source_segment=seg_idx,
                        source_span=relation.span or segment.span,
                        confidence=relation.confidence,
                        polarity="negative" if segment.counterexample else "positive",
                        status="proposal",
                        evidence_refs=segment_claim_evidence,
                        metadata={
                            "counterexample_segment": 1.0 if segment.counterexample else 0.0,
                            "fallback_backbone": 1.0,
                            "fallback_semantic_authority": 0.0,
                        },
                    )
                )
                claims.append(
                    SemanticClaim(
                        claim_id=f"claim:relation:{seg_idx}:{idx}",
                        claim_kind="relation",
                        source_segment=seg_idx,
                        source_span=relation.span or segment.span,
                        confidence=relation.confidence,
                        status="proposal",
                        subject_entity_id=subj_id,
                        predicate=relation.relation,
                        object_entity_id=obj_id,
                        object_value=relation.right,
                        proposition_id=event_id,
                        event_id=event_id,
                        speaker_entity_id=segment_speaker_id,
                        speaker_name=segment_speaker_name,
                        epistemic_status=claim_profile.epistemic_status,
                        claim_source="fallback_extraction",
                        semantic_mode=claim_profile.semantic_mode,
                        quantifier_mode=claim_profile.quantifier_mode,
                        evidence_refs=segment_claim_evidence,
                    )
                )

            for idx, goal in enumerate(goal_hints):
                target_id = resolve_entity(
                    goal.goal_value,
                    semantic_type="entity",
                    source_segment=seg_idx,
                    source_span=goal.span or segment.span,
                    confidence=goal.confidence,
                    status=goal.status,
                )
                goal_id = f"goal:{seg_idx}:{idx}"
                goals.append(
                    SemanticGoal(
                        goal_id=goal_id,
                        goal_name=goal.goal_name,
                        goal_value=goal.goal_value,
                        target_entity_id=target_id,
                        source_segment=seg_idx,
                        source_span=goal.span or segment.span,
                        confidence=goal.confidence,
                        status="proposal",
                        evidence_refs=segment_claim_evidence,
                    )
                )
                claims.append(
                    SemanticClaim(
                        claim_id=f"claim:goal:{seg_idx}:{idx}",
                        claim_kind="goal",
                        source_segment=seg_idx,
                        source_span=goal.span or segment.span,
                        confidence=goal.confidence,
                        status="proposal",
                        predicate=goal.goal_name,
                        object_entity_id=target_id,
                        object_value=goal.goal_value,
                        proposition_id=goal_id,
                        goal_id=goal_id,
                        speaker_entity_id=segment_speaker_id,
                        speaker_name=segment_speaker_name,
                        epistemic_status=claim_profile.epistemic_status,
                        claim_source="fallback_extraction",
                        semantic_mode=claim_profile.semantic_mode,
                        quantifier_mode=claim_profile.quantifier_mode,
                        evidence_refs=segment_claim_evidence,
                    )
                )

            if segment_entity_ids:
                last_salient_entity_id = segment_salient_entity_id or segment_entity_ids[-1]

        entities = tuple(
            entity_acc[entity_id].finalize(entity_id)
            for entity_id in sorted(entity_acc.keys())
        )
        mentions, discourse_relations, temporal_markers, explanations = build_semantic_context_objects(
            document,
            entities,
        )
        metadata = dict(document.metadata)
        metadata.update(
            {
                "scene_entities": float(len(entities)),
                "scene_states": float(len(states)),
                "scene_events": float(len(events)),
                "scene_goals": float(len(goals)),
                "scene_claims": float(len(claims)),
                "scene_claim_attributed": float(sum(1 for claim in claims if claim.speaker_entity_id)),
                "scene_claim_nonasserted": float(
                    sum(1 for claim in claims if str(claim.epistemic_status) != "asserted")
                ),
                "scene_claim_generic": float(sum(1 for claim in claims if str(claim.semantic_mode) == "generic")),
                "scene_claim_rule": float(sum(1 for claim in claims if str(claim.semantic_mode) == "rule")),
                "scene_claim_obligation": float(sum(1 for claim in claims if str(claim.semantic_mode) == "obligation")),
                "scene_mentions": float(len(mentions)),
                "scene_discourse_relations": float(len(discourse_relations)),
                "scene_temporal_markers": float(len(temporal_markers)),
                "scene_explanations": float(len(explanations)),
                "scene_coreference_links": float(len(coreference_links)),
                "scene_negative_events": float(sum(1 for event in events if event.polarity == "negative")),
                "scene_event_modalities": float(sum(1 for event in events if event.modality)),
                "scene_event_conditions": float(sum(1 for event in events if event.condition)),
                "scene_event_explanations": float(sum(1 for event in events if event.explanation)),
                "scene_event_temporal_anchors": float(sum(1 for event in events if event.temporal)),
                "scene_entity_aliases": float(sum(max(len(entity.aliases) - 1, 0) for entity in entities)),
                "scene_mean_entity_confidence": float(
                    sum(entity.confidence for entity in entities) / max(len(entities), 1)
                ),
                "scene_mean_event_confidence": float(
                    sum(event.confidence for event in events) / max(len(events), 1)
                ),
                "scene_fallback_backbone_active": 1.0,
                "scene_fallback_low_authority": 1.0,
                "scene_backbone_replaceable": 1.0,
                "scene_fallback_relation_proposals": relation_proposals,
                "scene_fallback_goal_proposals": goal_proposals,
                "scene_fallback_entity_proposals": entity_proposals,
                "scene_fallback_event_proposals": event_proposals,
                "scene_fallback_attributed_claims": float(sum(1 for claim in claims if claim.speaker_entity_id)),
                "scene_fallback_nonasserted_claims": float(
                    sum(1 for claim in claims if str(claim.epistemic_status) != "asserted")
                ),
                "scene_fallback_generic_claims": float(
                    sum(1 for claim in claims if str(claim.semantic_mode) == "generic")
                ),
                "scene_fallback_rule_claims": float(
                    sum(1 for claim in claims if str(claim.semantic_mode) == "rule")
                ),
            }
        )
        return SemanticSceneGraph(
            language=document.language,
            source_text=document.source_text,
            entities=entities,
            states=tuple(states),
            events=tuple(events),
            goals=tuple(goals),
            claims=tuple(claims),
            mentions=mentions,
            discourse_relations=discourse_relations,
            temporal_markers=temporal_markers,
            explanations=explanations,
            coreference_links=tuple(coreference_links),
            metadata=metadata,
        )
