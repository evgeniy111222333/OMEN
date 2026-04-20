from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .types import (
    GroundedGoalHint,
    GroundedRelationHint,
    GroundedStateHint,
    GroundedTextDocument,
    GroundedTextSegment,
)


_WORD_RE = re.compile(r"[^\W_][\w'’`.-]*", re.UNICODE)
_QUOTE_RE = re.compile(r"[\"“”«»]([^\"“”«»]+)[\"“”«»]", re.UNICODE)
_STATE_PAIR_RE = re.compile(
    r"(?P<key>[^\W\d_][\w.-]*)\s*(?P<op>=|:)\s*(?P<value>\"[^\"]+\"|«[^»]+»|[^\s,;|]+)",
    re.UNICODE,
)

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
    }
)

_NEGATION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE | re.UNICODE)
    for pattern in (
        r"\bnot\b",
        r"\bnever\b",
        r"\bno\b",
        r"\bwithout\b",
        r"\bhowever\b",
        r"\bbut\b",
        r"\bfailed\b",
        r"\bне\b",
        r"\bнемає\b",
        r"\bбез\b",
        r"\bале\b",
        r"\bоднак\b",
        r"\bзбій\b",
        r"\bпомилка\b",
        r"\bнеправиль\w*\b",
    )
)

_GOAL_MARKERS: Tuple[Tuple[str, float], ...] = (
    ("expected", 0.60),
    ("expected result", 0.65),
    ("goal", 0.60),
    ("target", 0.60),
    ("desired", 0.58),
    ("need", 0.54),
    ("must", 0.54),
    ("should", 0.54),
    ("want", 0.52),
    ("aim", 0.52),
    ("мета", 0.60),
    ("ціль", 0.60),
    ("завдання", 0.60),
    ("очікується", 0.60),
    ("очікуваний результат", 0.65),
    ("потрібно", 0.56),
    ("треба", 0.56),
    ("має", 0.52),
    ("повинен", 0.56),
)

_RELATION_MARKERS: Tuple[Tuple[str, str, float], ...] = (
    ("results in", "causes", 0.72),
    ("leads to", "causes", 0.72),
    ("because", "causes", 0.66),
    ("triggers", "causes", 0.70),
    ("trigger", "causes", 0.68),
    ("causes", "causes", 0.70),
    ("cause", "causes", 0.68),
    ("contains", "contains", 0.66),
    ("becomes", "becomes", 0.66),
    ("become", "becomes", 0.64),
    ("generates", "generates", 0.76),
    ("generate", "generates", 0.74),
    ("creates", "generates", 0.74),
    ("create", "generates", 0.72),
    ("opens", "opens", 0.74),
    ("open", "opens", 0.72),
    ("has", "has", 0.58),
    ("have", "has", 0.58),
    ("were", "is", 0.54),
    ("was", "is", 0.54),
    ("are", "is", 0.54),
    ("is", "is", 0.54),
    ("призводить до", "causes", 0.74),
    ("спричиняє", "causes", 0.74),
    ("викликає", "causes", 0.72),
    ("містить", "contains", 0.68),
    ("перетворюється", "becomes", 0.68),
    ("стає", "becomes", 0.64),
    ("генерують", "generates", 0.78),
    ("генерує", "generates", 0.78),
    ("створюють", "generates", 0.76),
    ("створює", "generates", 0.76),
    ("відчиняються", "opens", 0.78),
    ("відчиняє", "opens", 0.76),
    ("відкривають", "opens", 0.76),
    ("відкриває", "opens", 0.76),
    ("мають", "has", 0.60),
    ("має", "has", 0.60),
    ("є", "is", 0.58),
)


def _strip_wrapping_quotes(text: str) -> str:
    stripped = text.strip()
    if len(stripped) >= 2 and (
        (stripped[0] == '"' and stripped[-1] == '"')
        or (stripped[0] == "«" and stripped[-1] == "»")
    ):
        return stripped[1:-1].strip()
    return stripped


def normalize_symbol_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    if not text:
        return None
    text = (
        text.replace("’", "'")
        .replace("`", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("«", '"')
        .replace("»", '"')
    )
    text = _strip_wrapping_quotes(text)
    text = text.replace("'", "")
    text = re.sub(r"[^\w\s.-]+", " ", text, flags=re.UNICODE)
    text = re.sub(r"[\s.-]+", "_", text, flags=re.UNICODE)
    text = text.strip("_")
    return text[:64] if text else None


def tokenize_semantic_words(segment: str, *, limit: Optional[int] = None) -> List[str]:
    tokens: List[str] = []
    seen: Set[str] = set()
    for raw in _WORD_RE.findall(unicodedata.normalize("NFKC", segment)):
        normalized = normalize_symbol_text(raw)
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        tokens.append(normalized)
        if limit is not None and len(tokens) >= int(limit):
            break
    return tokens


def _normalize_segment(segment: str) -> str:
    return re.sub(
        r"^\s*(?:step\s*\d+[:.)-]*|\d+[:.)-]*)\s*",
        "",
        segment.strip(),
        flags=re.IGNORECASE | re.UNICODE,
    )


def split_text_segments(text: str, *, max_segments: int = 24) -> List[str]:
    lines = [_normalize_segment(line) for line in text.splitlines() if line.strip()]
    if len(lines) >= 2:
        return lines[: max(1, int(max_segments))]
    normalized = text.replace("\r", "\n").strip()
    if not normalized:
        return []
    sentence_like = [
        _normalize_segment(chunk)
        for chunk in re.split(r"(?<=[.!?;])\s+|\n+", normalized)
        if chunk.strip()
    ]
    if sentence_like:
        return sentence_like[: max(1, int(max_segments))]
    return [_normalize_segment(normalized[:256])]


def _flatten_payload(payload: Any, prefix: str = "") -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    if isinstance(payload, dict):
        for key, value in list(payload.items())[:8]:
            normalized_key = normalize_symbol_text(key)
            if not normalized_key:
                continue
            child_prefix = normalized_key if not prefix else f"{prefix}_{normalized_key}"
            pairs.extend(_flatten_payload(value, child_prefix))
        return pairs
    if isinstance(payload, (list, tuple)):
        for idx, value in enumerate(list(payload)[:6]):
            child_prefix = prefix or f"item_{idx}"
            pairs.extend(_flatten_payload(value, child_prefix))
        return pairs
    normalized_key = normalize_symbol_text(prefix or "value")
    normalized_value = normalize_symbol_text(payload)
    if normalized_key and normalized_value:
        pairs.append((normalized_key, normalized_value))
    return pairs


def extract_structured_pairs(segment: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    stripped = segment.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            payload = json.loads(stripped)
        except Exception:
            payload = None
        if payload is not None:
            for pair in _flatten_payload(payload):
                if pair in seen:
                    continue
                seen.add(pair)
                pairs.append(pair)
    for match in _STATE_PAIR_RE.finditer(segment):
        key = normalize_symbol_text(match.group("key"))
        value = normalize_symbol_text(match.group("value"))
        if key is None or value is None:
            continue
        pair = (key, value)
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
    return pairs


def _quoted_focus(fragment: str, *, left: bool) -> Optional[str]:
    matches = [normalize_symbol_text(match) for match in _QUOTE_RE.findall(fragment)]
    matches = [match for match in matches if match]
    if not matches:
        return None
    return matches[-1] if left else matches[0]


def _focus_symbol(fragment: str, *, left: bool) -> Optional[str]:
    quoted = _quoted_focus(fragment, left=left)
    if quoted is not None:
        return quoted
    tokens = tokenize_semantic_words(fragment)
    filtered = [token for token in tokens if token not in _FOCUS_WRAPPER_TOKENS]
    tokens = filtered or tokens
    if not tokens:
        return None
    if left:
        phrase = "_".join(tokens[-3:])
    else:
        phrase = "_".join(tokens[:3])
    return normalize_symbol_text(phrase)


def _marker_regex(marker: str) -> re.Pattern[str]:
    escaped = re.escape(marker)
    if any(ch.isalnum() for ch in marker):
        return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE | re.UNICODE)
    return re.compile(escaped, re.IGNORECASE | re.UNICODE)


_RELATION_PATTERNS: Tuple[Tuple[re.Pattern[str], str, float], ...] = tuple(
    (_marker_regex(marker), canonical, confidence)
    for marker, canonical, confidence in sorted(
        _RELATION_MARKERS,
        key=lambda item: len(item[0]),
        reverse=True,
    )
)

_GOAL_PATTERNS: Tuple[Tuple[re.Pattern[str], float], ...] = tuple(
    (_marker_regex(marker), confidence)
    for marker, confidence in sorted(
        _GOAL_MARKERS,
        key=lambda item: len(item[0]),
        reverse=True,
    )
)


def extract_relation_hints(segment: str) -> List[GroundedRelationHint]:
    relations: List[GroundedRelationHint] = []
    seen: Set[Tuple[str, str, str]] = set()
    for pattern, canonical, confidence in _RELATION_PATTERNS:
        for match in pattern.finditer(segment):
            left = _focus_symbol(segment[: match.start()], left=True)
            right = _focus_symbol(segment[match.end() :], left=False)
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
                )
            )
    chain_parts = [
        _focus_symbol(part, left=False)
        for part in re.split(r"\s*(?:->|=>|then|потім)\s*", segment, flags=re.IGNORECASE | re.UNICODE)
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
                )
            )
    return relations


def extract_goal_hints(segment: str, *, structured_pairs: Optional[Sequence[Tuple[str, str]]] = None) -> List[GroundedGoalHint]:
    goals: List[GroundedGoalHint] = []
    seen: Set[Tuple[str, str]] = set()
    for key, value in list(structured_pairs or extract_structured_pairs(segment)):
        if key not in _GOAL_KEYS:
            continue
        pair = (key, value)
        if pair in seen:
            continue
        seen.add(pair)
        goals.append(GroundedGoalHint(goal_name=key, goal_value=value, confidence=0.64))
    for pattern, confidence in _GOAL_PATTERNS:
        match = pattern.search(segment)
        if match is None:
            continue
        goal_value = _focus_symbol(segment[match.end() :], left=False)
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
            )
        )
    return goals


def is_counterexample_text(segment: str) -> bool:
    normalized = unicodedata.normalize("NFKC", segment).casefold()
    return any(pattern.search(normalized) is not None for pattern in _NEGATION_PATTERNS)


def ground_text_document(
    text: str,
    *,
    language: str = "text",
    max_segments: int = 24,
    token_limit: int = 8,
) -> GroundedTextDocument:
    segments_out: List[GroundedTextSegment] = []
    for idx, segment in enumerate(split_text_segments(text, max_segments=max_segments)):
        normalized = unicodedata.normalize("NFKC", segment)
        states = [
            GroundedStateHint(key=key, value=value)
            for key, value in extract_structured_pairs(segment)
        ]
        relations = extract_relation_hints(segment)
        goals = extract_goal_hints(segment, structured_pairs=[(state.key, state.value) for state in states])
        segments_out.append(
            GroundedTextSegment(
                index=idx,
                text=segment,
                normalized_text=normalized,
                tokens=tuple(tokenize_semantic_words(segment, limit=token_limit)),
                states=tuple(states),
                relations=tuple(relations),
                goals=tuple(goals),
                counterexample=is_counterexample_text(segment),
            )
        )
    metadata: Dict[str, float] = {
        "grounding_segments": float(len(segments_out)),
        "grounding_tokens": float(sum(len(segment.tokens) for segment in segments_out)),
        "grounding_state_hints": float(sum(len(segment.states) for segment in segments_out)),
        "grounding_relation_hints": float(sum(len(segment.relations) for segment in segments_out)),
        "grounding_goal_hints": float(sum(len(segment.goals) for segment in segments_out)),
        "grounding_counterexample_segments": float(sum(1 for segment in segments_out if segment.counterexample)),
        "grounding_multilingual": 1.0 if any(ord(ch) > 127 for ch in text) else 0.0,
    }
    return GroundedTextDocument(
        language=language or "text",
        source_text=text,
        segments=tuple(segments_out),
        metadata=metadata,
    )
