from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import replace
from typing import Any, Dict, List, Optional, Set, Tuple

from .source_routing import build_parser_candidates, infer_source_profile
from .types import (
    GroundedEntityHint,
    GroundedEventHint,
    GroundedGoalHint,
    GroundedRelationHint,
    GroundedStructuralUnit,
    GroundedStateHint,
    GroundedTextDocument,
    GroundedTextSegment,
    GroundingSourceProfile,
    GroundingSpan,
)


_WORD_RE = re.compile(r"[^\W\d_](?:[\w'’`.-]*[^\W_])?", re.UNICODE)
_STATE_PAIR_RE = re.compile(
    r"(?P<key>[^\W\d_][\w.-]*)\s*(?P<op>=|:)\s*(?P<value>\"[^\"]+\"|«[^»]+»|[^\s,;|]+)",
    re.UNICODE,
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
        r"\bfail(?:ed|s|ure)?\b",
        r"\bcannot\b",
        r"\bcan't\b",
        r"\bне\b",
        r"\bнемає\b",
        r"\bбез\b",
        r"\bале\b",
        r"\bоднак\b",
        r"\bзбій\b",
        r"\bпомилка\b",
        r"\bневдач\w*\b",
        r"\bнеправиль\w*\b",
    )
)

_DEFAULT_SOURCE_ID = "source:primary"
_DEFAULT_DOCUMENT_ID = "document:primary"
_DEFAULT_EPISODE_ID = "episode:primary"

_GOAL_KEYS = frozenset(
    {
        "goal",
        "target",
        "desired",
        "expected",
        "objective",
        "aim",
        "мета",
        "ціль",
        "очікується",
        "очікуваний_результат",
    }
)
_STRUCTURED_RECORD_UNITS = frozenset(
    {
        "key_value_record",
        "json_record",
        "log_entry",
        "table_row",
    }
)
_NATURAL_PRIMARY_UNITS = frozenset({"speaker_turn", "clause"})
_NATURAL_SUPPORT_UNITS = frozenset({"speaker_turn", "clause", "citation_region"})
_NATURAL_STRUCTURAL_SUBTYPES = frozenset({"dialogue_text", "instructional_text"})


def _maybe_repair_mojibake(text: str) -> str:
    raw = str(text or "")
    mojibake_score = sum(raw.count(marker) for marker in ("Р", "С", "Ѓ", "І", "ї"))
    if mojibake_score < 4:
        return raw
    try:
        repaired = raw.encode("cp1251").decode("utf-8")
    except Exception:
        return raw
    repaired_score = sum("а" <= ch.casefold() <= "я" or ch in {"є", "і", "ї", "ґ"} for ch in repaired)
    original_score = sum("а" <= ch.casefold() <= "я" or ch in {"є", "і", "ї", "ґ"} for ch in raw)
    return repaired if repaired_score > original_score else raw


def _normalize_with_char_map(text: str) -> Tuple[str, List[int]]:
    repaired = _maybe_repair_mojibake(str(text or ""))
    normalized_chars: List[str] = []
    char_map: List[int] = []
    for idx, char in enumerate(repaired):
        replaced = (
            char.replace("’", "'")
            .replace("`", "'")
            .replace("“", '"')
            .replace("”", '"')
            .replace("„", '"')
        )
        normalized_piece = unicodedata.normalize("NFKC", replaced)
        for normalized_char in normalized_piece:
            normalized_chars.append(normalized_char)
            char_map.append(idx)
    return "".join(normalized_chars), char_map


def _normalize_unicode_text(text: str) -> str:
    return _normalize_with_char_map(text)[0]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _strip_wrapping_quotes(text: str) -> str:
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] in "\"«“”„" and stripped[-1] in "\"»“”„":
        return stripped[1:-1].strip()
    return stripped


def normalize_symbol_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = _normalize_unicode_text(str(value)).strip().casefold()
    if not text:
        return None
    text = _strip_wrapping_quotes(text).replace("'", "")
    text = re.sub(r"[^\w\s:.-]+", " ", text, flags=re.UNICODE)
    text = re.sub(r"[\s.-]+", "_", text, flags=re.UNICODE)
    text = text.strip("_")
    return text[:80] if text else None


def tokenize_semantic_words(segment: str, *, limit: Optional[int] = None) -> List[str]:
    tokens: List[str] = []
    seen: Set[str] = set()
    for raw in _WORD_RE.findall(_normalize_unicode_text(segment)):
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
        r"^\s*(?:step\s*\d+[:.)-]*|\d+[:.)-]*|крок\s*\d+[:.)-]*)\s*",
        "",
        _normalize_unicode_text(segment).strip(),
        flags=re.IGNORECASE | re.UNICODE,
    )


def _char_to_byte_offset(text: str, char_offset: int) -> int:
    safe_offset = max(0, min(len(text), int(char_offset)))
    return len(text[:safe_offset].encode("utf-8"))


def _build_grounding_span(
    text: str,
    *,
    start: int,
    end: int,
    source_id: str = "",
    document_id: str = "",
    episode_id: str = "",
) -> GroundingSpan:
    safe_start = max(0, min(len(text), int(start)))
    safe_end = max(safe_start, min(len(text), int(end)))
    return GroundingSpan(
        start=safe_start,
        end=safe_end,
        text=text[safe_start:safe_end],
        byte_start=_char_to_byte_offset(text, safe_start),
        byte_end=_char_to_byte_offset(text, safe_end),
        source_id=str(source_id or ""),
        document_id=str(document_id or ""),
        episode_id=str(episode_id or ""),
    )


def split_text_segments(text: str, *, max_segments: int = 24) -> List[str]:
    lines = [_normalize_segment(line) for line in str(text or "").splitlines() if line.strip()]
    if len(lines) >= 2:
        return lines[: max(1, int(max_segments))]
    normalized = _normalize_unicode_text(text).replace("\r", "\n").strip()
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


def _locate_segment_span(
    text: str,
    segment: str,
    start_cursor: int,
    *,
    source_id: str = "",
    document_id: str = "",
    episode_id: str = "",
) -> GroundingSpan:
    raw = str(segment or "")
    if not raw:
        cursor = max(0, int(start_cursor))
        return _build_grounding_span(
            text,
            start=cursor,
            end=cursor,
            source_id=source_id,
            document_id=document_id,
            episode_id=episode_id,
        )
    idx = text.find(raw, max(0, int(start_cursor)))
    if idx < 0:
        idx = text.find(raw)
    if idx < 0:
        stripped = raw.strip()
        idx = text.find(stripped, max(0, int(start_cursor))) if stripped else -1
        raw = stripped or raw
    if idx < 0 and raw:
        normalized_text, char_map = _normalize_with_char_map(text)
        normalized_raw = _normalize_unicode_text(raw)
        normalized_start = 0
        search_cursor = max(0, int(start_cursor))
        for pos, original_idx in enumerate(char_map):
            if original_idx >= search_cursor:
                normalized_start = pos
                break
        normalized_idx = normalized_text.find(normalized_raw, normalized_start) if normalized_raw else -1
        if normalized_idx < 0 and normalized_raw:
            normalized_idx = normalized_text.find(normalized_raw)
        if normalized_idx >= 0 and char_map:
            idx = char_map[normalized_idx]
            normalized_end = min(normalized_idx + len(normalized_raw), len(char_map))
            end = char_map[normalized_end - 1] + 1
            return _build_grounding_span(
                text,
                start=idx,
                end=end,
                source_id=source_id,
                document_id=document_id,
                episode_id=episode_id,
            )
    if idx < 0:
        idx = max(0, int(start_cursor))
    end = min(len(text), idx + len(raw))
    return _build_grounding_span(
        text,
        start=idx,
        end=end,
        source_id=source_id,
        document_id=document_id,
        episode_id=episode_id,
    )


def split_text_segments_with_spans(
    text: str,
    *,
    max_segments: int = 24,
    source_id: str = "",
    document_id: str = "",
    episode_id: str = "",
) -> List[Tuple[str, GroundingSpan]]:
    out: List[Tuple[str, GroundingSpan]] = []
    cursor = 0
    for segment in split_text_segments(text, max_segments=max_segments):
        span = _locate_segment_span(
            str(text or ""),
            segment,
            cursor,
            source_id=source_id,
            document_id=document_id,
            episode_id=episode_id,
        )
        out.append((segment, span))
        cursor = max(span.end, cursor)
    return out


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
    stripped = _normalize_unicode_text(segment).strip()
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
    for match in _STATE_PAIR_RE.finditer(stripped):
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


def is_counterexample_text(segment: str) -> bool:
    normalized = _normalize_unicode_text(segment).casefold()
    return any(pattern.search(normalized) is not None for pattern in _NEGATION_PATTERNS)


def _subspan(parent_span: Optional[GroundingSpan], parent_text: str, fragment: str, occurrence: int = 0) -> Optional[GroundingSpan]:
    if parent_span is None:
        return None
    raw_fragment = str(fragment or "").strip()
    if not raw_fragment:
        return None
    parent_source_text = str(parent_span.text or parent_text or "")
    start_at = 0
    idx = -1
    matched_text = raw_fragment
    char_start_rel: Optional[int] = None
    char_end_rel: Optional[int] = None
    for _ in range(max(1, int(occurrence) + 1)):
        idx = parent_source_text.find(raw_fragment, start_at)
        if idx < 0:
            break
        start_at = idx + len(raw_fragment)
    if idx >= 0:
        char_start_rel = idx
        char_end_rel = idx + len(raw_fragment)
        matched_text = parent_source_text[char_start_rel:char_end_rel]
    else:
        normalized_parent_text, char_map = _normalize_with_char_map(parent_source_text)
        normalized_fragment = _normalize_unicode_text(raw_fragment)
        start_at = 0
        normalized_idx = -1
        for _ in range(max(1, int(occurrence) + 1)):
            normalized_idx = normalized_parent_text.find(normalized_fragment, start_at)
            if normalized_idx < 0:
                break
            start_at = normalized_idx + len(normalized_fragment)
        if normalized_idx < 0:
            normalized_idx = normalized_parent_text.find(normalized_fragment)
        if normalized_idx < 0 or not char_map:
            return None
        char_start_rel = char_map[normalized_idx]
        normalized_end = min(normalized_idx + len(normalized_fragment), len(char_map))
        char_end_rel = char_map[normalized_end - 1] + 1
        matched_text = parent_source_text[char_start_rel:char_end_rel]
    assert char_start_rel is not None and char_end_rel is not None
    start = int(parent_span.start) + char_start_rel
    end = int(parent_span.start) + char_end_rel
    byte_start = None
    byte_end = None
    if parent_span.byte_start is not None:
        byte_prefix = len(parent_source_text[:char_start_rel].encode("utf-8"))
        byte_fragment = len(matched_text.encode("utf-8"))
        byte_start = int(parent_span.byte_start) + byte_prefix
        byte_end = byte_start + byte_fragment
    return GroundingSpan(
        start=start,
        end=end,
        text=matched_text,
        byte_start=byte_start,
        byte_end=byte_end,
        source_id=parent_span.source_id,
        document_id=parent_span.document_id,
        episode_id=parent_span.episode_id,
    )


def _field_value(value: str, *, fallback: str = "") -> str:
    normalized = normalize_symbol_text(value)
    if normalized:
        return normalized
    return fallback or str(value or "").strip()


def _extract_clause_units(
    segment: str,
    *,
    seg_idx: int,
    span: Optional[GroundingSpan],
    routing: GroundingSourceProfile,
) -> List[GroundedStructuralUnit]:
    units: List[GroundedStructuralUnit] = []
    clause_chunks = [
        chunk.strip()
        for chunk in re.split(r"(?<=[,;])\s+|\s+(?=але\b|однак\b|бо\b|if\b|when\b|because\b|then\b|after\b|before\b|якщо\b|коли\b|потім\b|після\b|перед\b)", segment, flags=re.IGNORECASE | re.UNICODE)
        if chunk.strip()
    ]
    if not clause_chunks:
        clause_chunks = [segment.strip()] if segment.strip() else []
    for idx, clause in enumerate(clause_chunks):
        markers = [
            marker
            for marker in ("if", "when", "because", "then", "after", "before", "якщо", "коли", "бо", "потім", "після", "перед", "але", "однак")
            if re.search(rf"(?<!\w){re.escape(marker)}(?!\w)", clause, flags=re.IGNORECASE | re.UNICODE)
        ]
        units.append(
            GroundedStructuralUnit(
                unit_id=f"clause:{seg_idx}:{idx}",
                unit_type="clause",
                text=clause,
                source_segment=seg_idx,
                span=_subspan(span, segment, clause, idx),
                confidence=max(0.52, float(routing.confidence) * 0.88),
                status="supported",
                references=tuple(f"marker:{marker}" for marker in markers),
            )
        )
    return units


def _extract_speaker_turn_unit(
    segment: str,
    *,
    seg_idx: int,
    span: Optional[GroundingSpan],
    routing: GroundingSourceProfile,
) -> List[GroundedStructuralUnit]:
    match = re.match(r"^(user|assistant|speaker|q|a)\s*[:>-]\s*(.+)$", segment, flags=re.IGNORECASE | re.UNICODE)
    if match is None:
        return []
    speaker = _field_value(match.group(1), fallback=match.group(1).strip().casefold())
    utterance = match.group(2).strip()
    return [
        GroundedStructuralUnit(
            unit_id=f"speaker_turn:{seg_idx}:0",
            unit_type="speaker_turn",
            text=segment.strip(),
            source_segment=seg_idx,
            span=span,
            confidence=max(0.58, float(routing.confidence)),
            status="supported",
            fields=(("speaker", speaker), ("utterance", _field_value(utterance, fallback=utterance[:96]))),
            references=(f"subtype:{routing.subtype}",),
        )
    ]


def _extract_citation_units(
    segment: str,
    *,
    seg_idx: int,
    span: Optional[GroundingSpan],
    routing: GroundingSourceProfile,
) -> List[GroundedStructuralUnit]:
    units: List[GroundedStructuralUnit] = []
    patterns = (
        r"\[[0-9]{1,3}\]",
        r"\([A-ZА-ЯІЇЄҐ][A-Za-zА-Яа-яІіЇїЄєҐґ]+,\s*\d{4}\)",
        r"\bet al\.\b",
    )
    match_index = 0
    for pattern in patterns:
        for match in re.finditer(pattern, segment):
            text_value = match.group(0)
            units.append(
                GroundedStructuralUnit(
                    unit_id=f"citation:{seg_idx}:{match_index}",
                    unit_type="citation_region",
                    text=text_value,
                    source_segment=seg_idx,
                    span=_subspan(span, segment, text_value, match_index),
                    confidence=max(0.50, float(routing.confidence) * 0.82),
                    status="supported",
                    references=(f"subtype:{routing.subtype}",),
                )
            )
            match_index += 1
    return units


def _extract_table_row_unit(
    segment: str,
    *,
    seg_idx: int,
    span: Optional[GroundingSpan],
    routing: GroundingSourceProfile,
) -> List[GroundedStructuralUnit]:
    delimiter = ""
    if segment.count("|") >= 2:
        delimiter = "|"
    elif segment.count("\t") >= 2:
        delimiter = "\t"
    elif segment.count(",") >= 2:
        delimiter = ","
    if not delimiter:
        return []
    columns = [column.strip() for column in segment.split(delimiter)]
    fields = tuple(
        (f"col_{idx}", _field_value(column, fallback=column[:64]))
        for idx, column in enumerate(columns)
        if column.strip()
    )
    return [
        GroundedStructuralUnit(
            unit_id=f"table_row:{seg_idx}:0",
            unit_type="table_row",
            text=segment.strip(),
            source_segment=seg_idx,
            span=span,
            confidence=max(0.60, float(routing.confidence)),
            status="supported",
            fields=fields,
            references=(f"delimiter:{repr(delimiter)}",),
        )
    ]


def _extract_log_entry_unit(
    segment: str,
    *,
    seg_idx: int,
    span: Optional[GroundingSpan],
    routing: GroundingSourceProfile,
) -> List[GroundedStructuralUnit]:
    timestamp_match = re.search(r"\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\b", segment)
    level_match = re.search(r"\b(info|warn|warning|error|debug|trace)\b", segment, flags=re.IGNORECASE | re.UNICODE)
    if timestamp_match is None and level_match is None:
        return []
    fields: List[Tuple[str, str]] = []
    if timestamp_match is not None:
        fields.append(("timestamp", _field_value(timestamp_match.group(0), fallback=timestamp_match.group(0))))
    if level_match is not None:
        fields.append(("level", _field_value(level_match.group(1), fallback=level_match.group(1).lower())))
    return [
        GroundedStructuralUnit(
            unit_id=f"log_entry:{seg_idx}:0",
            unit_type="log_entry",
            text=segment.strip(),
            source_segment=seg_idx,
            span=span,
            confidence=max(0.60, float(routing.confidence)),
            status="supported",
            fields=tuple(fields),
            references=(f"subtype:{routing.subtype}",),
        )
    ]


def _extract_section_header_unit(
    segment: str,
    *,
    seg_idx: int,
    span: Optional[GroundingSpan],
    routing: GroundingSourceProfile,
) -> List[GroundedStructuralUnit]:
    stripped = segment.strip()
    if not re.match(r"^\[[^\]]+\]$", stripped):
        return []
    header = stripped[1:-1].strip()
    return [
        GroundedStructuralUnit(
            unit_id=f"section_header:{seg_idx}:0",
            unit_type="section_header",
            text=stripped,
            source_segment=seg_idx,
            span=span,
            confidence=max(0.60, float(routing.confidence)),
            status="supported",
            fields=(("section", _field_value(header, fallback=header)),),
            references=(f"subtype:{routing.subtype}",),
        )
    ]


def _extract_key_value_units(
    segment: str,
    *,
    seg_idx: int,
    span: Optional[GroundingSpan],
    routing: GroundingSourceProfile,
    state_pairs: List[Tuple[str, str]],
) -> List[GroundedStructuralUnit]:
    units: List[GroundedStructuralUnit] = []
    for idx, (key, value) in enumerate(state_pairs):
        units.append(
            GroundedStructuralUnit(
                unit_id=f"key_value:{seg_idx}:{idx}",
                unit_type="key_value_record",
                text=f"{key}={value}",
                source_segment=seg_idx,
                span=span,
                confidence=max(0.62, float(routing.confidence)),
                status="supported",
                fields=((key, value),),
                references=(f"subtype:{routing.subtype}",),
            )
        )
    if routing.subtype == "json_records" and state_pairs:
        units.append(
            GroundedStructuralUnit(
                unit_id=f"json_record:{seg_idx}:0",
                unit_type="json_record",
                text=segment.strip(),
                source_segment=seg_idx,
                span=span,
                confidence=max(0.64, float(routing.confidence)),
                status="supported",
                fields=tuple(state_pairs[:16]),
                references=("parser:json_parser",),
            )
        )
    return units


def extract_structural_units(
    segment: str,
    *,
    seg_idx: int,
    span: Optional[GroundingSpan],
    routing: GroundingSourceProfile,
    state_pairs: Optional[List[Tuple[str, str]]] = None,
) -> List[GroundedStructuralUnit]:
    units: List[GroundedStructuralUnit] = []
    structural_pairs = list(state_pairs or extract_structured_pairs(segment))
    units.extend(_extract_section_header_unit(segment, seg_idx=seg_idx, span=span, routing=routing))
    if routing.modality in {"structured_text", "mixed"}:
        units.extend(
            _extract_key_value_units(
                segment,
                seg_idx=seg_idx,
                span=span,
                routing=routing,
                state_pairs=structural_pairs,
            )
        )
        units.extend(_extract_log_entry_unit(segment, seg_idx=seg_idx, span=span, routing=routing))
        units.extend(_extract_table_row_unit(segment, seg_idx=seg_idx, span=span, routing=routing))
    if routing.modality in {"natural_text", "mixed"}:
        units.extend(_extract_speaker_turn_unit(segment, seg_idx=seg_idx, span=span, routing=routing))
        units.extend(_extract_clause_units(segment, seg_idx=seg_idx, span=span, routing=routing))
        units.extend(_extract_citation_units(segment, seg_idx=seg_idx, span=span, routing=routing))
    return units


def _segment_primary_kind(
    routing: Optional[GroundingSourceProfile],
    structural_units: Tuple[GroundedStructuralUnit, ...],
) -> str:
    modality = str(getattr(routing, "modality", "") or "")
    subtype = str(getattr(routing, "subtype", "") or "")
    unit_types = {
        str(getattr(unit, "unit_type", "") or "")
        for unit in tuple(structural_units or ())
    }
    if "section_header" in unit_types:
        return "section_header"
    if modality == "structured_text" and unit_types.intersection(_STRUCTURED_RECORD_UNITS):
        return "structured_record"
    if (
        modality in {"natural_text", "mixed"}
        and subtype in _NATURAL_STRUCTURAL_SUBTYPES
        and unit_types.intersection(_NATURAL_PRIMARY_UNITS)
    ):
        return "natural_structural"
    return ""


def _refine_segment_routing(
    document_routing: GroundingSourceProfile,
    segment_routing: GroundingSourceProfile,
    structural_units: Tuple[GroundedStructuralUnit, ...],
) -> GroundingSourceProfile:
    doc_subtype = str(getattr(document_routing, "subtype", "") or "")
    seg_modality = str(getattr(segment_routing, "modality", "") or "")
    seg_subtype = str(getattr(segment_routing, "subtype", "") or "")
    unit_types = {
        str(getattr(unit, "unit_type", "") or "")
        for unit in tuple(structural_units or ())
    }
    if (
        doc_subtype == "instructional_text"
        and seg_modality == "natural_text"
        and seg_subtype == "generic_text"
        and "clause" in unit_types
    ):
        lifted = replace(
            segment_routing,
            subtype="instructional_text",
            verification_path=str(getattr(document_routing, "verification_path", "") or segment_routing.verification_path),
            confidence=max(float(segment_routing.confidence), float(document_routing.confidence) * 0.85),
            ambiguity=min(float(segment_routing.ambiguity), max(float(document_routing.ambiguity), 0.35)),
            evidence={**dict(segment_routing.evidence or {}), "document_subtype_lift": 1.0},
        )
        return replace(lifted, parser_candidates=build_parser_candidates(lifted))
    return segment_routing


def _speaker_turn_utterance(unit: GroundedStructuralUnit) -> str:
    raw = str(getattr(unit, "text", "") or "")
    match = re.match(r"^(user|assistant|speaker|q|a)\s*[:>-]\s*(.+)$", raw, flags=re.IGNORECASE | re.UNICODE)
    if match is not None:
        return match.group(2).strip()
    utterance = next(
        (
            str(value).strip()
            for key, value in tuple(getattr(unit, "fields", ()) or ())
            if str(key or "") == "utterance" and str(value or "").strip()
        ),
        "",
    )
    if utterance:
        return utterance
    return raw.strip()


def _build_document_semantic_hints(
    segment: str,
    *,
    span: Optional[GroundingSpan],
    routing: GroundingSourceProfile,
    structural_units: Tuple[GroundedStructuralUnit, ...],
    state_pairs: List[Tuple[str, str]],
) -> Tuple[
    Tuple[GroundedStateHint, ...],
    Tuple[GroundedRelationHint, ...],
    Tuple[GroundedGoalHint, ...],
    Tuple[GroundedEntityHint, ...],
    Tuple[GroundedEventHint, ...],
]:
    primary_kind = _segment_primary_kind(routing, structural_units)
    state_hints: List[GroundedStateHint] = []
    relation_hints: List[GroundedRelationHint] = []
    goal_hints: List[GroundedGoalHint] = []
    entity_hints: List[GroundedEntityHint] = []
    event_hints: List[GroundedEventHint] = []
    seen_states: Set[Tuple[str, str]] = set()
    seen_relations: Set[Tuple[str, str, str]] = set()
    seen_goals: Set[Tuple[str, str]] = set()
    seen_entities: Set[Tuple[str, str]] = set()
    seen_events: Set[Tuple[str, str, str, str, str, str, str]] = set()

    def add_state(key: str, value: str, *, confidence: float, source: str, hint_span: Optional[GroundingSpan]) -> None:
        pair = (str(key), str(value))
        if pair in seen_states:
            return
        seen_states.add(pair)
        state_hints.append(
            GroundedStateHint(
                key=str(key),
                value=str(value),
                confidence=float(confidence),
                source=source,
                status="supported",
                span=hint_span,
            )
        )

    def add_relation(hint: GroundedRelationHint) -> None:
        pair = (str(hint.left), str(hint.relation), str(hint.right))
        if pair in seen_relations:
            return
        seen_relations.add(pair)
        relation_hints.append(hint)

    def add_goal(hint: GroundedGoalHint) -> None:
        pair = (str(hint.goal_name), str(hint.goal_value))
        if pair in seen_goals:
            return
        seen_goals.add(pair)
        goal_hints.append(hint)

    def add_entity(hint: GroundedEntityHint) -> None:
        pair = (str(hint.name), str(hint.semantic_type))
        if pair in seen_entities:
            return
        seen_entities.add(pair)
        entity_hints.append(hint)

    def add_event(hint: GroundedEventHint) -> None:
        pair = (
            str(hint.subject),
            str(hint.predicate),
            str(hint.object_name or ""),
            str(hint.modality or ""),
            str(hint.condition or ""),
            str(hint.explanation or ""),
            str(hint.temporal or ""),
        )
        if pair in seen_events:
            return
        seen_events.add(pair)
        event_hints.append(hint)

    for key, value in list(state_pairs):
        if key in _GOAL_KEYS:
            add_goal(
                GroundedGoalHint(
                    goal_name=str(key),
                    goal_value=str(value),
                    confidence=max(0.66, float(getattr(routing, "confidence", 0.0)) * 0.92),
                    source="deterministic_structural_parser",
                    status="supported",
                    span=span,
                )
            )
            continue
        add_state(
            key,
            value,
            confidence=max(0.62, float(getattr(routing, "confidence", 0.0))),
            source="deterministic_structural_parser",
            hint_span=span,
        )

    if primary_kind == "structured_record":
        for hint in extract_entity_hints(
            segment,
            structured_pairs=state_pairs,
            relations=tuple(relation_hints),
            goals=tuple(goal_hints),
            span=span,
        ):
            add_entity(hint)
    elif primary_kind == "natural_structural":
        all_natural_units = tuple(
            unit
            for unit in tuple(structural_units or ())
            if str(getattr(unit, "unit_type", "") or "") in _NATURAL_SUPPORT_UNITS
        )
        speaker_turn_units = tuple(
            unit for unit in all_natural_units if str(getattr(unit, "unit_type", "") or "") == "speaker_turn"
        )
        citation_units = tuple(
            unit for unit in all_natural_units if str(getattr(unit, "unit_type", "") or "") == "citation_region"
        )
        natural_units = speaker_turn_units + citation_units if speaker_turn_units else all_natural_units
        for unit in natural_units:
            unit_text = (
                _speaker_turn_utterance(unit)
                if str(getattr(unit, "unit_type", "") or "") == "speaker_turn"
                else str(getattr(unit, "text", "") or "").strip()
            )
            if not unit_text:
                continue
            unit_span = getattr(unit, "span", None) or span
            unit_pairs = extract_structured_pairs(unit_text)
            unit_goals = extract_goal_hints(unit_text, structured_pairs=unit_pairs, span=unit_span)
            unit_goal_pairs = {
                (str(getattr(goal, "goal_name", "")), str(getattr(goal, "goal_value", "")))
                for goal in unit_goals
            }
            for key, value in unit_pairs:
                if (str(key), str(value)) in unit_goal_pairs:
                    continue
                add_state(
                    key,
                    value,
                    confidence=max(0.58, float(getattr(unit, "confidence", 0.0))),
                    source="natural_structural_parser",
                    hint_span=unit_span,
                )
            for hint in unit_goals:
                add_goal(hint)
            unit_relations = extract_relation_hints(unit_text, span=unit_span)
            for hint in unit_relations:
                add_relation(hint)
            for hint in extract_entity_hints(
                unit_text,
                structured_pairs=unit_pairs,
                relations=unit_relations,
                goals=unit_goals,
                span=unit_span,
            ):
                add_entity(hint)
            for hint in extract_event_hints(unit_text, relations=unit_relations, span=unit_span):
                add_event(hint)

    return (
        tuple(state_hints),
        tuple(relation_hints),
        tuple(goal_hints),
        tuple(entity_hints),
        tuple(event_hints),
    )


def _document_semantic_authority(segments: Tuple[GroundedTextSegment, ...]) -> Dict[str, float]:
    active_segments = tuple(segment for segment in segments if str(getattr(segment, "text", "") or "").strip())
    if not active_segments:
        return {
            "grounding_document_structural_primary_segments": 0.0,
            "grounding_document_structural_primary_ratio": 0.0,
            "grounding_document_structured_record_segments": 0.0,
            "grounding_document_natural_structural_segments": 0.0,
            "grounding_document_state_segments": 0.0,
            "grounding_document_relation_segments": 0.0,
            "grounding_document_goal_segments": 0.0,
            "grounding_document_event_segments": 0.0,
            "grounding_document_state_authority": 0.0,
            "grounding_document_relation_authority": 0.0,
            "grounding_document_goal_authority": 0.0,
            "grounding_document_event_authority": 0.0,
            "grounding_document_semantic_authority": 0.0,
        }
    total_segments = float(len(active_segments))
    primary_kinds = [
        _segment_primary_kind(
            getattr(segment, "routing", None),
            tuple(getattr(segment, "structural_units", ()) or ()),
        )
        for segment in active_segments
    ]
    structural_primary_segments = float(sum(1 for kind in primary_kinds if kind))
    structured_record_segments = float(sum(1 for kind in primary_kinds if kind == "structured_record"))
    natural_structural_segments = float(sum(1 for kind in primary_kinds if kind == "natural_structural"))
    state_segments = float(sum(1 for segment in active_segments if tuple(getattr(segment, "states", ()) or ())))
    relation_segments = float(sum(1 for segment in active_segments if tuple(getattr(segment, "relations", ()) or ())))
    goal_segments = float(sum(1 for segment in active_segments if tuple(getattr(segment, "goals", ()) or ())))
    event_segments = float(sum(1 for segment in active_segments if tuple(getattr(segment, "events", ()) or ())))
    primary_ratio = structural_primary_segments / max(total_segments, 1.0)
    state_ratio = state_segments / max(total_segments, 1.0)
    relation_ratio = relation_segments / max(total_segments, 1.0)
    goal_ratio = goal_segments / max(total_segments, 1.0)
    event_ratio = event_segments / max(total_segments, 1.0)
    structured_ratio = structured_record_segments / max(total_segments, 1.0)
    natural_ratio = natural_structural_segments / max(total_segments, 1.0)
    state_authority = 0.0
    if state_segments > 0.0 or structured_record_segments > 0.0:
        state_authority = _clip01(primary_ratio * (0.50 + (0.50 * max(state_ratio, structured_ratio))))
    relation_authority = 0.0
    if relation_segments > 0.0:
        relation_authority = _clip01(primary_ratio * (0.55 + (0.45 * max(relation_ratio, natural_ratio))))
    goal_authority = 0.0
    if goal_segments > 0.0:
        goal_authority = _clip01(primary_ratio * (0.55 + (0.45 * goal_ratio)))
    event_authority = 0.0
    if event_segments > 0.0:
        event_authority = _clip01(primary_ratio * (0.55 + (0.45 * max(event_ratio, natural_ratio))))
    semantic_authority = _clip01(max(state_authority, relation_authority, goal_authority, event_authority))
    return {
        "grounding_document_structural_primary_segments": float(structural_primary_segments),
        "grounding_document_structural_primary_ratio": float(primary_ratio),
        "grounding_document_structured_record_segments": float(structured_record_segments),
        "grounding_document_natural_structural_segments": float(natural_structural_segments),
        "grounding_document_state_segments": float(state_segments),
        "grounding_document_relation_segments": float(relation_segments),
        "grounding_document_goal_segments": float(goal_segments),
        "grounding_document_event_segments": float(event_segments),
        "grounding_document_state_authority": float(state_authority),
        "grounding_document_relation_authority": float(relation_authority),
        "grounding_document_goal_authority": float(goal_authority),
        "grounding_document_event_authority": float(event_authority),
        "grounding_document_semantic_authority": float(semantic_authority),
    }


def _modality_flag(profile: GroundingSourceProfile, modality: str) -> float:
    return 1.0 if str(profile.modality or "") == modality else 0.0


def _script_flag(profile: GroundingSourceProfile, script: str) -> float:
    return 1.0 if str(profile.script or "") == script else 0.0


def ground_text_document(
    text: str,
    *,
    language: str = "text",
    max_segments: int = 24,
    token_limit: int = 8,
    source_id: str = _DEFAULT_SOURCE_ID,
    document_id: str = _DEFAULT_DOCUMENT_ID,
    episode_id: str = _DEFAULT_EPISODE_ID,
) -> GroundedTextDocument:
    raw_text = str(text or "")
    normalized_text = _normalize_unicode_text(raw_text)
    source_id = str(source_id or _DEFAULT_SOURCE_ID)
    document_id = str(document_id or _DEFAULT_DOCUMENT_ID)
    episode_id = str(episode_id or _DEFAULT_EPISODE_ID)
    document_routing = infer_source_profile(normalized_text)
    segments_out: List[GroundedTextSegment] = []
    document_structural_units: List[GroundedStructuralUnit] = []
    for idx, (segment, span) in enumerate(
        split_text_segments_with_spans(
            raw_text,
            max_segments=max_segments,
            source_id=source_id,
            document_id=document_id,
            episode_id=episode_id,
        )
    ):
        normalized = _normalize_unicode_text(segment)
        segment_routing = infer_source_profile(normalized)
        state_pairs = extract_structured_pairs(segment)
        structural_units = extract_structural_units(
            segment,
            seg_idx=idx,
            span=span,
            routing=segment_routing,
            state_pairs=state_pairs,
        )
        segment_routing = _refine_segment_routing(
            document_routing,
            segment_routing,
            tuple(structural_units),
        )
        states, relations, goals, entities, events = _build_document_semantic_hints(
            segment,
            span=span,
            routing=segment_routing,
            structural_units=tuple(structural_units),
            state_pairs=state_pairs,
        )
        document_structural_units.extend(structural_units)
        segments_out.append(
            GroundedTextSegment(
                index=idx,
                text=segment,
                normalized_text=normalized,
                span=span,
                routing=segment_routing,
                tokens=tuple(tokenize_semantic_words(segment, limit=token_limit)),
                structural_units=tuple(structural_units),
                states=tuple(states),
                relations=tuple(relations),
                goals=tuple(goals),
                entities=tuple(entities),
                events=tuple(events),
                counterexample=is_counterexample_text(segment),
            )
        )
    structural_unit_counts: Dict[str, float] = {}
    for unit in document_structural_units:
        structural_unit_counts[unit.unit_type] = structural_unit_counts.get(unit.unit_type, 0.0) + 1.0
    char_coverage = float(
        sum(segment.span.char_length for segment in segments_out if segment.span is not None)
        / max(len(raw_text), 1)
    )
    raw_bytes = raw_text.encode("utf-8")
    byte_coverage = float(
        sum(segment.span.byte_length for segment in segments_out if segment.span is not None)
        / max(len(raw_bytes), 1)
    )
    metadata: Dict[str, float] = {
        "grounding_segments": float(len(segments_out)),
        "grounding_tokens": float(sum(len(segment.tokens) for segment in segments_out)),
        "grounding_structural_units": float(len(document_structural_units)),
        "grounding_state_hints": float(sum(len(segment.states) for segment in segments_out)),
        "grounding_relation_hints": float(sum(len(segment.relations) for segment in segments_out)),
        "grounding_goal_hints": float(sum(len(segment.goals) for segment in segments_out)),
        "grounding_entity_hints": float(sum(len(segment.entities) for segment in segments_out)),
        "grounding_event_hints": float(sum(len(segment.events) for segment in segments_out)),
        "grounding_counterexample_segments": float(sum(1 for segment in segments_out if segment.counterexample)),
        "grounding_multilingual": 1.0 if any(ord(ch) > 127 for ch in str(text or "")) else 0.0,
        "grounding_condition_hints": float(sum(1 for segment in segments_out for event in segment.events if str(event.condition or "").strip())),
        "grounding_explanation_hints": float(sum(1 for segment in segments_out for event in segment.events if str(event.explanation or "").strip())),
        "grounding_temporal_hints": float(sum(1 for segment in segments_out for event in segment.events if str(event.temporal or "").strip())),
        "grounding_modal_hints": float(sum(1 for segment in segments_out for event in segment.events if str(event.modality or "").strip())),
        "grounding_document_routing_confidence": float(document_routing.confidence),
        "grounding_document_routing_ambiguity": float(document_routing.ambiguity),
        "grounding_document_parser_candidates": float(len(document_routing.parser_candidates)),
        "grounding_segment_parser_candidates": float(
            sum(len(segment.routing.parser_candidates) for segment in segments_out if segment.routing is not None)
        ),
        "grounding_segment_code_segments": float(
            sum(1 for segment in segments_out if segment.routing is not None and segment.routing.modality == "code")
        ),
        "grounding_segment_natural_text_segments": float(
            sum(1 for segment in segments_out if segment.routing is not None and segment.routing.modality == "natural_text")
        ),
        "grounding_segment_structured_text_segments": float(
            sum(1 for segment in segments_out if segment.routing is not None and segment.routing.modality == "structured_text")
        ),
        "grounding_segment_mixed_segments": float(
            sum(1 for segment in segments_out if segment.routing is not None and segment.routing.modality == "mixed")
        ),
        "grounding_segment_unknown_segments": float(
            sum(1 for segment in segments_out if segment.routing is not None and segment.routing.modality == "unknown")
        ),
        "grounding_span_coverage": char_coverage,
        "grounding_span_char_coverage": char_coverage,
        "grounding_span_byte_coverage": byte_coverage,
        "grounding_span_segments_with_byte_offsets": float(
            sum(
                1
                for segment in segments_out
                if segment.span is not None
                and segment.span.byte_start is not None
                and segment.span.byte_end is not None
            )
        ),
        "grounding_structural_layer": 1.0,
        "grounding_structural_state_pairs": float(sum(len(segment.states) for segment in segments_out)),
        "grounding_document_modality_code": _modality_flag(document_routing, "code"),
        "grounding_document_modality_natural_text": _modality_flag(document_routing, "natural_text"),
        "grounding_document_modality_structured_text": _modality_flag(document_routing, "structured_text"),
        "grounding_document_modality_mixed": _modality_flag(document_routing, "mixed"),
        "grounding_document_modality_unknown": _modality_flag(document_routing, "unknown"),
        "grounding_document_script_latin": _script_flag(document_routing, "latin"),
        "grounding_document_script_cyrillic": _script_flag(document_routing, "cyrillic"),
        "grounding_document_script_mixed": _script_flag(document_routing, "mixed"),
        "grounding_document_script_unknown": _script_flag(document_routing, "unknown"),
    }
    metadata.update(
        {
            "grounding_clause_units": float(structural_unit_counts.get("clause", 0.0)),
            "grounding_speaker_turn_units": float(structural_unit_counts.get("speaker_turn", 0.0)),
            "grounding_citation_units": float(structural_unit_counts.get("citation_region", 0.0)),
            "grounding_key_value_units": float(structural_unit_counts.get("key_value_record", 0.0)),
            "grounding_json_record_units": float(structural_unit_counts.get("json_record", 0.0)),
            "grounding_log_entry_units": float(structural_unit_counts.get("log_entry", 0.0)),
            "grounding_table_row_units": float(structural_unit_counts.get("table_row", 0.0)),
            "grounding_section_header_units": float(structural_unit_counts.get("section_header", 0.0)),
        }
    )
    metadata.update(_document_semantic_authority(tuple(segments_out)))
    return GroundedTextDocument(
        language=language or "text",
        source_text=raw_text,
        source_id=source_id,
        document_id=document_id,
        episode_id=episode_id,
        routing=document_routing,
        structural_units=tuple(document_structural_units),
        segments=tuple(segments_out),
        metadata=metadata,
    )


def extract_relation_hints(segment: str, *, span=None):
    from .heuristic_backbone import extract_relation_hints as _extract_relation_hints

    return _extract_relation_hints(segment, span=span)


def extract_goal_hints(segment: str, *, structured_pairs=None, span=None):
    from .heuristic_backbone import extract_goal_hints as _extract_goal_hints

    return _extract_goal_hints(segment, structured_pairs=structured_pairs, span=span)


def extract_entity_hints(segment: str, *, structured_pairs=None, relations=None, goals=None, span=None):
    from .heuristic_backbone import extract_entity_hints as _extract_entity_hints

    return _extract_entity_hints(
        segment,
        structured_pairs=structured_pairs,
        relations=relations,
        goals=goals,
        span=span,
    )


def extract_event_hints(segment: str, *, relations=None, span=None):
    from .heuristic_backbone import extract_event_hints as _extract_event_hints

    return _extract_event_hints(segment, relations=relations, span=span)
