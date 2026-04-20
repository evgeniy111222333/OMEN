from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

from .source_routing import infer_source_profile
from .types import (
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


def _normalize_unicode_text(text: str) -> str:
    return unicodedata.normalize(
        "NFKC",
        _maybe_repair_mojibake(str(text or ""))
        .replace("’", "'")
        .replace("`", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("„", '"'),
    )


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


def _locate_segment_span(text: str, segment: str, start_cursor: int) -> GroundingSpan:
    raw = str(segment or "")
    if not raw:
        return GroundingSpan(start=max(0, int(start_cursor)), end=max(0, int(start_cursor)), text="")
    idx = text.find(raw, max(0, int(start_cursor)))
    if idx < 0:
        idx = text.find(raw)
    if idx < 0:
        stripped = raw.strip()
        idx = text.find(stripped, max(0, int(start_cursor))) if stripped else -1
        raw = stripped or raw
    if idx < 0:
        idx = max(0, int(start_cursor))
    end = min(len(text), idx + len(raw))
    return GroundingSpan(start=int(idx), end=int(end), text=text[idx:end])


def split_text_segments_with_spans(
    text: str,
    *,
    max_segments: int = 24,
) -> List[Tuple[str, GroundingSpan]]:
    out: List[Tuple[str, GroundingSpan]] = []
    cursor = 0
    for segment in split_text_segments(text, max_segments=max_segments):
        span = _locate_segment_span(str(text or ""), segment, cursor)
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
    start_at = 0
    idx = -1
    for _ in range(max(1, int(occurrence) + 1)):
        idx = parent_text.find(raw_fragment, start_at)
        if idx < 0:
            break
        start_at = idx + len(raw_fragment)
    if idx < 0:
        idx = parent_text.find(raw_fragment)
    if idx < 0:
        return None
    start = int(parent_span.start) + idx
    end = start + len(raw_fragment)
    return GroundingSpan(start=start, end=end, text=parent_text[idx : idx + len(raw_fragment)])


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
) -> GroundedTextDocument:
    normalized_text = _normalize_unicode_text(str(text or ""))
    document_routing = infer_source_profile(normalized_text)
    segments_out: List[GroundedTextSegment] = []
    document_structural_units: List[GroundedStructuralUnit] = []
    for idx, (segment, span) in enumerate(split_text_segments_with_spans(text, max_segments=max_segments)):
        normalized = _normalize_unicode_text(segment)
        segment_routing = infer_source_profile(normalized)
        state_pairs = extract_structured_pairs(segment)
        states = [
            GroundedStateHint(key=key, value=value, confidence=0.62, source="deterministic_structural_parser", status="supported", span=span)
            for key, value in state_pairs
        ]
        structural_units = extract_structural_units(
            segment,
            seg_idx=idx,
            span=span,
            routing=segment_routing,
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
                relations=tuple(),
                goals=tuple(),
                entities=tuple(),
                events=tuple(),
                counterexample=is_counterexample_text(segment),
            )
        )
    structural_unit_counts: Dict[str, float] = {}
    for unit in document_structural_units:
        structural_unit_counts[unit.unit_type] = structural_unit_counts.get(unit.unit_type, 0.0) + 1.0
    metadata: Dict[str, float] = {
        "grounding_segments": float(len(segments_out)),
        "grounding_tokens": float(sum(len(segment.tokens) for segment in segments_out)),
        "grounding_structural_units": float(len(document_structural_units)),
        "grounding_state_hints": float(sum(len(segment.states) for segment in segments_out)),
        "grounding_relation_hints": 0.0,
        "grounding_goal_hints": 0.0,
        "grounding_entity_hints": 0.0,
        "grounding_event_hints": 0.0,
        "grounding_counterexample_segments": float(sum(1 for segment in segments_out if segment.counterexample)),
        "grounding_multilingual": 1.0 if any(ord(ch) > 127 for ch in str(text or "")) else 0.0,
        "grounding_condition_hints": 0.0,
        "grounding_explanation_hints": 0.0,
        "grounding_temporal_hints": 0.0,
        "grounding_modal_hints": 0.0,
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
        "grounding_span_coverage": float(
            sum(
                max(0, int(segment.span.end) - int(segment.span.start))
                for segment in segments_out
                if segment.span is not None
            ) / max(len(str(text or "")), 1)
        ),
        "grounding_structural_layer": 1.0,
        "grounding_structural_state_pairs": float(sum(len(segment.states) for segment in segments_out)),
        "grounding_document_semantic_authority": 0.0,
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
    return GroundedTextDocument(
        language=language or "text",
        source_text=str(text or ""),
        routing=document_routing,
        structural_units=tuple(document_structural_units),
        segments=tuple(segments_out),
        metadata=metadata,
    )


def extract_relation_hints(segment: str):
    from .heuristic_backbone import extract_relation_hints as _extract_relation_hints

    return _extract_relation_hints(segment)


def extract_goal_hints(segment: str, *, structured_pairs=None):
    from .heuristic_backbone import extract_goal_hints as _extract_goal_hints

    return _extract_goal_hints(segment, structured_pairs=structured_pairs)


def extract_entity_hints(segment: str, *, structured_pairs=None, relations=None, goals=None):
    from .heuristic_backbone import extract_entity_hints as _extract_entity_hints

    return _extract_entity_hints(
        segment,
        structured_pairs=structured_pairs,
        relations=relations,
        goals=goals,
    )


def extract_event_hints(segment: str, *, relations=None):
    from .heuristic_backbone import extract_event_hints as _extract_event_hints

    return _extract_event_hints(segment, relations=relations)
