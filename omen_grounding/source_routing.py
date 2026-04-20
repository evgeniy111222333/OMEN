from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Sequence, Tuple

from .types import GroundingParserCandidate, GroundingSourceProfile


_ROUTER_LANGUAGE_MARKERS: Dict[str, Tuple[Tuple[str, float], ...]] = {
    "python": (
        ("def ", 2.6),
        ("import ", 1.8),
        ("class ", 1.4),
        ("self.", 1.6),
        ("elif ", 1.2),
        ("lambda ", 1.0),
        ("__name__ == '__main__'", 1.8),
    ),
    "javascript": (
        ("function ", 2.1),
        ("const ", 2.0),
        ("let ", 1.7),
        ("=>", 1.5),
        ("console.log", 1.5),
        ("this.", 1.2),
        ("constructor(", 1.4),
        ("module.exports", 1.3),
    ),
    "typescript": (
        ("interface ", 2.2),
        (": string", 1.8),
        (": number", 1.8),
        (": boolean", 1.6),
        ("implements ", 1.2),
        ("readonly ", 1.1),
        (" as ", 0.8),
    ),
    "java": (
        ("public class", 2.3),
        ("private ", 1.4),
        ("protected ", 1.2),
        ("system.out", 1.8),
        ("public static void main", 2.3),
        ("new ", 0.8),
        ("@override", 1.2),
    ),
    "rust": (
        ("fn ", 2.4),
        ("let mut", 1.7),
        ("impl ", 1.6),
        ("::", 1.0),
        ("pub ", 1.2),
        ("->", 0.8),
        ("vec<", 0.8),
        ("&str", 1.2),
    ),
    "go": (
        ("package ", 2.2),
        ("func ", 2.1),
        (":=", 1.8),
        ("fmt.", 1.3),
        ("go ", 0.9),
        ("defer ", 1.1),
    ),
    "c": (
        ("#include", 2.0),
        ("printf(", 1.7),
        ("malloc(", 1.4),
        ("int main(", 2.0),
        ("typedef ", 1.1),
    ),
    "cpp": (
        ("#include", 1.4),
        ("std::", 2.0),
        ("cout <<", 2.0),
        ("vector<", 1.5),
        ("auto ", 1.0),
    ),
    "bash": (
        ("#!/bin/", 2.2),
        ("echo ", 1.2),
        (" fi", 0.7),
        (" then", 0.7),
        (" done", 0.7),
        ("$(", 1.0),
        ("$1", 1.0),
        ("export ", 1.1),
    ),
    "lua": (
        ("local ", 1.8),
        ("function ", 1.6),
        ("require(", 1.3),
        (" ipairs(", 1.0),
        (" nil", 0.8),
        (" then", 0.6),
    ),
}

_NATURAL_SECTION_HEADING_MARKERS = {
    "abstract",
    "introduction",
    "background",
    "method",
    "methods",
    "results",
    "discussion",
    "conclusion",
    "references",
    "summary",
}


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _weighted_marker_score(text: str, markers: Tuple[Tuple[str, float], ...]) -> float:
    score = 0.0
    for marker, weight in markers:
        if marker in text:
            score += weight
    return score


def _normalized_score_map(scores: Dict[str, float]) -> Dict[str, float]:
    clipped = {key: max(float(value), 0.0) for key, value in scores.items()}
    total = sum(clipped.values())
    if total <= 1e-8:
        return {key: 0.0 for key in clipped}
    return {key: round(value / total, 4) for key, value in clipped.items()}


def _best_scored_label(
    scores: Dict[str, float],
    *,
    default: str,
    min_score: float,
) -> str:
    if not scores:
        return default
    label, score = max(scores.items(), key=lambda item: item[1])
    return label if score >= min_score else default


def _split_inline_field(line: str) -> Tuple[str, str, str] | None:
    colon_idx = line.find(":")
    equals_idx = line.find("=")
    if colon_idx < 0 and equals_idx < 0:
        return None
    if colon_idx >= 0 and (equals_idx < 0 or colon_idx < equals_idx):
        sep = ":"
        idx = colon_idx
    else:
        sep = "="
        idx = equals_idx
    key = line[:idx].strip()
    value = line[idx + 1 :].strip()
    if not key or not value:
        return None
    return key, value, sep


def _looks_natural_section_heading_line(line: str) -> bool:
    split = _split_inline_field(line)
    if split is None:
        return False
    key, value, sep = split
    if sep != ":":
        return False
    if key.lower() not in _NATURAL_SECTION_HEADING_MARKERS:
        return False
    return len(re.findall(r"[A-Za-z]{3,}", value)) >= 2


def _looks_structured_field_line(line: str) -> bool:
    split = _split_inline_field(line)
    if split is None:
        return False
    key, value, sep = split
    lower_line = line.lower()
    if re.match(r"^(user|assistant|speaker|q|a)\s*[:>-]", lower_line):
        return False
    if sep == ":" and _looks_natural_section_heading_line(line):
        return False
    if len(key.split()) > 3:
        return False
    if re.search(r"[{}\[\]]", value):
        return True
    if "," in value or ";" in value or "\t" in value:
        return True
    if "=" in value:
        return True
    if re.search(r"\b(true|false|null|none|on|off|yes|no)\b", value.lower()):
        return True
    if re.search(r"https?://|postgres://|mysql://|sqlite://", value.lower()):
        return True
    if re.search(r"\d", value):
        return True
    if len(re.findall(r"[A-Za-z0-9_.-]+", value)) <= 4 and len(value) <= 32:
        return True
    return False


def _looks_comment_prose_line(line: str) -> bool:
    stripped = line.strip()
    marker = ""
    for candidate in ("//", "#", "/*", "*"):
        if stripped.startswith(candidate):
            marker = candidate
            break
    if not marker:
        return False
    content = stripped[len(marker) :].strip("/*- \t")
    if len(content) < 12:
        return False
    return len(re.findall(r"[A-Za-z]{3,}", content)) >= 2


def infer_script_profile(text: str) -> Tuple[str, Dict[str, float]]:
    scores = {
        "latin": 0.0,
        "cyrillic": 0.0,
        "digits_punct": 0.0,
        "other": 0.0,
    }
    for ch in str(text or ""):
        codepoint = ord(ch)
        if ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            scores["latin"] += 1.0
        elif 0x0400 <= codepoint <= 0x052F:
            scores["cyrillic"] += 1.0
        elif ch.isdigit() or ch.isspace() or re.match(r"[^\w\s]", ch):
            scores["digits_punct"] += 0.3
        else:
            scores["other"] += 1.0
    total_letters = scores["latin"] + scores["cyrillic"] + scores["other"]
    if total_letters <= 1e-8:
        return "unknown", {"latin": 0.0, "cyrillic": 0.0, "mixed": 0.0, "unknown": 1.0}
    letter_profile = _normalized_score_map(
        {
            "latin": scores["latin"],
            "cyrillic": scores["cyrillic"],
            "other": scores["other"],
        }
    )
    dominant = max(letter_profile.items(), key=lambda item: item[1])
    secondary = sorted(letter_profile.values(), reverse=True)[1] if len(letter_profile) >= 2 else 0.0
    mixed = 1.0 if dominant[1] < 0.85 and secondary > 0.15 else 0.0
    script = "mixed" if mixed > 0.0 else ("unknown" if dominant[1] < 0.20 else dominant[0])
    script_profile = {
        "latin": float(letter_profile.get("latin", 0.0)),
        "cyrillic": float(letter_profile.get("cyrillic", 0.0)),
        "mixed": float(mixed),
        "unknown": 0.0 if script != "unknown" else 1.0,
    }
    return script, script_profile


def _infer_structured_text_subtype(
    stripped: str,
    lower: str,
    probe: Sequence[str],
    *,
    json_like_records: int,
) -> str:
    log_score = 0.0
    dialogue_like_lines = sum(
        1
        for line in probe
        if re.match(r"^(user|assistant|speaker|q|a)\s*[:>-]", line.lower())
    )
    if re.search(r"traceback|exception|stack trace", lower):
        log_score += 1.6
    level_prefixed_lines = sum(
        1
        for line in probe
        if re.match(r"^(info|warn|warning|error|debug|trace)\b", line.lower())
    )
    if any(
        re.search(r"^\d{4}-\d{2}-\d{2}", line)
        or re.match(r"^(info|warn|warning|error|debug|trace)\b", line.lower())
        for line in probe
    ):
        log_score += 1.0
    if level_prefixed_lines >= 1:
        log_score += 1.2
    if any(
        re.match(r"^(info|warn|warning|error|debug|trace)\b", line.lower())
        and "=" in line
        for line in probe
    ):
        log_score += 1.2
    if sum(
        1
        for line in probe
        if re.search(r"^\d{4}-\d{2}-\d{2}", line)
        or re.search(r"\b(info|warn|warning|error|debug|trace)\b", line.lower())
    ) >= 2:
        log_score += 1.6

    table_score = 0.0
    if sum(1 for line in probe if "|" in line) >= 2:
        table_score += 1.8
    if sum(1 for line in probe if "\t" in line) >= 2:
        table_score += 1.6
    if sum(1 for line in probe if line.count(",") >= 2) >= 2:
        table_score += 1.2
    if any(line.count("|") >= 2 or line.count("\t") >= 2 or line.count(",") >= 3 for line in probe):
        table_score += 1.0

    config_score = 0.0
    if sum(1 for line in probe if re.match(r"^\[[^\]]+\]$", line.strip())) >= 1:
        config_score += 1.8
    if sum(1 for line in probe if _looks_structured_field_line(line)) >= 2:
        config_score += 1.4
    if any(re.match(r"^[A-Za-z0-9_.-]+\.[A-Za-z0-9_.-]+\s*=\s*.+$", line) for line in probe):
        config_score += 1.0
    if sum(1 for line in probe if line.startswith(("export ", "--", "set ", "env "))) >= 1:
        config_score += 1.0
    if dialogue_like_lines >= 2:
        config_score = max(config_score - 1.5, 0.0)

    state_score = 0.0
    state_marker_lines = sum(
        1
        for line in probe
        if re.search(r"\b(step|state|goal|target|status|next)\b", line.lower())
    )
    if state_marker_lines >= 1:
        state_score += 1.0
    if state_marker_lines >= 2:
        state_score += 1.0
    if any(re.match(r"^(step\d+|\d+\)|\d+\.)", line.lower()) for line in probe):
        state_score += 0.8

    key_value_score = 0.35 * sum(1 for line in probe if _looks_structured_field_line(line))
    if re.search(r"^\s*[\[{]", stripped):
        key_value_score += 0.6

    structured_scores = {
        "json_records": float(json_like_records) * 2.0 + (1.0 if stripped.startswith(("{", "[")) else 0.0),
        "log_text": log_score,
        "table_text": table_score,
        "config_text": config_score,
        "key_value_records": key_value_score + state_score,
    }
    return _best_scored_label(structured_scores, default="key_value_records", min_score=1.0)


def _infer_natural_text_subtype(
    stripped: str,
    lower: str,
    probe: Sequence[str],
    *,
    relation_hits: int,
) -> str:
    scientific_score = 0.0
    for marker, weight in (
        ("abstract", 1.1),
        ("introduction", 1.0),
        ("background", 1.0),
        ("method", 1.0),
        ("methods", 1.0),
        ("results", 1.0),
        ("discussion", 1.0),
        ("conclusion", 0.8),
        ("references", 0.7),
        ("doi", 0.8),
        ("dataset", 0.8),
        ("experiment", 0.8),
        ("baseline", 0.6),
        ("hypothesis", 0.7),
        ("statistically significant", 1.0),
    ):
        if marker in lower:
            scientific_score += weight
    if re.search(r"\[[0-9]{1,3}\]", stripped):
        scientific_score += 0.8
    if re.search(r"\bet al\.", lower):
        scientific_score += 0.8
    if re.search(r"\bp\s*[<=>]\s*0\.\d+", lower):
        scientific_score += 1.0

    instructional_score = 0.0
    for marker, weight in (
        ("step ", 0.8),
        ("how to", 1.0),
        ("install", 0.9),
        ("run", 0.6),
        ("usage", 0.8),
        ("follow", 0.7),
        ("first", 0.5),
        ("then", 0.5),
        ("finally", 0.6),
        ("must", 0.5),
        ("should", 0.5),
    ):
        if marker in lower:
            instructional_score += weight
    if sum(1 for line in probe if re.match(r"^(step|\d+\.)\s", line.lower())) >= 2:
        instructional_score += 1.0

    dialogue_score = 0.0
    if sum(1 for line in probe if re.match(r"^(user|assistant|speaker|q|a)\s*[:>-]", line.lower())) >= 1:
        dialogue_score += 1.6
    if stripped.count("?") >= 2:
        dialogue_score += 0.7
    if stripped.count('"') >= 4:
        dialogue_score += 0.4

    legal_score = 0.0
    for marker, weight in (
        ("hereby", 1.0),
        ("shall", 0.9),
        ("pursuant", 0.9),
        ("agreement", 0.7),
        ("contract", 0.8),
        ("section ", 0.6),
        ("article ", 0.6),
        ("whereas", 0.8),
        ("liability", 0.7),
    ):
        if marker in lower:
            legal_score += weight

    medical_score = 0.0
    for marker, weight in (
        ("patient", 0.8),
        ("diagnosis", 0.9),
        ("symptom", 0.7),
        ("treatment", 0.8),
        ("dose", 0.8),
        ("dosage", 0.8),
        ("contraindication", 0.9),
        ("trial", 0.6),
        ("clinical", 0.8),
    ):
        if marker in lower:
            medical_score += weight

    narrative_score = 0.0
    for marker, weight in (
        ("chapter", 0.6),
        ("suddenly", 0.6),
        ("she said", 0.7),
        ("he said", 0.7),
        ("they said", 0.7),
        ("once ", 0.5),
    ):
        if marker in lower:
            narrative_score += weight

    claim_score = 0.25 * float(relation_hits)
    if re.search(r"\btherefore\b|\bwe conclude\b|\bthis shows\b|\bclaim\b", lower):
        claim_score += 0.8

    natural_scores = {
        "scientific_text": scientific_score,
        "instructional_text": instructional_score,
        "dialogue_text": dialogue_score,
        "legal_text": legal_score,
        "medical_text": medical_score,
        "narrative_text": narrative_score,
        "claim_text": claim_score,
        "generic_text": 0.4,
    }
    return _best_scored_label(natural_scores, default="generic_text", min_score=0.8)


def verification_path_for_source(modality: str, subtype: str) -> str:
    if modality == "code":
        return "ast_program_verification"
    if modality == "mixed":
        return "mixed_hybrid_verification"
    if modality == "structured_text":
        if subtype == "log_text":
            return "log_trace_verification"
        if subtype == "config_text":
            return "config_schema_verification"
        if subtype == "table_text":
            return "table_consistency_verification"
        return "structured_state_verification"
    if modality == "natural_text":
        if subtype == "scientific_text":
            return "scientific_claim_verification"
        if subtype == "dialogue_text":
            return "dialogue_state_verification"
        return "natural_language_claim_verification"
    return "fallback_verification"


def build_parser_candidates(profile: GroundingSourceProfile) -> Tuple[GroundingParserCandidate, ...]:
    candidates: List[GroundingParserCandidate] = []
    modality = str(profile.modality or "unknown")
    subtype = str(profile.subtype or "unknown")
    confidence = float(profile.confidence)
    language = str(profile.language or "text")
    if modality == "code":
        candidates.append(
            GroundingParserCandidate(
                parser_name=f"ast_parser:{language}",
                confidence=max(0.55, confidence),
                role="primary",
            )
        )
        candidates.append(
            GroundingParserCandidate(
                parser_name="trace_builder",
                confidence=max(0.45, confidence * 0.85),
                role="support",
            )
        )
    elif subtype == "json_records":
        candidates.append(GroundingParserCandidate("json_parser", max(0.65, confidence), role="primary"))
        candidates.append(GroundingParserCandidate("json_schema_probe", max(0.45, confidence * 0.8), role="support"))
    elif subtype == "config_text":
        candidates.append(GroundingParserCandidate("kv_record_parser", max(0.62, confidence), role="primary"))
        candidates.append(GroundingParserCandidate("ini_section_parser", max(0.50, confidence * 0.82), role="support"))
    elif subtype == "key_value_records":
        candidates.append(GroundingParserCandidate("kv_record_parser", max(0.60, confidence), role="primary"))
    elif subtype == "log_text":
        candidates.append(GroundingParserCandidate("log_entry_parser", max(0.62, confidence), role="primary"))
        candidates.append(GroundingParserCandidate("timestamp_parser", max(0.50, confidence * 0.8), role="support"))
    elif subtype == "table_text":
        candidates.append(GroundingParserCandidate("table_row_parser", max(0.62, confidence), role="primary"))
        candidates.append(GroundingParserCandidate("delimiter_schema_probe", max(0.48, confidence * 0.8), role="support"))
    elif subtype == "dialogue_text":
        candidates.append(GroundingParserCandidate("speaker_turn_parser", max(0.58, confidence), role="primary"))
        candidates.append(GroundingParserCandidate("utterance_clause_segmenter", max(0.46, confidence * 0.8), role="support"))
    elif subtype == "instructional_text":
        candidates.append(GroundingParserCandidate("instruction_step_parser", max(0.58, confidence), role="primary"))
        candidates.append(GroundingParserCandidate("clause_segmenter", max(0.46, confidence * 0.8), role="support"))
    elif subtype == "scientific_text":
        candidates.append(GroundingParserCandidate("citation_span_parser", max(0.55, confidence), role="primary"))
        candidates.append(GroundingParserCandidate("section_role_parser", max(0.48, confidence * 0.82), role="support"))
    elif modality == "natural_text":
        candidates.append(GroundingParserCandidate("clause_segmenter", max(0.52, confidence), role="primary"))
        candidates.append(GroundingParserCandidate("discourse_marker_parser", max(0.45, confidence * 0.82), role="support"))
    elif modality == "mixed":
        candidates.append(GroundingParserCandidate("mixed_content_router", max(0.58, confidence), role="primary"))
        candidates.append(GroundingParserCandidate("code_block_probe", max(0.45, confidence * 0.8), role="support"))
        candidates.append(GroundingParserCandidate("clause_segmenter", max(0.42, confidence * 0.72), role="support"))
    else:
        candidates.append(GroundingParserCandidate("fallback_segment_router", max(0.35, confidence), role="fallback"))
    return tuple(candidates)


def infer_source_profile(
    text: str,
    *,
    parser_lang: Optional[str] = None,
    supported_languages: Sequence[str] = (),
) -> GroundingSourceProfile:
    stripped = str(text or "").strip()
    if not stripped:
        script, script_profile = infer_script_profile("")
        empty = GroundingSourceProfile(
            language="text",
            script=script,
            domain="empty",
            confidence=0.0,
            modality="unknown",
            subtype="empty",
            verification_path="fallback_verification",
            ambiguity=1.0,
            profile={"code": 0.0, "natural_text": 0.0, "structured_text": 0.0, "mixed": 0.0, "unknown": 1.0},
            script_profile=script_profile,
            evidence={"code_score": 0.0, "structured_score": 0.0, "observation_score": 0.0},
        )
        return GroundingSourceProfile(**{**empty.__dict__, "parser_candidates": build_parser_candidates(empty)})

    lower = stripped.lower()
    raw_lines = [line for line in stripped.splitlines() if line.strip()]
    lines = [line.strip() for line in raw_lines]
    probe = lines[: min(len(lines), 8)]
    supported = tuple(sorted(set(supported_languages) | set(_ROUTER_LANGUAGE_MARKERS.keys())))
    language_scores: Dict[str, float] = {
        lang: _weighted_marker_score(lower, _ROUTER_LANGUAGE_MARKERS.get(lang, ()))
        for lang in supported
    }

    if re.search(r"^\s*def\s+\w+\s*\(", stripped, flags=re.MULTILINE):
        language_scores["python"] = language_scores.get("python", 0.0) + 1.5
    if re.search(r"^\s*for\s+\w+\s+in\s+range\s*\(", stripped, flags=re.MULTILINE):
        language_scores["python"] = language_scores.get("python", 0.0) + 2.0
    if re.search(r"^\s*if\b.+:\s*$", stripped, flags=re.MULTILINE):
        language_scores["python"] = language_scores.get("python", 0.0) + 1.2
    if "print(" in lower:
        language_scores["python"] = language_scores.get("python", 0.0) + 0.8
    if any(line.endswith(":") for line in probe) and any(line.startswith(("    ", "\t")) for line in raw_lines[1:]):
        language_scores["python"] = language_scores.get("python", 0.0) + 1.0
    if re.search(r"^\s*class\s+\w+\s*\{", stripped, flags=re.MULTILINE):
        language_scores["javascript"] = language_scores.get("javascript", 0.0) + 0.8
        language_scores["typescript"] = language_scores.get("typescript", 0.0) + 0.8
    if re.search(r"^\s*fn\s+\w+\s*\(", stripped, flags=re.MULTILINE):
        language_scores["rust"] = language_scores.get("rust", 0.0) + 1.4
    if re.search(r"^\s*package\s+\w+", stripped, flags=re.MULTILINE):
        language_scores["go"] = language_scores.get("go", 0.0) + 1.2
    if re.search(r"^\s*#include\s+[<\"]", stripped, flags=re.MULTILINE):
        language_scores["c"] = language_scores.get("c", 0.0) + 1.2
        language_scores["cpp"] = language_scores.get("cpp", 0.0) + 1.2

    general_code_score = 0.0
    if any(token in stripped for token in ("{", "}", ";", "(", ")", "=>")):
        general_code_score += 0.8
    if any(line.startswith(("def ", "class ", "fn ", "function ", "package ", "#include")) for line in probe):
        general_code_score += 1.0
    if any(line.startswith(("    ", "\t")) for line in raw_lines[1:]):
        general_code_score += 0.5

    structured_score = 0.0
    json_like_records = 0
    dialogue_like_lines = sum(
        1
        for line in probe
        if re.match(r"^(user|assistant|speaker|q|a)\s*[:>-]", line.lower())
    )
    natural_heading_lines = sum(1 for line in probe if _looks_natural_section_heading_line(line))
    comment_prose_lines = sum(1 for line in raw_lines if _looks_comment_prose_line(line))
    for line in probe:
        if line.startswith("{") and line.endswith("}") and ":" in line:
            json_like_records += 1
            structured_score += 1.3
            try:
                json.loads(line)
                structured_score += 0.8
            except Exception:
                pass
        elif _looks_structured_field_line(line):
            structured_score += 0.5
        if line.count("|") >= 2 or line.count("\t") >= 2 or line.count(",") >= 3:
            structured_score += 0.8
    log_like_lines = sum(
        1
        for line in probe
        if re.search(r"^\d{4}-\d{2}-\d{2}", line)
        or re.search(r"\b(info|warn|warning|error|debug|trace)\b", line.lower())
    )
    if log_like_lines >= 1:
        structured_score += 1.0
    if log_like_lines >= 2:
        structured_score += 0.8
    config_like_lines = sum(
        1
        for line in probe
        if (
            re.match(r"^\[[^\]]+\]$", line.strip())
            or (
                not re.match(r"^(user|assistant|speaker|q|a)\s*[:>-]", line.lower())
                and re.match(r"^[A-Za-z0-9_.-]+\s*[:=]\s*.+$", line)
            )
        )
    )
    if config_like_lines >= 1:
        structured_score += 1.0
    if config_like_lines >= 2:
        structured_score += 0.6
    table_like_lines = sum(
        1
        for line in probe
        if "|" in line or "\t" in line or line.count(",") >= 2
    )
    if table_like_lines >= 1:
        structured_score += 1.0
    if table_like_lines >= 2:
        structured_score += 0.6
    if json_like_records >= 2:
        structured_score += 1.4
    if any(re.search(r"\b(step|state|goal|target|status|next)\b", line.lower()) for line in probe):
        structured_score += 1.1
    if dialogue_like_lines >= 2:
        structured_score = max(structured_score - 1.5, 0.0)
    if natural_heading_lines >= 1:
        structured_score = max(structured_score - 0.6 * float(natural_heading_lines), 0.0)

    relation_hits = len(re.findall(r"\b(is|becomes|causes|leads to|requires|must|not|after|before|however)\b", lower))
    sentence_hits = len(re.findall(r"[.!?](?:\s|$)", stripped))
    observation_score = min(float(relation_hits) * 0.45, 3.2) + min(float(sentence_hits) * 0.25, 1.5)
    if natural_heading_lines >= 1:
        observation_score += 0.9 + 0.35 * float(natural_heading_lines - 1)
    if comment_prose_lines >= 1:
        observation_score += 0.55 + 0.25 * float(comment_prose_lines - 1)
    if dialogue_like_lines >= 2:
        observation_score += 1.0

    ranked_languages = sorted(language_scores.items(), key=lambda item: item[1], reverse=True)
    top_lang, top_score = ranked_languages[0] if ranked_languages else ("python", 0.0)
    second_score = ranked_languages[1][1] if len(ranked_languages) > 1 else 0.0
    parser_supported = parser_lang if isinstance(parser_lang, str) and parser_lang in language_scores else None
    code_score = top_score + general_code_score
    if parser_supported is not None:
        code_score = max(code_score, language_scores.get(parser_supported, 0.0) + general_code_score + 0.4)

    domain_scores = {
        "code": code_score,
        "structured_observation": structured_score,
        "observation_text": observation_score,
    }

    short_structured_hint = (
        json_like_records >= 1
        or log_like_lines >= 1
        or config_like_lines >= 1
        or table_like_lines >= 1
    )
    if structured_score >= max(code_score * 1.05, observation_score * 0.95, 2.2) or (
        short_structured_hint and structured_score >= 1.4 and code_score < 1.0 and observation_score < 0.8
    ):
        language = "json" if json_like_records > 0 else "text"
        domain = "structured_observation"
        selected_score = structured_score
    elif observation_score >= max(code_score * 1.10, structured_score * 0.90, 1.5):
        language = "text"
        domain = "observation_text"
        selected_score = observation_score
    elif code_score >= 1.6:
        language = top_lang
        if parser_supported is not None and (
            top_score < 1.5 or language_scores.get(parser_supported, 0.0) >= top_score - 0.2
        ):
            language = parser_supported
        domain = "code"
        selected_score = code_score
    else:
        language = "text"
        domain = "text"
        selected_score = max(structured_score, observation_score, code_score)

    domain_runner_up = (
        max(score for key, score in domain_scores.items() if key != domain)
        if domain in domain_scores
        else 0.0
    )
    parser_agreement = 0.15 if parser_supported is not None and language == parser_supported else 0.0
    confidence = 0.35 + 0.08 * min(selected_score, 6.0) + 0.10 * max(top_score - second_score, 0.0)
    confidence += 0.08 * max(selected_score - domain_runner_up, 0.0) + parser_agreement
    confidence = max(0.05, min(confidence, 0.99))

    natural_text_score = max(observation_score, 0.6 if domain == "text" else 0.0)
    mixed_score = min(code_score, max(structured_score, natural_text_score))
    modality_scores = {
        "code": code_score,
        "natural_text": natural_text_score,
        "structured_text": structured_score,
        "mixed": mixed_score,
        "unknown": 0.8 if domain == "text" and max(code_score, structured_score, observation_score) < 1.2 else 0.0,
    }
    profile = _normalized_score_map(modality_scores)

    if domain == "code":
        primary_modality = "code"
    elif domain == "structured_observation":
        primary_modality = "structured_text"
    elif domain in ("observation_text", "text"):
        primary_modality = "natural_text"
    else:
        primary_modality = "unknown"

    modality = primary_modality
    mixed_noncode_signal = natural_text_score >= 0.4 or structured_score >= 2.0
    if domain != "empty" and code_score >= 1.6 and mixed_score >= 1.0 and mixed_noncode_signal:
        modality = "mixed"

    if modality == "code":
        subtype = "program_source"
    elif modality == "structured_text":
        subtype = _infer_structured_text_subtype(
            stripped,
            lower,
            probe,
            json_like_records=json_like_records,
        )
    elif modality == "natural_text":
        subtype = _infer_natural_text_subtype(
            stripped,
            lower,
            probe,
            relation_hits=relation_hits,
        )
    elif modality == "mixed":
        subtype = "mixed_code_structured" if structured_score >= natural_text_score else "mixed_code_text"
    else:
        subtype = "unknown"
    verification_path = verification_path_for_source(modality, subtype)

    evidence = {
        "code_score": round(code_score, 4),
        "structured_score": round(structured_score, 4),
        "observation_score": round(observation_score, 4),
        "natural_text_score": round(natural_text_score, 4),
        "mixed_score": round(mixed_score, 4),
        "top_language_score": round(top_score, 4),
        "second_language_score": round(second_score, 4),
        "parser_agreement": 1.0 if parser_supported is not None and language == parser_supported else 0.0,
    }
    ambiguity = _clip01(max(1.0 - confidence, profile.get("mixed", 0.0), profile.get("unknown", 0.0)))
    script, script_profile = infer_script_profile(stripped)
    result = GroundingSourceProfile(
        language=language,
        script=script,
        domain=domain,
        confidence=confidence,
        modality=modality,
        subtype=subtype,
        verification_path=verification_path,
        ambiguity=ambiguity,
        profile=profile,
        script_profile=script_profile,
        evidence=evidence,
    )
    return GroundingSourceProfile(
        **{**result.__dict__, "parser_candidates": build_parser_candidates(result)}
    )
