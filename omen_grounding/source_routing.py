from __future__ import annotations

import json
import re
from dataclasses import dataclass
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

_LANGUAGE_PRECEDENCE: Tuple[str, ...] = (
    "python",
    "typescript",
    "javascript",
    "rust",
    "go",
    "cpp",
    "c",
    "java",
    "bash",
    "lua",
)

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

_SCIENTIFIC_MARKERS = (
    "abstract",
    "introduction",
    "background",
    "method",
    "methods",
    "results",
    "discussion",
    "conclusion",
    "references",
    "doi",
    "dataset",
    "experiment",
    "baseline",
    "hypothesis",
    "statistically significant",
)
_INSTRUCTIONAL_MARKERS = (
    "step ",
    "how to",
    "install",
    "usage",
    "follow",
    "first",
    "then",
    "finally",
    "must",
    "should",
)
_LEGAL_MARKERS = (
    "hereby",
    "shall",
    "pursuant",
    "agreement",
    "contract",
    "section ",
    "article ",
)
_MEDICAL_MARKERS = (
    "patient",
    "diagnosis",
    "treatment",
    "symptom",
    "symptoms",
    "dose",
    "dosage",
    "mg",
    "clinical",
)
_NARRATIVE_MARKERS = (
    "once",
    "suddenly",
    "afterward",
    "remembered",
    "walked",
    "looked",
    "said",
)


@dataclass(frozen=True)
class _RoutingLedger:
    stripped: str
    lower: str
    raw_lines: Tuple[str, ...]
    lines: Tuple[str, ...]
    probe: Tuple[str, ...]
    parser_supported: Optional[str]
    json_like_records: int
    dialogue_like_lines: int
    natural_heading_lines: int
    comment_prose_lines: int
    structured_field_lines: int
    delimited_structured_lines: int
    section_header_lines: int
    log_like_lines: int
    config_like_lines: int
    table_like_lines: int
    instruction_like_lines: int
    state_marker_lines: int
    relation_hits: int
    sentence_hits: int
    scientific_hits: int
    legal_hits: int
    medical_hits: int
    narrative_hits: int
    code_block_starters: int
    code_punctuation_markers: int
    code_indent_lines: int
    code_body_lines: int
    language_marker_hits: Dict[str, int]
    language_weight_hits: Dict[str, float]
    language_regex_hits: Dict[str, int]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalized_score_map(scores: Dict[str, float]) -> Dict[str, float]:
    clipped = {key: max(float(value), 0.0) for key, value in scores.items()}
    total = sum(clipped.values())
    if total <= 1e-8:
        return {key: 0.0 for key in clipped}
    return {key: round(value / total, 4) for key, value in clipped.items()}


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


def _marker_presence_hits(text: str, markers: Tuple[Tuple[str, float], ...]) -> Tuple[int, float]:
    hits = 0
    weight_total = 0.0
    for marker, weight in markers:
        if marker in text:
            hits += 1
            weight_total += float(weight)
    return hits, weight_total


def _count_marker_hits(text: str, markers: Sequence[str]) -> int:
    return sum(1 for marker in markers if marker in text)


def _language_regex_hits(text: str) -> Dict[str, int]:
    hits: Dict[str, int] = {language: 0 for language in _ROUTER_LANGUAGE_MARKERS}
    if re.search(r"^\s*def\s+\w+\s*\(", text, flags=re.MULTILINE):
        hits["python"] += 1
    if re.search(r"^\s*for\s+\w+\s+in\s+range\s*\(", text, flags=re.MULTILINE):
        hits["python"] += 1
    if re.search(r"^\s*if\b.+:\s*$", text, flags=re.MULTILINE):
        hits["python"] += 1
    if "print(" in text:
        hits["python"] += 1
    if re.search(r"^\s*(raise|return)\b", text, flags=re.MULTILINE):
        hits["python"] += 1
    if re.search(r"\[[^\]]+\bfor\b[^\]]+\bin\b[^\]]+\]", text):
        hits["python"] += 1
    if re.search(r"^\s*class\s+\w+\s*\{", text, flags=re.MULTILINE):
        hits["javascript"] += 1
        hits["typescript"] += 1
    if re.search(r"^\s*fn\s+\w+\s*\(", text, flags=re.MULTILINE):
        hits["rust"] += 1
    if re.search(r"^\s*package\s+\w+", text, flags=re.MULTILINE):
        hits["go"] += 1
    if re.search(r"^\s*#include\s+[<\"]", text, flags=re.MULTILINE):
        hits["c"] += 1
        hits["cpp"] += 1
    return hits


def _build_routing_ledger(
    stripped: str,
    *,
    parser_lang: Optional[str],
    supported_languages: Sequence[str],
) -> _RoutingLedger:
    lower = stripped.lower()
    raw_lines = tuple(line for line in stripped.splitlines() if line.strip())
    lines = tuple(line.strip() for line in raw_lines)
    probe = lines[: min(len(lines), 8)]
    supported = tuple(sorted(set(supported_languages) | set(_ROUTER_LANGUAGE_MARKERS.keys())))

    language_marker_hits: Dict[str, int] = {}
    language_weight_hits: Dict[str, float] = {}
    regex_hits = _language_regex_hits(stripped)
    for language in supported:
        marker_hits, weight_total = _marker_presence_hits(lower, _ROUTER_LANGUAGE_MARKERS.get(language, ()))
        language_marker_hits[language] = marker_hits
        language_weight_hits[language] = weight_total
        if language in regex_hits:
            language_marker_hits[language] += regex_hits[language]
            language_weight_hits[language] += float(regex_hits[language])

    json_like_records = 0
    structured_field_lines = 0
    delimited_structured_lines = 0
    for line in probe:
        if line.startswith("{") and line.endswith("}") and ":" in line:
            json_like_records += 1
            continue
        if _looks_structured_field_line(line):
            structured_field_lines += 1
        if line.count("|") >= 2 or line.count("\t") >= 2 or line.count(",") >= 3:
            delimited_structured_lines += 1

    dialogue_like_lines = sum(
        1
        for line in probe
        if re.match(r"^(user|assistant|speaker|q|a)\s*[:>-]", line.lower())
    )
    natural_heading_lines = sum(1 for line in probe if _looks_natural_section_heading_line(line))
    comment_prose_lines = sum(1 for line in raw_lines if _looks_comment_prose_line(line))
    section_header_lines = sum(1 for line in probe if re.match(r"^\[[^\]]+\]$", line.strip()))
    log_like_lines = sum(
        1
        for line in probe
        if re.search(r"^\d{4}-\d{2}-\d{2}", line)
        or re.search(r"\b(info|warn|warning|error|debug|trace)\b", line.lower())
    )
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
    table_like_lines = sum(
        1
        for line in probe
        if "|" in line or "\t" in line or line.count(",") >= 2
    )
    instruction_like_lines = sum(
        1
        for line in probe
        if re.match(r"^(step|\d+\.)\s", line.lower()) or re.match(r"^(step\d+|\d+\)|\d+\.)", line.lower())
    )
    state_marker_lines = sum(
        1
        for line in probe
        if re.search(r"\b(step|state|goal|target|status|next)\b", line.lower())
    )
    relation_hits = len(
        re.findall(r"\b(is|becomes|causes|leads to|requires|must|not|after|before|however)\b", lower)
    )
    sentence_hits = len(re.findall(r"[.!?](?:\s|$)", stripped))
    scientific_hits = _count_marker_hits(lower, _SCIENTIFIC_MARKERS)
    if re.search(r"\[[0-9]{1,3}\]", stripped):
        scientific_hits += 1
    if re.search(r"\bet al\.", lower):
        scientific_hits += 1
    if re.search(r"\bp\s*[<=>]\s*0\.\d+", lower):
        scientific_hits += 1
    legal_hits = _count_marker_hits(lower, _LEGAL_MARKERS)
    medical_hits = _count_marker_hits(lower, _MEDICAL_MARKERS)
    narrative_hits = _count_marker_hits(lower, _NARRATIVE_MARKERS)
    code_block_starters = sum(
        1
        for line in probe
        if line.startswith(("def ", "class ", "fn ", "function ", "package ", "#include", "#!/"))
    )
    code_punctuation_markers = sum(1 for token in ("{", "}", ";", "(", ")", "=>") if token in stripped)
    code_indent_lines = sum(1 for line in raw_lines[1:] if line.startswith(("    ", "\t")))
    code_body_lines = sum(
        1
        for line in raw_lines
        if re.match(r"^\s*(raise|return|yield|break|continue)\b", line)
        or re.match(r"^\s+[A-Za-z_][A-Za-z0-9_]*\s*=", line)
        or re.search(r"\[[^\]]+\bfor\b[^\]]+\bin\b[^\]]+\]", line)
    )

    parser_supported = (
        parser_lang
        if isinstance(parser_lang, str) and parser_lang in _ROUTER_LANGUAGE_MARKERS
        else None
    )
    return _RoutingLedger(
        stripped=stripped,
        lower=lower,
        raw_lines=raw_lines,
        lines=lines,
        probe=probe,
        parser_supported=parser_supported,
        json_like_records=json_like_records,
        dialogue_like_lines=dialogue_like_lines,
        natural_heading_lines=natural_heading_lines,
        comment_prose_lines=comment_prose_lines,
        structured_field_lines=structured_field_lines,
        delimited_structured_lines=delimited_structured_lines,
        section_header_lines=section_header_lines,
        log_like_lines=log_like_lines,
        config_like_lines=config_like_lines,
        table_like_lines=table_like_lines,
        instruction_like_lines=instruction_like_lines,
        state_marker_lines=state_marker_lines,
        relation_hits=relation_hits,
        sentence_hits=sentence_hits,
        scientific_hits=scientific_hits,
        legal_hits=legal_hits,
        medical_hits=medical_hits,
        narrative_hits=narrative_hits,
        code_block_starters=code_block_starters,
        code_punctuation_markers=code_punctuation_markers,
        code_indent_lines=code_indent_lines,
        code_body_lines=code_body_lines,
        language_marker_hits=language_marker_hits,
        language_weight_hits=language_weight_hits,
        language_regex_hits=regex_hits,
    )


def _language_strength_tuple(ledger: _RoutingLedger, language: str) -> Tuple[int, int, float, int, int]:
    parser_match = 1 if ledger.parser_supported == language else 0
    marker_hits = int(ledger.language_marker_hits.get(language, 0))
    regex_hits = int(ledger.language_regex_hits.get(language, 0))
    weight_hits = float(ledger.language_weight_hits.get(language, 0.0))
    precedence = -_LANGUAGE_PRECEDENCE.index(language) if language in _LANGUAGE_PRECEDENCE else -999
    return (marker_hits, regex_hits, weight_hits, parser_match, precedence)


def _select_code_language(ledger: _RoutingLedger) -> Tuple[str, int, float, int, int]:
    languages = tuple(ledger.language_marker_hits.keys()) or _LANGUAGE_PRECEDENCE
    ranked = sorted(languages, key=lambda language: _language_strength_tuple(ledger, language), reverse=True)
    top = ranked[0] if ranked else "python"
    if ledger.parser_supported is not None:
        parser_tuple = _language_strength_tuple(ledger, ledger.parser_supported)
        top_tuple = _language_strength_tuple(ledger, top)
        if parser_tuple[0] >= max(top_tuple[0] - 1, 0) and parser_tuple[1] >= top_tuple[1]:
            top = ledger.parser_supported
    top_tuple = _language_strength_tuple(ledger, top)
    return top, top_tuple[0], top_tuple[2], top_tuple[1], top_tuple[3]


def _supports_for_ledger(
    ledger: _RoutingLedger,
    *,
    top_language_hits: int,
    top_language_weight: float,
    parser_match: int,
) -> Dict[str, float]:
    parser_bias = parser_match
    parser_only_hint = (
        parser_match >= 1
        and top_language_hits == 0
        and ledger.code_block_starters == 0
        and ledger.code_body_lines == 0
        and ledger.code_indent_lines == 0
    )
    strong_structured_carrier = (
        ledger.json_like_records >= 1
        or ledger.log_like_lines >= 2
        or ledger.table_like_lines >= 2
        or ledger.section_header_lines >= 1
        or ledger.structured_field_lines >= 2
    )
    if parser_only_hint and strong_structured_carrier:
        parser_bias = 0
    code_language_hits = top_language_hits
    code_language_weight = top_language_weight
    has_code_specific_signal = (
        top_language_hits >= 2
        or ledger.code_block_starters >= 1
        or ledger.code_body_lines >= 1
        or ledger.code_indent_lines >= 2
        or parser_bias >= 1
    )
    if not has_code_specific_signal:
        code_language_hits = 0
        code_language_weight = 0.0
    punctuation_support = min(ledger.code_punctuation_markers, 3) if has_code_specific_signal else 0
    code_support = float(
        code_language_hits
        + ledger.code_block_starters
        + punctuation_support
        + min(ledger.code_indent_lines, 2)
        + min(ledger.code_body_lines, 2)
        + parser_bias
    )
    structured_support = float(
        (3 * ledger.json_like_records)
        + (2 * ledger.log_like_lines)
        + (2 * ledger.table_like_lines)
        + (2 * ledger.section_header_lines)
        + ledger.config_like_lines
        + ledger.structured_field_lines
        + min(ledger.state_marker_lines, 2)
        + ledger.delimited_structured_lines
    )
    natural_support = float(
        (2 * ledger.dialogue_like_lines)
        + (2 * ledger.natural_heading_lines)
        + ledger.comment_prose_lines
        + min(ledger.relation_hits, 4)
        + min(ledger.sentence_hits, 3)
        + min(ledger.scientific_hits, 3)
        + min(ledger.instruction_like_lines, 2)
        + min(ledger.legal_hits, 2)
        + min(ledger.medical_hits, 2)
        + min(ledger.narrative_hits, 2)
    )
    mixed_support = 0.0
    non_code_support = max(structured_support, natural_support, float(ledger.comment_prose_lines))
    if code_support > 0.0 and non_code_support > 0.0:
        mixed_support = max(2.0, min(code_support, non_code_support) + 1.0)
    return {
        "code": round(code_support + (0.25 * code_language_weight), 4),
        "structured_text": round(structured_support, 4),
        "natural_text": round(natural_support, 4),
        "mixed": round(mixed_support, 4),
    }


def _has_code_route(ledger: _RoutingLedger, code_support: float) -> bool:
    strong_structured_carrier = (
        ledger.json_like_records >= 1
        or ledger.log_like_lines >= 2
        or ledger.table_like_lines >= 2
        or ledger.section_header_lines >= 1
        or ledger.structured_field_lines >= 2
    )
    parser_code_hint = (
        ledger.parser_supported is not None
        and ledger.parser_supported not in {"text", "json"}
        and not strong_structured_carrier
    )
    return bool(
        code_support >= 3.0
        or ledger.code_block_starters >= 1
        or (ledger.code_indent_lines >= 2 and ledger.code_body_lines >= 1)
        or (parser_code_hint and (ledger.code_punctuation_markers >= 1 or ledger.code_body_lines >= 1))
    )


def _has_structured_route(ledger: _RoutingLedger) -> bool:
    if ledger.stripped.startswith("#!/"):
        return False
    if ledger.json_like_records >= 1:
        return True
    if ledger.log_like_lines >= 2 or ledger.table_like_lines >= 2:
        return True
    if ledger.section_header_lines >= 1:
        return True
    code_like_signal = (
        ledger.code_block_starters >= 1
        or ledger.code_body_lines >= 1
        or ledger.code_punctuation_markers >= 3
        or max((int(value) for value in ledger.language_marker_hits.values()), default=0) >= 2
    )
    if code_like_signal:
        return False
    if ledger.config_like_lines >= 2 and ledger.dialogue_like_lines == 0 and ledger.natural_heading_lines == 0:
        return True
    if (
        ledger.structured_field_lines >= 1
        and ledger.dialogue_like_lines == 0
        and ledger.natural_heading_lines == 0
        and ledger.relation_hits == 0
        and ledger.sentence_hits == 0
        and ledger.comment_prose_lines == 0
    ):
        return True
    return (
        ledger.structured_field_lines >= 2
        and ledger.dialogue_like_lines == 0
        and ledger.natural_heading_lines == 0
    )


def _has_natural_route(ledger: _RoutingLedger, script: str) -> bool:
    if ledger.dialogue_like_lines >= 1:
        return True
    if ledger.natural_heading_lines >= 1 or ledger.scientific_hits >= 1:
        return True
    if ledger.instruction_like_lines >= 1 or ledger.relation_hits >= 1 or ledger.sentence_hits >= 1:
        return True
    if ledger.comment_prose_lines >= 1 or ledger.legal_hits >= 1 or ledger.medical_hits >= 1 or ledger.narrative_hits >= 1:
        return True
    return script in {"latin", "cyrillic", "mixed"} and bool(re.search(r"[A-Za-zА-Яа-яІіЇїЄєҐґ]", ledger.stripped))


def _has_mixed_route(ledger: _RoutingLedger, *, code_route: bool, structured_route: bool, natural_route: bool) -> bool:
    if not code_route:
        return False
    if structured_route:
        return True
    return bool(
        natural_route
        and (
            ledger.comment_prose_lines >= 1
            or ledger.natural_heading_lines >= 1
            or ledger.instruction_like_lines >= 1
            or ledger.dialogue_like_lines >= 1
            or ledger.relation_hits >= 1
        )
    )


def _infer_structured_text_subtype(ledger: _RoutingLedger) -> str:
    rules = (
        ("json_records", ledger.json_like_records >= 1),
        (
            "log_text",
            ledger.log_like_lines >= 1 and ledger.log_like_lines >= max(ledger.table_like_lines, ledger.config_like_lines),
        ),
        (
            "table_text",
            ledger.table_like_lines >= 2 and ledger.table_like_lines >= max(ledger.log_like_lines, ledger.config_like_lines),
        ),
        (
            "config_text",
            ledger.section_header_lines >= 1
            or (
                ledger.config_like_lines >= 2
                and ledger.delimited_structured_lines == 0
                and ledger.natural_heading_lines == 0
                and ledger.instruction_like_lines == 0
            ),
        ),
        ("key_value_records", ledger.structured_field_lines >= 1 or ledger.state_marker_lines >= 1),
    )
    for subtype, matched in rules:
        if matched:
            return subtype
    return "key_value_records"


def _infer_natural_text_subtype(ledger: _RoutingLedger) -> str:
    rules = (
        ("scientific_text", ledger.scientific_hits >= 1 or ledger.natural_heading_lines >= 1),
        ("dialogue_text", ledger.dialogue_like_lines >= 1),
        (
            "instructional_text",
            ledger.instruction_like_lines >= 1
            or (ledger.state_marker_lines >= 2 and ledger.sentence_hits == 0)
            or ("how to" in ledger.lower),
        ),
        ("legal_text", ledger.legal_hits >= 2),
        ("medical_text", ledger.medical_hits >= 2),
        ("narrative_text", ledger.narrative_hits >= 2),
        ("claim_text", ledger.relation_hits >= 1),
        ("generic_text", True),
    )
    for subtype, matched in rules:
        if matched:
            return subtype
    return "generic_text"


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
        if subtype == "legal_text":
            return "legal_clause_verification"
        if subtype == "medical_text":
            return "medical_fact_verification"
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
            evidence={"route_registry_version": 1.0, "code_score": 0.0, "structured_score": 0.0, "observation_score": 0.0},
        )
        return GroundingSourceProfile(**{**empty.__dict__, "parser_candidates": build_parser_candidates(empty)})

    ledger = _build_routing_ledger(
        stripped,
        parser_lang=parser_lang,
        supported_languages=supported_languages,
    )
    selected_language, top_language_hits, top_language_weight, _top_regex_hits, parser_match = _select_code_language(ledger)
    supports = _supports_for_ledger(
        ledger,
        top_language_hits=top_language_hits,
        top_language_weight=top_language_weight,
        parser_match=parser_match,
    )
    code_support = float(supports["code"])
    structured_support = float(supports["structured_text"])
    natural_support = float(supports["natural_text"])
    mixed_support = float(supports["mixed"])

    script, script_profile = infer_script_profile(stripped)
    code_route = _has_code_route(ledger, code_support)
    structured_route = _has_structured_route(ledger)
    natural_route = _has_natural_route(ledger, script)
    mixed_route = _has_mixed_route(
        ledger,
        code_route=code_route,
        structured_route=structured_route,
        natural_route=natural_route,
    )

    if mixed_route:
        modality = "mixed"
        domain = "code"
    elif code_route:
        modality = "code"
        domain = "code"
    elif structured_route:
        modality = "structured_text"
        domain = "structured_observation"
    elif natural_route:
        modality = "natural_text"
        domain = "observation_text" if natural_support > 0.0 else "text"
    else:
        modality = "unknown"
        domain = "text"

    if modality == "code":
        language = selected_language
    elif modality == "structured_text":
        language = "json" if ledger.json_like_records > 0 else "text"
    else:
        language = "text"

    if modality == "code":
        subtype = "program_source"
    elif modality == "structured_text":
        subtype = _infer_structured_text_subtype(ledger)
    elif modality == "natural_text":
        subtype = _infer_natural_text_subtype(ledger)
    elif modality == "mixed":
        subtype = (
            "mixed_code_structured"
            if (
                structured_route
                or ledger.instruction_like_lines >= 1
                or ledger.comment_prose_lines >= 1
                or ledger.state_marker_lines >= 1
            )
            else "mixed_code_text"
        )
    else:
        subtype = "unknown"

    verification_path = verification_path_for_source(modality, subtype)
    profile_scores = {
        "code": code_support,
        "natural_text": natural_support if modality != "unknown" else 0.0,
        "structured_text": structured_support,
        "mixed": mixed_support,
        "unknown": 1.0 if modality == "unknown" else 0.0,
    }
    profile = _normalized_score_map(profile_scores)
    selected_support = max(profile_scores.get(modality, 0.0), mixed_support if modality == "mixed" else 0.0)
    runner_up = max(
        value for key, value in profile_scores.items() if key != modality
    ) if len(profile_scores) > 1 else 0.0
    parser_agreement = 1.0 if ledger.parser_supported is not None and language == ledger.parser_supported else 0.0
    confidence = 0.35 + (0.08 * min(selected_support, 6.0)) + (0.06 * max(selected_support - runner_up, 0.0))
    confidence += 0.05 * parser_agreement
    if modality == "unknown":
        confidence = min(confidence, 0.25)
    confidence = max(0.05, min(confidence, 0.99))
    ambiguity = _clip01(max(1.0 - confidence, profile.get("mixed", 0.0), profile.get("unknown", 0.0)))

    evidence = {
        "route_registry_version": 1.0,
        "route_tie_break_precedence": 1.0,
        "route_rule_code": 1.0 if code_route else 0.0,
        "route_rule_structured_text": 1.0 if structured_route else 0.0,
        "route_rule_natural_text": 1.0 if natural_route else 0.0,
        "route_rule_mixed": 1.0 if mixed_route else 0.0,
        "feature_json_like_records": float(ledger.json_like_records),
        "feature_dialogue_like_lines": float(ledger.dialogue_like_lines),
        "feature_natural_heading_lines": float(ledger.natural_heading_lines),
        "feature_comment_prose_lines": float(ledger.comment_prose_lines),
        "feature_structured_field_lines": float(ledger.structured_field_lines),
        "feature_delimited_structured_lines": float(ledger.delimited_structured_lines),
        "feature_section_header_lines": float(ledger.section_header_lines),
        "feature_log_like_lines": float(ledger.log_like_lines),
        "feature_config_like_lines": float(ledger.config_like_lines),
        "feature_table_like_lines": float(ledger.table_like_lines),
        "feature_instruction_like_lines": float(ledger.instruction_like_lines),
        "feature_state_marker_lines": float(ledger.state_marker_lines),
        "feature_relation_hits": float(ledger.relation_hits),
        "feature_sentence_hits": float(ledger.sentence_hits),
        "feature_scientific_hits": float(ledger.scientific_hits),
        "feature_legal_hits": float(ledger.legal_hits),
        "feature_medical_hits": float(ledger.medical_hits),
        "feature_narrative_hits": float(ledger.narrative_hits),
        "feature_code_block_starters": float(ledger.code_block_starters),
        "feature_code_punctuation_markers": float(ledger.code_punctuation_markers),
        "feature_code_indent_lines": float(ledger.code_indent_lines),
        "feature_code_body_lines": float(ledger.code_body_lines),
        "top_language_score": float(top_language_hits),
        "second_language_score": float(
            sorted((float(value) for value in ledger.language_marker_hits.values()), reverse=True)[1]
            if len(ledger.language_marker_hits) > 1
            else 0.0
        ),
        "parser_agreement": parser_agreement,
        "code_score": round(code_support, 4),
        "structured_score": round(structured_support, 4),
        "observation_score": round(natural_support, 4),
        "natural_text_score": round(natural_support, 4),
        "mixed_score": round(mixed_support, 4),
    }

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
