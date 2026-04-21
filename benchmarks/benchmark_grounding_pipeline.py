from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omen_grounding import ground_text_to_symbolic
from omen_symbolic.execution_trace import build_symbolic_trace_bundle


@dataclass(frozen=True)
class GroundingBenchmarkCase:
    name: str
    language: str
    text: str


CASES: Sequence[GroundingBenchmarkCase] = (
    GroundingBenchmarkCase(
        name="uk_conditional_coref",
        language="uk",
        text=(
            "Якщо тривога спрацювала, диспетчер відкриває двері о 10:00.\n"
            "Потім він відкриває шлюз, бо евакуація активна.\n"
            "Система повинна створювати безпечний вихід."
        ),
    ),
    GroundingBenchmarkCase(
        name="en_causal_repair",
        language="text",
        text=(
            "dispatcher opens door\n"
            "however dispatcher opens door failed\n"
            "at 10:00 dispatcher opens door because alarm triggered"
        ),
    ),
    GroundingBenchmarkCase(
        name="log_pattern_growth",
        language="log",
        text=(
            "user=guest result=failed_login ip=external alert=triggered\n"
            "user=unknown result=failed_login ip=external alert=triggered\n"
            "user=hacker result=failed_login ip=external alert=triggered\n"
            "user=admin result=success ip=internal alert=none"
        ),
    ),
    GroundingBenchmarkCase(
        name="instruction_sequence",
        language="text",
        text=(
            "step 1 open panel\n"
            "step 2 enable backup pump\n"
            "if pressure drops then trigger alarm\n"
            "goal stable cooling"
        ),
    ),
    GroundingBenchmarkCase(
        name="scientific_citation",
        language="text",
        text="Abstract: aspirin causes relief (Smith, 2024).",
    ),
    GroundingBenchmarkCase(
        name="rule_bridge",
        language="text",
        text="Rule all stars generate planets.",
    ),
    GroundingBenchmarkCase(
        name="hidden_cause_door",
        language="text",
        text="door opens but no green card",
    ),
    GroundingBenchmarkCase(
        name="uk_rule_exception",
        language="uk",
        text=(
            'Правило: "Зелена картка" відчиняє "Двері".\n'
            "Факт: У Боба немає зеленої картки."
        ),
    ),
)


def _percentile(values: Sequence[float], ratio: float) -> float:
    if not values:
        return 0.0
    ranked = sorted(float(value) for value in values)
    pos = min(max(int(round((len(ranked) - 1) * ratio)), 0), len(ranked) - 1)
    return float(ranked[pos])


def _language_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".py", ".pyw"}:
        return "python"
    if suffix == ".json":
        return "json"
    if suffix in {".log", ".out"}:
        return "log"
    if suffix in {".ini", ".cfg", ".conf", ".toml", ".yaml", ".yml"}:
        return "config"
    return "text"


def _load_file_cases(
    input_files: Sequence[str],
    input_globs: Sequence[str],
    *,
    file_char_limit: int = 0,
) -> List[GroundingBenchmarkCase]:
    file_paths: List[Path] = []
    seen: set[str] = set()
    for raw_path in input_files:
        path = Path(raw_path)
        resolved = path if path.is_absolute() else (ROOT / path)
        resolved = resolved.resolve()
        if resolved.is_file() and str(resolved) not in seen:
            seen.add(str(resolved))
            file_paths.append(resolved)
    for raw_glob in input_globs:
        for path in sorted(ROOT.glob(raw_glob)):
            resolved = path.resolve()
            if resolved.is_file() and str(resolved) not in seen:
                seen.add(str(resolved))
                file_paths.append(resolved)
    cases: List[GroundingBenchmarkCase] = []
    for idx, path in enumerate(file_paths):
        with path.open("r", encoding="utf-8") as handle:
            text = handle.read(file_char_limit) if file_char_limit > 0 else handle.read()
        cases.append(
            GroundingBenchmarkCase(
                name=f"file_{idx:02d}_{path.stem[:18]}",
                language=_language_for_path(path),
                text=text,
            )
        )
    return cases


def _benchmark_pipeline(
    case: GroundingBenchmarkCase,
    *,
    iterations: int,
    max_segments: int,
    allow_heuristic_fallback: bool = True,
) -> Dict[str, float]:
    latencies_ms: List[float] = []
    result = None
    for _ in range(max(1, iterations)):
        start = time.perf_counter()
        result = ground_text_to_symbolic(
            case.text,
            language=case.language,
            max_segments=max_segments,
            allow_heuristic_fallback=allow_heuristic_fallback,
        )
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
    assert result is not None
    return {
        "pipeline_mean_ms": float(statistics.mean(latencies_ms)),
        "pipeline_p95_ms": _percentile(latencies_ms, 0.95),
        "scene_entities": float(len(result.scene.entities)),
        "scene_events": float(len(result.scene.events)),
        "scene_claim_attributed": float(result.scene.metadata.get("scene_claim_attributed", 0.0)),
        "scene_claim_nonasserted": float(result.scene.metadata.get("scene_claim_nonasserted", 0.0)),
        "scene_coreference_links": float(result.scene.metadata.get("scene_coreference_links", 0.0)),
        "compiled_hypotheses": float(result.compiled.metadata.get("compiled_hypotheses", 0.0)),
        "compiled_attributed_hypotheses": float(result.compiled.metadata.get("compiled_attributed_hypotheses", 0.0)),
        "compiled_nonasserted_hypotheses": float(result.compiled.metadata.get("compiled_nonasserted_hypotheses", 0.0)),
        "compiled_candidate_rules": float(result.compiled.metadata.get("compiled_candidate_rules", 0.0)),
        "compiled_event_frames": float(result.compiled.metadata.get("compiled_event_frames", 0.0)),
        "world_state_records": float(result.world_state.metadata.get("grounding_world_state_records", 0.0)),
        "world_state_cited_records": float(result.world_state.metadata.get("grounding_world_state_cited_records", 0.0)),
        "world_state_rule_lifecycle_records": float(
            result.world_state.metadata.get("grounding_world_state_rule_lifecycle_records", 0.0)
        ),
        "world_state_nonasserted_records": float(
            result.world_state.metadata.get("grounding_world_state_nonasserted_records", 0.0)
        ),
        "verification_records": float(result.verification.metadata.get("verification_records", 0.0)),
        "verification_hidden_cause_records": float(
            result.verification.metadata.get("verification_hidden_cause_records", 0.0)
        ),
        "verification_nonasserted_pressure": float(
            result.verification.metadata.get("verification_nonasserted_pressure", 0.0)
        ),
        "world_state_hidden_cause_records": float(
            result.world_state.metadata.get("grounding_world_state_hidden_cause_records", 0.0)
        ),
        "char_cov": float(result.document.metadata.get("grounding_span_char_coverage", 0.0)),
        "byte_cov": float(result.document.metadata.get("grounding_span_byte_coverage", 0.0)),
        "sem_auth": float(result.document.metadata.get("grounding_document_semantic_authority", 0.0)),
        "fb_rate": float(result.scene.metadata.get("scene_heuristic_fallback_retained_rate", 0.0)),
        "fb_segments": float(result.scene.metadata.get("scene_heuristic_fallback_retained_segments", 0.0)),
        "fb_default": float(result.scene.metadata.get("scene_default_heuristic_backbone_active", 0.0)),
        "lb_active": float(result.scene.metadata.get("scene_learned_backbone_active", 0.0)),
        "lb_default": float(result.scene.metadata.get("scene_default_learned_backbone_active", 0.0)),
        "teacher_boot": float(result.scene.metadata.get("scene_bootstrap_teacher_active", 0.0)),
        "missing_backbone": float(result.scene.metadata.get("scene_missing_semantic_backbone", 0.0)),
    }


def _benchmark_trace(case: GroundingBenchmarkCase, *, iterations: int) -> Dict[str, float]:
    latencies_ms: List[float] = []
    bundle = None
    for _ in range(max(1, iterations)):
        start = time.perf_counter()
        bundle = build_symbolic_trace_bundle(case.text, lang_hint=case.language, max_steps=8, max_counterexamples=2)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
    assert bundle is not None
    return {
        "trace_mean_ms": float(statistics.mean(latencies_ms)),
        "trace_p95_ms": _percentile(latencies_ms, 0.95),
        "trace_grounding_facts": float(bundle.metadata.get("grounding_facts", 0.0)),
        "trace_interlingua_relations": float(bundle.metadata.get("interlingua_relations", 0.0)),
        "trace_validation_records": float(bundle.metadata.get("verifier_stack_records", 0.0)),
        "trace_hidden_cause_records": float(bundle.metadata.get("verification_hidden_cause_records", 0.0)),
        "trace_byte_span": float(bundle.metadata.get("grounding_byte_span_traceability", 0.0)),
        "parser_ag": float(bundle.metadata.get("grounding_parser_agreement", 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight grounding pipeline benchmark.")
    parser.add_argument("--iterations", type=int, default=12, help="Iterations per case.")
    parser.add_argument("--max-segments", type=int, default=8, help="Max segments for ground_text_to_symbolic.")
    parser.add_argument("--input-file", action="append", default=[], help="Extra input file to benchmark.")
    parser.add_argument("--input-glob", action="append", default=[], help="Extra input glob relative to repo root.")
    parser.add_argument(
        "--file-char-limit",
        type=int,
        default=0,
        help="Read at most this many characters from each extra input file (0 means full file).",
    )
    parser.add_argument(
        "--compare-no-fallback",
        action="store_true",
        help="Also run each case with implicit heuristic fallback disabled.",
    )
    parser.add_argument(
        "--skip-trace",
        action="store_true",
        help="Skip symbolic trace timing to keep runs lightweight on real files.",
    )
    args = parser.parse_args()
    cases = tuple(CASES) + tuple(
        _load_file_cases(
            args.input_file,
            args.input_glob,
            file_char_limit=max(0, int(args.file_char_limit)),
        )
    )

    print(f"grounding benchmark iterations={args.iterations} max_segments={args.max_segments}")
    print(
        "case".ljust(24),
        "pipe_mean_ms".rjust(12),
        "pipe_p95".rjust(10),
        "trace_mean".rjust(12),
        "trace_p95".rjust(10),
        "char_cov".rjust(10),
        "byte_cov".rjust(10),
        "sem_auth".rjust(10),
        "fb_rate".rjust(10),
        "fb_def".rjust(8),
        "lb_act".rjust(8),
        "lb_def".rjust(8),
        "teach".rjust(8),
        "par_ag".rjust(10),
        "byte_tr".rjust(10),
        "events".rjust(8),
        "coref".rjust(8),
        "attr".rjust(8),
        "nonasrt".rjust(8),
        "hcause".rjust(8),
        "hcwld".rjust(8),
        "rules".rjust(8),
        "hyp".rjust(8),
        "world".rjust(8),
        "cited".rjust(8),
        "rlife".rjust(8),
        "dep_ev".rjust(8),
        "dep_hyp".rjust(8),
        "dep_wld".rjust(8),
    )
    for case in cases:
        pipeline_stats = _benchmark_pipeline(case, iterations=args.iterations, max_segments=args.max_segments)
        if args.skip_trace:
            trace_stats = {
                "trace_mean_ms": 0.0,
                "trace_p95_ms": 0.0,
                "trace_grounding_facts": 0.0,
                "trace_interlingua_relations": 0.0,
                "trace_validation_records": 0.0,
                "trace_hidden_cause_records": 0.0,
                "trace_byte_span": 0.0,
                "parser_ag": 0.0,
            }
        else:
            trace_stats = _benchmark_trace(case, iterations=args.iterations)
        dep_events = 0.0
        dep_hypotheses = 0.0
        dep_world = 0.0
        if args.compare_no_fallback:
            no_fallback_stats = _benchmark_pipeline(
                case,
                iterations=args.iterations,
                max_segments=args.max_segments,
                allow_heuristic_fallback=False,
            )
            dep_events = pipeline_stats["scene_events"] - no_fallback_stats["scene_events"]
            dep_hypotheses = pipeline_stats["compiled_hypotheses"] - no_fallback_stats["compiled_hypotheses"]
            dep_world = pipeline_stats["world_state_records"] - no_fallback_stats["world_state_records"]
        print(
            case.name.ljust(24),
            f"{pipeline_stats['pipeline_mean_ms']:.2f}".rjust(12),
            f"{pipeline_stats['pipeline_p95_ms']:.2f}".rjust(10),
            f"{trace_stats['trace_mean_ms']:.2f}".rjust(12),
            f"{trace_stats['trace_p95_ms']:.2f}".rjust(10),
            f"{pipeline_stats['char_cov']:.2f}".rjust(10),
            f"{pipeline_stats['byte_cov']:.2f}".rjust(10),
            f"{pipeline_stats['sem_auth']:.2f}".rjust(10),
            f"{pipeline_stats['fb_rate']:.2f}".rjust(10),
            f"{pipeline_stats['fb_default']:.0f}".rjust(8),
            f"{pipeline_stats['lb_active']:.0f}".rjust(8),
            f"{pipeline_stats['lb_default']:.0f}".rjust(8),
            f"{pipeline_stats['teacher_boot']:.0f}".rjust(8),
            f"{trace_stats['parser_ag']:.2f}".rjust(10),
            f"{trace_stats['trace_byte_span']:.2f}".rjust(10),
            f"{pipeline_stats['scene_events']:.0f}".rjust(8),
            f"{pipeline_stats['scene_coreference_links']:.0f}".rjust(8),
            f"{pipeline_stats['compiled_attributed_hypotheses']:.0f}".rjust(8),
            f"{pipeline_stats['compiled_nonasserted_hypotheses']:.0f}".rjust(8),
            f"{pipeline_stats['verification_hidden_cause_records']:.0f}".rjust(8),
            f"{pipeline_stats['world_state_hidden_cause_records']:.0f}".rjust(8),
            f"{pipeline_stats['compiled_candidate_rules']:.0f}".rjust(8),
            f"{pipeline_stats['compiled_hypotheses']:.0f}".rjust(8),
            f"{pipeline_stats['world_state_records']:.0f}".rjust(8),
            f"{pipeline_stats['world_state_cited_records']:.0f}".rjust(8),
            f"{pipeline_stats['world_state_rule_lifecycle_records']:.0f}".rjust(8),
            f"{dep_events:.0f}".rjust(8),
            f"{dep_hypotheses:.0f}".rjust(8),
            f"{dep_world:.0f}".rjust(8),
        )


if __name__ == "__main__":
    main()
