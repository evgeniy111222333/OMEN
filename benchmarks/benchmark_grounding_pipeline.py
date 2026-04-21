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


def _benchmark_pipeline(case: GroundingBenchmarkCase, *, iterations: int, max_segments: int) -> Dict[str, float]:
    latencies_ms: List[float] = []
    result = None
    for _ in range(max(1, iterations)):
        start = time.perf_counter()
        result = ground_text_to_symbolic(case.text, language=case.language, max_segments=max_segments)
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
    args = parser.parse_args()

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
    )
    for case in CASES:
        pipeline_stats = _benchmark_pipeline(case, iterations=args.iterations, max_segments=args.max_segments)
        trace_stats = _benchmark_trace(case, iterations=args.iterations)
        print(
            case.name.ljust(24),
            f"{pipeline_stats['pipeline_mean_ms']:.2f}".rjust(12),
            f"{pipeline_stats['pipeline_p95_ms']:.2f}".rjust(10),
            f"{trace_stats['trace_mean_ms']:.2f}".rjust(12),
            f"{trace_stats['trace_p95_ms']:.2f}".rjust(10),
            f"{pipeline_stats['char_cov']:.2f}".rjust(10),
            f"{pipeline_stats['byte_cov']:.2f}".rjust(10),
            f"{pipeline_stats['sem_auth']:.2f}".rjust(10),
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
        )


if __name__ == "__main__":
    main()
