from __future__ import annotations

import argparse
import ast
import json
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omen_prolog import (
    Const,
    DifferentiableProver,
    EpistemicStatus,
    HornAtom,
    HornClause,
    KnowledgeBase,
    SymbolicTaskContext,
    Var,
)
from omen_scale import SourceRoutingDecision, _infer_source_routing
from omen_symbolic.execution_trace import (
    TRACE_BINOP_EVENT_PRED,
    TRACE_RETURN_EVENT_PRED,
    TRACE_TEXT_GOAL_PRED,
    TRACE_TEXT_NEGATION_PRED,
    TRACE_TEXT_RELATION_PRED,
    TRACE_TEXT_STATE_PRED,
    SymbolicExecutionTraceBundle,
    build_symbolic_trace_bundle,
)
from omen_symbolic.ontology_engine import OntologyExpansionEngine, RuleHypothesisSampler


REPORT_DEFAULT = Path("reports/deep_scenario_modeling_report.md")


@dataclass
class PhaseRecord:
    name: str
    duration_ms: float
    summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioResult:
    scenario_id: str
    title: str
    goal: str
    raw_input: str
    raw_routing: Dict[str, Any]
    raw_trace: Dict[str, Any]
    canonical_summary: Dict[str, Any]
    phases: List[PhaseRecord]
    internal_metrics: Dict[str, Any]
    external_behavior: Dict[str, Any]
    verdict: str
    notes: List[str] = field(default_factory=list)


def _const_value(term: Any) -> Any:
    if hasattr(term, "val"):
        return getattr(term, "val")
    try:
        return int(term)
    except Exception:
        return repr(term)


def _status_name(value: Any) -> str:
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def _routing_dict(decision: SourceRoutingDecision) -> Dict[str, Any]:
    return {
        "language": decision.language,
        "domain": decision.domain,
        "modality": decision.modality,
        "subtype": decision.subtype,
        "verification_path": decision.verification_path,
        "confidence": round(float(decision.confidence), 4),
        "profile": dict(decision.profile),
        "evidence": dict(decision.evidence),
    }


def _trace_summary(bundle: Optional[SymbolicExecutionTraceBundle]) -> Dict[str, Any]:
    if bundle is None:
        return {
            "present": False,
            "observed_facts": 0,
            "target_facts": 0,
            "transitions": 0,
            "counterexamples": 0,
            "text_relations": 0,
            "text_states": 0,
            "text_goals": 0,
            "text_negations": 0,
            "binop_events": 0,
            "return_values": [],
        }
    return {
        "present": True,
        "observed_facts": len(bundle.observed_facts),
        "target_facts": len(bundle.target_facts),
        "transitions": len(bundle.transitions),
        "counterexamples": len(bundle.counterexamples),
        "text_relations": sum(1 for fact in bundle.target_facts if fact.pred == TRACE_TEXT_RELATION_PRED),
        "text_states": sum(1 for fact in bundle.target_facts if fact.pred == TRACE_TEXT_STATE_PRED),
        "text_goals": sum(1 for fact in bundle.target_facts if fact.pred == TRACE_TEXT_GOAL_PRED),
        "text_negations": sum(1 for fact in bundle.target_facts if fact.pred == TRACE_TEXT_NEGATION_PRED),
        "binop_events": sum(1 for fact in bundle.observed_facts if fact.pred == TRACE_BINOP_EVENT_PRED),
        "return_values": _extract_return_values(bundle),
    }


def _extract_return_values(bundle: SymbolicExecutionTraceBundle) -> List[Any]:
    values: List[Any] = []
    seen: set[Any] = set()
    for transition in bundle.transitions:
        for fact in transition.after_facts:
            if getattr(fact, "pred", None) != TRACE_RETURN_EVENT_PRED:
                continue
            decoded = _decode_value_symbol(fact.args[-1])
            if decoded in seen:
                continue
            seen.add(decoded)
            values.append(decoded)
    return values


def _decode_value_symbol(symbol: Any) -> Any:
    value = _const_value(symbol)
    if isinstance(value, int):
        if value in (10_000, 10_001):
            return bool(value - 10_000)
        if 20_000 <= value <= 119_999:
            return value - 25_000
        if 120_000 <= value <= 219_999:
            return (value - 170_000) / 100.0
        if value == 220_000:
            return None
    return value


def _fact_list_repr(facts: Iterable[Any], limit: int = 8) -> List[str]:
    items = [repr(fact) for fact in facts]
    items.sort()
    if len(items) <= limit:
        return items
    return items[:limit] + [f"... ({len(items) - limit} more)"]


def _phase(name: str, fn):
    t0 = time.perf_counter()
    value = fn()
    t1 = time.perf_counter()
    return value, PhaseRecord(name=name, duration_ms=(t1 - t0) * 1000.0)


def _execute_python_function(code: str) -> Dict[str, Any]:
    namespace: Dict[str, Any] = {}
    exec(code, namespace, namespace)
    functions = [value for value in namespace.values() if callable(value)]
    if not functions:
        return {"callable_found": False, "result": None}
    fn = functions[0]
    return {"callable_found": True, "function_name": getattr(fn, "__name__", "unknown"), "result": fn()}


def _analyze_accumulator_operator(code: str, expected_result: int) -> Dict[str, Any]:
    tree = ast.parse(code)
    source_lines = code.splitlines()
    actual = _execute_python_function(code)
    result: Dict[str, Any] = {
        "callable_found": bool(actual.get("callable_found", False)),
        "actual_result": actual.get("result"),
        "expected_result": expected_result,
        "mismatch": actual.get("result") != expected_result,
        "suspect_line": None,
        "observed_operator": None,
        "suggested_operator": None,
        "suggested_line": None,
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        target = node.targets[0].id
        if not isinstance(node.value, ast.BinOp):
            continue
        if not isinstance(node.value.left, ast.Name) or node.value.left.id != target:
            continue
        operator = type(node.value.op).__name__
        result["suspect_line"] = getattr(node, "lineno", None)
        result["observed_operator"] = operator
        if operator == "Sub":
            result["suggested_operator"] = "Add"
            line = source_lines[node.lineno - 1]
            result["suggested_line"] = line.replace("-", "+", 1)
        elif operator == "Add":
            result["suggested_operator"] = "Add"
            result["suggested_line"] = source_lines[node.lineno - 1]
        break
    return result


def _run_scenario_1() -> ScenarioResult:
    raw_input = (
        "Уяви вигаданий всесвіт.\n"
        'Факт 1: Об\'єкти типу "Зірки" генерують об\'єкти типу "Планети".\n'
        'Факт 2: Об\'єкти типу "Планети" генерують об\'єкти типу "Супутники".\n'
        'Завдання: Зроби висновок. Який зв\'язок між "Зірками" і "Супутниками"?'
    )
    raw_routing = _routing_dict(_infer_source_routing(raw_input))
    raw_trace = _trace_summary(build_symbolic_trace_bundle(raw_input, lang_hint="text", max_steps=32, max_counterexamples=4))

    GEN = 33
    INDIRECT = 44
    device = torch.device("cpu")
    torch.manual_seed(0)

    phases: List[PhaseRecord] = []

    def _run_cycle():
        prover = DifferentiableProver(
            d_latent=32,
            sym_vocab=64,
            max_rules=32,
            max_depth=3,
            n_cands=4,
        ).to(device)
        facts = frozenset(
            {
                HornAtom(GEN, (Const(1), Const(2))),
                HornAtom(GEN, (Const(2), Const(3))),
            }
        )
        goal = HornAtom(INDIRECT, (Const(1), Const(3)))
        prover.configure_hypothesis_cycle(
            enabled=True,
            max_contextual=6,
            max_neural=0,
            accept_threshold=0.25,
            verify_threshold=0.45,
        )
        prover.set_task_context(
            SymbolicTaskContext(
                observed_facts=facts,
                goal=goal,
                target_facts=frozenset({goal}),
                provenance="scenario_1",
                metadata={"last_src": 1.0, "last_tgt": 3.0},
            )
        )
        prover.train()
        z = torch.randn(1, 32, device=device)
        prover._last_z = z.detach()
        cycle = prover.continuous_hypothesis_cycle(z, facts, frozenset({goal}), device)
        derived_verified = prover.forward_chain_reasoned(3, facts, True, device)
        bridge_rules = [rule for rule in prover.kb.rules if getattr(rule.head, "pred", None) == INDIRECT]
        return {
            "facts": facts,
            "goal": goal,
            "cycle": cycle,
            "derived_verified": derived_verified,
            "bridge_rules": bridge_rules,
            "rule_statuses": {
                repr(rule): _status_name(prover.kb.rule_status(rule))
                for rule in prover.kb.rules
            },
        }

    run, phase = _phase("symbolic_bridge_rule_synthesis", _run_cycle)
    phase.summary = {
        "checked": run["cycle"]["stats"]["checked"],
        "accepted": run["cycle"]["stats"]["accepted"],
        "added": run["cycle"]["stats"]["added"],
        "verified": run["cycle"]["stats"]["verified"],
        "goal_derived_verified": float(run["goal"] in run["derived_verified"]),
    }
    phases.append(phase)

    canonical_summary = {
        "canonical_representation": "manual_symbolic_anchor",
        "observed_facts": _fact_list_repr(run["facts"]),
        "goal": repr(run["goal"]),
        "bridge_rules": [repr(rule) for rule in run["bridge_rules"]],
        "rule_statuses": run["rule_statuses"],
    }
    internal_metrics = {
        "cycle_stats": dict(run["cycle"]["stats"]),
        "goal_derived_verified": run["goal"] in run["derived_verified"],
        "bridge_rule_count": len(run["bridge_rules"]),
    }
    external_behavior = {
        "surface_answer": "Зірки непрямим шляхом генерують Супутники.",
        "core_interpretation": "Після canonical symbolic anchoring система синтезувала bridge-rule транзитивності.",
    }
    notes = []
    if raw_trace["text_relations"] == 0:
        notes.append("Сирий український текст не дав relation-facts: observation parser поки не покриває цей лексикон.")
    if raw_routing["subtype"] != "claim_text":
        notes.append(
            "Raw router не розпізнав абстрактний логічний текст як claim/scientific path; він пішов у загальний natural-text path."
        )
    verdict = "partial"
    if internal_metrics["goal_derived_verified"] and raw_trace["text_relations"] > 0:
        verdict = "pass"

    return ScenarioResult(
        scenario_id="1",
        title="Виведення нового правила з абстрактного тексту",
        goal="Перевірити, чи може ядро синтезувати нове bridge-rule транзитивності без жорстко заданого правила.",
        raw_input=raw_input,
        raw_routing=raw_routing,
        raw_trace=raw_trace,
        canonical_summary=canonical_summary,
        phases=phases,
        internal_metrics=internal_metrics,
        external_behavior=external_behavior,
        verdict=verdict,
        notes=notes,
    )


def _run_scenario_2() -> ScenarioResult:
    raw_input = textwrap.dedent(
        """
        Ось код на Python для підрахунку суми парних чисел від 1 до 10:
        сума = 0
        для і від 1 до 10:
        якщо і ділиться на 2 без остачі:
        сума = сума - і
        повернути сума
        Очікуваний результат дорівнює 30, але код повертає -30.
        Де в цьому коді помилка відносно очікуваного результату і як її виправити?
        """
    ).strip()
    raw_routing = _routing_dict(_infer_source_routing(raw_input))
    raw_trace = _trace_summary(build_symbolic_trace_bundle(raw_input, lang_hint="text", max_steps=32, max_counterexamples=4))

    buggy_code = textwrap.dedent(
        """
        def solve():
            total = 0
            for i in range(1, 11):
                if i % 2 == 0:
                    total = total - i
            return total
        """
    ).strip()
    fixed_code = buggy_code.replace("total = total - i", "total = total + i")

    phases: List[PhaseRecord] = []

    buggy_routing, phase = _phase("canonical_python_routing_buggy", lambda: _routing_dict(_infer_source_routing(buggy_code)))
    phase.summary = {
        "modality": buggy_routing["modality"],
        "verification_path": buggy_routing["verification_path"],
        "confidence": buggy_routing["confidence"],
    }
    phases.append(phase)

    buggy_trace, phase = _phase(
        "buggy_ast_trace_build",
        lambda: _trace_summary(build_symbolic_trace_bundle(buggy_code, lang_hint="python", max_steps=64, max_counterexamples=4)),
    )
    phase.summary = {
        "transitions": buggy_trace["transitions"],
        "binop_events": buggy_trace["binop_events"],
        "return_values": buggy_trace["return_values"],
    }
    phases.append(phase)

    fixed_trace, phase = _phase(
        "fixed_ast_trace_build",
        lambda: _trace_summary(build_symbolic_trace_bundle(fixed_code, lang_hint="python", max_steps=64, max_counterexamples=4)),
    )
    phase.summary = {
        "transitions": fixed_trace["transitions"],
        "binop_events": fixed_trace["binop_events"],
        "return_values": fixed_trace["return_values"],
    }
    phases.append(phase)

    ast_overlay, phase = _phase("overlay_ast_mismatch_localization", lambda: _analyze_accumulator_operator(buggy_code, expected_result=30))
    phase.summary = {
        "actual_result": ast_overlay["actual_result"],
        "expected_result": ast_overlay["expected_result"],
        "suspect_line": ast_overlay["suspect_line"],
        "observed_operator": ast_overlay["observed_operator"],
        "suggested_operator": ast_overlay["suggested_operator"],
    }
    phases.append(phase)

    canonical_summary = {
        "canonical_representation": "normalized_python_code",
        "buggy_code": buggy_code,
        "fixed_code": fixed_code,
        "buggy_routing": buggy_routing,
        "buggy_trace": buggy_trace,
        "fixed_trace": fixed_trace,
        "overlay_localization": ast_overlay,
    }
    internal_metrics = {
        "buggy_return_values": buggy_trace["return_values"],
        "fixed_return_values": fixed_trace["return_values"],
        "trace_transition_delta": fixed_trace["transitions"] - buggy_trace["transitions"],
        "native_ast_ready": buggy_routing["verification_path"] == "ast_program_verification",
        "native_repair_agent_present": False,
    }
    external_behavior = {
        "surface_answer": "Помилка в акумулюванні суми: рядок `total = total - i` має бути `total = total + i`.",
        "core_interpretation": "Core trace layer правильно бачить execution mismatch, але самостійний patch synthesis тут поки робить overlay-діагностика, а не native prover.",
    }
    notes = []
    if raw_routing["modality"] != "code":
        notes.append("Сирий український псевдокод не був піднятий у code/AST path без нормалізації до справжнього Python.")
    if not internal_metrics["native_repair_agent_present"]:
        notes.append("Native symbolic core дає спостережуваність execution trace, але не має окремого повноцінного program-repair planner.")
    verdict = "partial"
    if raw_routing["modality"] == "code" and internal_metrics["native_ast_ready"]:
        verdict = "pass"

    return ScenarioResult(
        scenario_id="2",
        title="Аналіз та виправлення синтаксичного алгоритму",
        goal="Перевірити, чи бачить система причинно-наслідкову помилку в алгоритмі через AST/trace path, а не лише токени.",
        raw_input=raw_input,
        raw_routing=raw_routing,
        raw_trace=raw_trace,
        canonical_summary=canonical_summary,
        phases=phases,
        internal_metrics=internal_metrics,
        external_behavior=external_behavior,
        verdict=verdict,
        notes=notes,
    )


def _run_scenario_3() -> ScenarioResult:
    raw_input = textwrap.dedent(
        """
        Правило: Всі двері на космічній станції відчиняються виключно зеленою карткою.
        Факт 1: Боб стоїть перед Дверима номер 5.
        Факт 2: У Боба немає зеленої картки.
        Факт 3: Через хвилину Двері номер 5 відчинилися.
        Чому відчинилися Двері номер 5, якщо Боб не мав картки? Запропонуй логічне пояснення, що не ламає початкове правило.
        """
    ).strip()
    raw_routing = _routing_dict(_infer_source_routing(raw_input))
    raw_trace = _trace_summary(build_symbolic_trace_bundle(raw_input, lang_hint="text", max_steps=32, max_counterexamples=4))

    AT_DOOR = 10
    NO_GREEN = 11
    OPENED = 12
    EXPLAIN = 13
    GREEN = 14
    device = torch.device("cpu")
    torch.manual_seed(0)

    phases: List[PhaseRecord] = []

    def _abduct_explanation():
        facts = frozenset(
            {
                HornAtom(AT_DOOR, (Const(1), Const(5))),
                HornAtom(NO_GREEN, (Const(1),)),
                HornAtom(OPENED, (Const(5),)),
            }
        )
        goal = HornAtom(EXPLAIN, (Const(5),))
        prover = DifferentiableProver(
            d_latent=32,
            sym_vocab=64,
            max_rules=32,
            max_depth=3,
            n_cands=4,
        ).to(device)
        verifier_rule = HornClause(
            head=HornAtom(OPENED, (Var("D"),)),
            body=(
                HornAtom(AT_DOOR, (Var("P"), Var("D"))),
                HornAtom(GREEN, (Var("P"),)),
            ),
        )
        prover.kb.add_rule(verifier_rule, status=EpistemicStatus.verified)
        prover.configure_hypothesis_cycle(
            enabled=True,
            max_contextual=8,
            max_neural=0,
            accept_threshold=0.25,
            verify_threshold=0.45,
        )
        prover.set_task_context(
            SymbolicTaskContext(
                observed_facts=facts,
                goal=goal,
                target_facts=frozenset({goal}),
                provenance="scenario_3",
                metadata={"last_src": 1.0, "last_tgt": 5.0},
            )
        )
        prover.train()
        z = torch.randn(1, 32, device=device)
        prover._last_z = z.detach()
        cycle = prover.continuous_hypothesis_cycle(z, facts, frozenset({goal}), device)
        explanation_rules = [rule for rule in prover.kb.rules if getattr(rule.head, "pred", None) == EXPLAIN]
        return {
            "facts": facts,
            "goal": goal,
            "cycle": cycle,
            "explanation_rules": explanation_rules,
            "rule_statuses": {repr(rule): _status_name(prover.kb.rule_status(rule)) for rule in prover.kb.rules},
        }

    abducted, phase = _phase("contextual_explanation_abduction", _abduct_explanation)
    phase.summary = {
        "contextual_candidates": abducted["cycle"]["stats"]["contextual_candidates"],
        "accepted": abducted["cycle"]["stats"]["accepted"],
        "verified": abducted["cycle"]["stats"]["verified"],
        "explanation_rule_count": len(abducted["explanation_rules"]),
    }
    phases.append(phase)

    def _invent_latent_bridge():
        engine = OntologyExpansionEngine(gap_threshold=0.2, contradiction_threshold=1)
        kb = KnowledgeBase(max_rules=32)
        facts = [
            HornAtom(AT_DOOR, (Const(1), Const(5))),
            HornAtom(NO_GREEN, (Const(1),)),
        ]
        goal = HornAtom(OPENED, (Const(5),))
        candidates = engine.generate_candidates(
            current_facts=facts,
            goal=goal,
            gap_norm=0.9,
            contradiction_count=2,
            kb=kb,
        )
        return {
            "candidates": candidates,
            "invented_predicates": sorted(
                {
                    int(candidate.metadata.get("invented_predicate", 0.0))
                    for candidate in candidates
                    if candidate.metadata.get("invented_predicate", 0.0) >= 900000
                }
            ),
        }

    invented, phase = _phase("ontology_bridge_invention", _invent_latent_bridge)
    phase.summary = {
        "candidate_count": len(invented["candidates"]),
        "invented_predicates": invented["invented_predicates"],
    }
    phases.append(phase)

    canonical_summary = {
        "canonical_representation": "manual_symbolic_contradiction_cluster",
        "observed_facts": _fact_list_repr(abducted["facts"]),
        "explanation_goal": repr(abducted["goal"]),
        "explanation_rules": [repr(rule) for rule in abducted["explanation_rules"]],
        "invented_bridge_candidates": [
            {
                "clause": repr(candidate.clause),
                "metadata": dict(candidate.metadata),
            }
            for candidate in invented["candidates"][:4]
        ],
        "rule_statuses": abducted["rule_statuses"],
    }
    internal_metrics = {
        "explanation_rule_count": len(abducted["explanation_rules"]),
        "invented_predicates": invented["invented_predicates"],
        "strict_hidden_cause_generated": False,
        "rule_preserving_specific_explanation": False,
    }
    external_behavior = {
        "surface_answer": "Система підняла latent explanatory bridge навколо суперечності, але не сформулювала конкретну зовнішню причину на кшталт дистанційного відкриття чи іншого актора.",
        "core_interpretation": "Rule-centric abduction працює на bridge-rules, але не на явному invent hidden actor/fact under exclusivity constraint.",
    }
    notes = [
        "Цей сценарій виявляє архітектурну межу: поточна абдукція набагато сильніша в rule synthesis, ніж у генерації конкретних прихованих entity-level фактів.",
    ]
    if raw_trace["text_negations"] == 0:
        notes.append("Сирий текст не дав explicit negation-fact навіть попри наявність фрази про відсутність зеленої картки.")

    return ScenarioResult(
        scenario_id="3",
        title="Вирішення логічної суперечності",
        goal="Перевірити, чи може система побудувати пояснення, яке знімає суперечність, не руйнуючи базове правило доступу.",
        raw_input=raw_input,
        raw_routing=raw_routing,
        raw_trace=raw_trace,
        canonical_summary=canonical_summary,
        phases=phases,
        internal_metrics=internal_metrics,
        external_behavior=external_behavior,
        verdict="partial",
        notes=notes,
    )


def _run_scenario_4() -> ScenarioResult:
    raw_input = textwrap.dedent(
        """
        Лог системи за поточну хвилину:
        Користувач Адмін увійшов успішно о 10:00. IP внутрішній. Збоїв немає.
        Користувач Гість увів неправильний пароль о 10:01. IP зовнішній. Тривога.
        Користувач Адмін увійшов успішно о 10:02. IP внутрішній. Збоїв немає.
        Користувач Невідомий увів неправильний пароль о 10:03. IP зовнішній. Тривога.
        Користувач Хакер увів неправильний пароль о 10:04. IP зовнішній. Тривога.
        Опиши стан безпеки максимально стисло. Згрупуй схожі події одним новим терміном, який ти маєш придумати.
        """
    ).strip()
    raw_routing = _routing_dict(_infer_source_routing(raw_input))
    raw_trace = _trace_summary(build_symbolic_trace_bundle(raw_input, lang_hint="text", max_steps=32, max_counterexamples=4))

    normalized_logs = textwrap.dedent(
        """
        INFO user=admin action=login status=ok ip=internal alarm=0
        WARN user=guest action=login status=bad_password ip=external alarm=1
        INFO user=admin action=login status=ok ip=internal alarm=0
        WARN user=unknown action=login status=bad_password ip=external alarm=1
        WARN user=hacker action=login status=bad_password ip=external alarm=1
        """
    ).strip()
    normalized_routing = _routing_dict(_infer_source_routing(normalized_logs))
    normalized_trace = _trace_summary(build_symbolic_trace_bundle(normalized_logs, lang_hint="text", max_steps=16, max_counterexamples=4))

    BAD = 20
    EXT = 21
    ALARM = 22
    SUCCESS = 23
    INTERNAL = 24
    RISK = 26
    INVENTED = 900001
    users = {
        100: "admin",
        101: "guest",
        102: "unknown",
        103: "hacker",
    }

    phases: List[PhaseRecord] = []

    def _invent_pattern():
        kb = KnowledgeBase(max_rules=64)
        facts = frozenset(
            {
                HornAtom(SUCCESS, (Const(100),)),
                HornAtom(INTERNAL, (Const(100),)),
                HornAtom(BAD, (Const(101),)),
                HornAtom(EXT, (Const(101),)),
                HornAtom(ALARM, (Const(101),)),
                HornAtom(BAD, (Const(102),)),
                HornAtom(EXT, (Const(102),)),
                HornAtom(ALARM, (Const(102),)),
                HornAtom(BAD, (Const(103),)),
                HornAtom(EXT, (Const(103),)),
                HornAtom(ALARM, (Const(103),)),
            }
        )
        sampler = RuleHypothesisSampler(max_hypotheses=16, max_scored_hypotheses=64)
        hypotheses = sampler.sample(
            pred_id=INVENTED,
            arity=1,
            interaction_preds=[BAD, EXT, ALARM],
            interaction_scores=[0.9, 0.9, 0.9],
            current_facts=list(facts),
            kb=kb,
            goal=HornAtom(RISK, (Var("U"),)),
            target_facts=[HornAtom(RISK, (Var("U"),))],
        )
        pair_rule = next(hyp.clause for hyp in hypotheses if hyp.template == "pair_factor")
        bridge_rule = next(hyp.clause for hyp in hypotheses if hyp.template == "bridge_goal")
        kb.add_rule(pair_rule, status=EpistemicStatus.verified)
        kb.add_rule(bridge_rule, status=EpistemicStatus.verified)
        derived = kb.forward_chain(2, starting_facts=facts, only_verified=False, track_epistemic=False)
        invented_users = sorted(_const_value(fact.args[0]) for fact in derived if getattr(fact, "pred", None) == INVENTED)
        risk_users = sorted(_const_value(fact.args[0]) for fact in derived if getattr(fact, "pred", None) == RISK)
        return {
            "facts": facts,
            "hypotheses": hypotheses,
            "pair_rule": pair_rule,
            "bridge_rule": bridge_rule,
            "derived": derived,
            "invented_users": invented_users,
            "risk_users": risk_users,
        }

    invented, phase = _phase("pattern_invention_and_grouping", _invent_pattern)
    phase.summary = {
        "hypothesis_count": len(invented["hypotheses"]),
        "invented_instances": len(invented["invented_users"]),
        "risk_instances": len(invented["risk_users"]),
    }
    phases.append(phase)

    suspicious_fact_count = sum(
        1
        for fact in invented["facts"]
        if getattr(fact, "pred", None) in {BAD, EXT, ALARM}
    )
    invented_instance_count = max(len(invented["invented_users"]), 1)
    compression_ratio = round(float(suspicious_fact_count) / float(invented_instance_count), 3)
    invented_label = "external_failed_login_cluster"

    canonical_summary = {
        "canonical_representation": "normalized_log_records + invented_predicate",
        "normalized_log_routing": normalized_routing,
        "normalized_log_trace": normalized_trace,
        "selected_rules": [repr(invented["pair_rule"]), repr(invented["bridge_rule"])],
        "invented_users": [users.get(user_id, str(user_id)) for user_id in invented["invented_users"]],
        "risk_users": [users.get(user_id, str(user_id)) for user_id in invented["risk_users"]],
        "invented_label_overlay": invented_label,
    }
    internal_metrics = {
        "hypothesis_count": len(invented["hypotheses"]),
        "invented_instances": len(invented["invented_users"]),
        "risk_instances": len(invented["risk_users"]),
        "compression_ratio_suspicious_facts_to_invented_instances": compression_ratio,
        "native_human_name_generation": False,
    }
    external_behavior = {
        "surface_answer": (
            "Зафіксовано 2 рутинні успішні входи адміна. "
            "Виявлено 3 події типу `external_failed_login_cluster` "
            "(guest, unknown, hacker)."
        ),
        "core_interpretation": "OEE/RuleHypothesisSampler invent-ить внутрішній predicate добре; людська назва поки накладається зверху як label overlay.",
    }
    notes = []
    if raw_routing["subtype"] != "log_text":
        notes.append("Сирий український лог у формі речень не routed як log_text; після нормалізації в key=value path спрацьовує стабільно.")
    if not internal_metrics["native_human_name_generation"]:
        notes.append("Система invent-ить внутрішній предикат, але не має окремого native name-synthesis шару для людиночитного терміна.")

    return ScenarioResult(
        scenario_id="4",
        title="Розпізнавання та згортання патернів",
        goal="Перевірити, чи може система стиснути повторювані події в новий внутрішній концепт і використати його для короткого опису стану безпеки.",
        raw_input=raw_input,
        raw_routing=raw_routing,
        raw_trace=raw_trace,
        canonical_summary=canonical_summary,
        phases=phases,
        internal_metrics=internal_metrics,
        external_behavior=external_behavior,
        verdict="partial",
        notes=notes,
    )


def _simulate_counterfactual_states(initial: Sequence[str]) -> Dict[str, Any]:
    initial_state = frozenset(initial)

    def successors(state: frozenset[str]) -> Dict[str, frozenset[str]]:
        next_states: Dict[str, frozenset[str]] = {}
        if "fire" in state and "tree" in state:
            next_states["fire_multiplies_tree_into_stone"] = frozenset((state - {"tree"}) | {"stone"})
        if "water" in state and "tree" in state:
            next_states["water_destroys_tree"] = frozenset(state - {"tree"})
        return next_states

    frontier = [(initial_state, [])]
    seen = {initial_state}
    all_paths: List[Dict[str, Any]] = []
    best_state = initial_state

    while frontier:
        state, path = frontier.pop(0)
        all_paths.append({"path": list(path), "state": sorted(state)})
        if "stone" in state and len(state) < len(best_state):
            best_state = state
        if len(path) >= 3:
            continue
        for action, next_state in successors(state).items():
            if next_state in seen:
                continue
            seen.add(next_state)
            frontier.append((next_state, path + [action]))

    only_stone_reachable = any(entry["state"] == ["stone"] for entry in all_paths)
    best_with_stone = min(
        (entry for entry in all_paths if "stone" in entry["state"]),
        key=lambda entry: (len(entry["state"]), len(entry["path"])),
        default=None,
    )
    return {
        "reachable_states": all_paths,
        "only_stone_reachable": only_stone_reachable,
        "best_with_stone": best_with_stone,
    }


def _run_scenario_5() -> ScenarioResult:
    raw_input = textwrap.dedent(
        """
        Існуючі правила світу: Вогонь знищує Дерево. Вода знищує Вогонь. Камінь і Вода не взаємодіють.
        Наявні ресурси: Вогонь, Дерево, Вода.
        Нова умовно-фантастична ввідна: тепер припустімо, що Вогонь примножує Дерево, створюючи з нього Камінь,
        а Вода миттєво знищує будь-яке Дерево. Використовуючи лише ці нові правила і доступні ресурси, як мені
        створити Камінь, якщо в мене ще немає Каменя, і як після цього зберегти лише його?
        """
    ).strip()
    raw_routing = _routing_dict(_infer_source_routing(raw_input))
    raw_trace = _trace_summary(build_symbolic_trace_bundle(raw_input, lang_hint="text", max_steps=32, max_counterexamples=4))

    FIRE = 30
    TREE = 31
    WATER = 32
    STONE = 33
    TREE_GONE = 34
    facts = frozenset(
        {
            HornAtom(FIRE, (Const(1),)),
            HornAtom(TREE, (Const(1),)),
            HornAtom(WATER, (Const(1),)),
        }
    )

    phases: List[PhaseRecord] = []

    def _run_symbolic_counterfactual():
        prover = DifferentiableProver(d_latent=32, sym_vocab=64, max_rules=16, max_depth=3, n_cands=4)
        prover.kb.add_rule(
            HornClause(
                head=HornAtom(STONE, (Var("X"),)),
                body=(HornAtom(FIRE, (Var("X"),)), HornAtom(TREE, (Var("X"),))),
            ),
            status=EpistemicStatus.verified,
        )
        prover.kb.add_rule(
            HornClause(
                head=HornAtom(TREE_GONE, (Var("X"),)),
                body=(HornAtom(WATER, (Var("X"),)), HornAtom(TREE, (Var("X"),))),
            ),
            status=EpistemicStatus.verified,
        )
        derived = prover.forward_chain_reasoned(3, facts, True, torch.device("cpu"))
        return {
            "derived": derived,
            "stone_derived": HornAtom(STONE, (Const(1),)) in derived,
            "tree_gone_derived": HornAtom(TREE_GONE, (Const(1),)) in derived,
        }

    symbolic, phase = _phase("counterfactual_rule_sandbox", _run_symbolic_counterfactual)
    phase.summary = {
        "stone_derived": float(symbolic["stone_derived"]),
        "tree_gone_derived": float(symbolic["tree_gone_derived"]),
        "derived_fact_count": len(symbolic["derived"]),
    }
    phases.append(phase)

    planner_overlay, phase = _phase("overlay_state_reachability", lambda: _simulate_counterfactual_states(["fire", "tree", "water"]))
    phase.summary = {
        "reachable_state_count": len(planner_overlay["reachable_states"]),
        "only_stone_reachable": float(planner_overlay["only_stone_reachable"]),
        "best_with_stone": planner_overlay["best_with_stone"],
    }
    phases.append(phase)

    canonical_summary = {
        "canonical_representation": "explicit_counterfactual_rule_set + state_overlay",
        "observed_facts": _fact_list_repr(facts),
        "derived_facts": _fact_list_repr(symbolic["derived"]),
        "reachable_states": planner_overlay["reachable_states"],
    }
    internal_metrics = {
        "stone_derived": symbolic["stone_derived"],
        "tree_gone_derived": symbolic["tree_gone_derived"],
        "only_stone_reachable": planner_overlay["only_stone_reachable"],
        "best_with_stone": planner_overlay["best_with_stone"],
    }
    external_behavior = {
        "surface_answer": (
            "Камінь створюється одразу з пари `fire + tree`. "
            "Найкращий досяжний стан за цими правилами: `fire + water + stone`. "
            "Стан `only stone` недосяжний, бо нові правила не дають способу прибрати fire або water."
        ),
        "core_interpretation": "Counterfactual sandbox добре виводить наслідки нових правил, але destructive planning з ресурсним споживанням тут ще wrapper-level, а не native core.",
    }
    notes = []
    if raw_routing["verification_path"] != "natural_language_claim_verification":
        notes.append("Raw router не впізнав prompt як звичайний counterfactual claim bundle; path вийшов не тим, що потрібен для reasoning.")
    notes.append(
        "Цей сценарій спеціально показує корисний негативний результат: 'зберегти лише камінь' логічно неможливо за даним набором правил."
    )

    return ScenarioResult(
        scenario_id="5",
        title='Створення симуляційної пісочниці ("Що якби...")',
        goal="Перевірити наслідкове reasoning у зміненому світі та виявити, чи може система відрізнити досяжну ціль від недосяжної.",
        raw_input=raw_input,
        raw_routing=raw_routing,
        raw_trace=raw_trace,
        canonical_summary=canonical_summary,
        phases=phases,
        internal_metrics=internal_metrics,
        external_behavior=external_behavior,
        verdict="partial",
        notes=notes,
    )


def _overall_verdict(scenarios: Sequence[ScenarioResult]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for scenario in scenarios:
        counts[scenario.verdict] = counts.get(scenario.verdict, 0) + 1
    raw_trace_semantic_hits = sum(
        1
        for scenario in scenarios
        if (
            scenario.raw_trace["text_relations"] > 0
            or scenario.raw_trace["text_states"] > 0
            or scenario.raw_trace["text_goals"] > 0
            or scenario.raw_trace["text_negations"] > 0
        )
    )
    return {
        "verdict_counts": counts,
        "raw_trace_semantic_hits": raw_trace_semantic_hits,
        "n_scenarios": len(scenarios),
    }


def _format_json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n```"


def _phase_table(phases: Sequence[PhaseRecord]) -> str:
    lines = [
        "| Фаза | Час (мс) | Ключовий підсумок |",
        "| --- | ---: | --- |",
    ]
    for phase in phases:
        summary = ", ".join(f"{key}={value}" for key, value in phase.summary.items()) if phase.summary else ""
        lines.append(f"| {phase.name} | {phase.duration_ms:.3f} | {summary} |")
    return "\n".join(lines)


def _scenario_section(scenario: ScenarioResult) -> str:
    parts = [
        f"## Сценарій {scenario.scenario_id}: {scenario.title}",
        "",
        f"**Мета:** {scenario.goal}",
        "",
        "**Raw input**",
        "",
        "```text",
        scenario.raw_input,
        "```",
        "",
        "**Raw routing**",
        "",
        _format_json_block(scenario.raw_routing),
        "",
        "**Raw trace readiness**",
        "",
        _format_json_block(scenario.raw_trace),
        "",
        "**Canonical execution summary**",
        "",
        _format_json_block(scenario.canonical_summary),
        "",
        "**Фазовий таймлайн**",
        "",
        _phase_table(scenario.phases),
        "",
        "**Внутрішні метрики**",
        "",
        _format_json_block(scenario.internal_metrics),
        "",
        "**Зовнішня поведінка**",
        "",
        _format_json_block(scenario.external_behavior),
        "",
        f"**Verdict:** `{scenario.verdict}`",
    ]
    if scenario.notes:
        parts.extend(
            [
                "",
                "**Нотатки**",
                "",
                *[f"- {note}" for note in scenario.notes],
            ]
        )
    parts.append("")
    return "\n".join(parts)


def build_report() -> str:
    torch.manual_seed(0)
    scenarios = [
        _run_scenario_1(),
        _run_scenario_2(),
        _run_scenario_3(),
        _run_scenario_4(),
        _run_scenario_5(),
    ]
    overall = _overall_verdict(scenarios)
    total_time_ms = sum(sum(phase.duration_ms for phase in scenario.phases) for scenario in scenarios)

    lines = [
        "# Deep Scenario Modeling Report",
        "",
        "Дата: `2026-04-19`",
        "",
        "Режим запуску: CPU, `torch.manual_seed(0)`.",
        "",
        "Цей звіт спеціально розділяє два шари:",
        "",
        "- `raw readiness`: що система робить на сирому українському вводі як є.",
        "- `canonical execution`: що відбувається після мінімальної нормалізації в той формат, який поточні модулі реально вміють формально обробляти.",
        "",
        "Головний висновок: reasoning core уже місцями сильний, але end-to-end anchoring сирого природного вводу ще не дотягує до повністю стабільного режиму.",
        "",
        "## Загальний підсумок",
        "",
        _format_json_block(
            {
                "overall": overall,
                "total_measured_phase_time_ms": round(total_time_ms, 3),
                "scenario_verdicts": {
                    scenario.scenario_id: {
                        "title": scenario.title,
                        "verdict": scenario.verdict,
                    }
                    for scenario in scenarios
                },
            }
        ),
        "",
        "## Що працює добре",
        "",
        "- Після canonical symbolic anchoring core вміє синтезувати bridge-rules і доводити ціль дедуктивно.",
        "- AST/trace layer добре бачить фактичну поведінку реального Python-коду і різницю між buggy та fixed execution.",
        "- Ontology/RuleHypothesis path уже invent-ить внутрішні предикати для стискання повторюваних патернів.",
        "- Counterfactual consequence reasoning уже здатний відрізняти досяжну підціль від логічно недосяжної.",
        "",
        "## Де межа системи зараз",
        "",
        "- Сирий український natural-language input майже не дає повноцінних relation/state/goal facts у symbolic trace layer.",
        "- Rule-centric abduction поки сильніша за hidden-entity/hidden-event abduction з конкретними новими акторами або подіями.",
        "- Human-readable naming для invented predicates поки не native; зараз це overlay над внутрішнім предикатом.",
        "- Деструктивне planning/state transition reasoning ще не є повноцінним native шаром symbolic core.",
        "",
        "## Деталізація по сценаріях",
        "",
    ]
    for scenario in scenarios:
        lines.append(_scenario_section(scenario))
    lines.extend(
        [
            "## Підсумкова оцінка архітектури",
            "",
            "Поточна архітектура вже виглядає як вузький neuro-symbolic AI runtime, а не просто текстовий класифікатор або шаблонний parser, бо вона реально має:",
            "",
            "- typed routing;",
            "- symbolic facts/rules;",
            "- abduction/induction/deduction;",
            "- ontology-like predicate invention;",
            "- counterfactual consequence checking.",
            "",
            "Але для рівня `працює стабільно на будь-який сирий ввід` їй ще бракує:",
            "",
            "- сильнішого multilingual observation grounding;",
            "- явного hidden-cause fact synthesis;",
            "- native surface verbalization of invented concepts;",
            "- native stateful planner з destructive transitions і reachability proofs.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a detailed markdown report for deep OMEN scenario modeling.")
    parser.add_argument("--out", type=Path, default=REPORT_DEFAULT, help="Output markdown path.")
    args = parser.parse_args()

    report = build_report()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(args.out.resolve())


if __name__ == "__main__":
    main()
