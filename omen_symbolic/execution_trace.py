from __future__ import annotations

import ast
import json
import math
import re
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from omen_grounding import (
    CompiledSymbolicSegment,
    GroundingDocumentSummary,
    GroundingRuntimeContract,
    SemanticGroundingBackbone,
    compile_scene_context_graph_records,
    compile_scene_context_symbolic_atoms,
    compile_interlingua_graph_records,
    compile_ontology_symbolic_atoms,
    compile_scene_symbolic_atoms,
    compile_world_state_symbolic_atoms,
    extract_goal_hints,
    extract_relation_hints,
    extract_structured_pairs,
    ground_text_document,
    is_counterexample_text,
    normalize_symbol_text,
    run_grounding_orchestrator,
    split_text_segments,
    tokenize_semantic_words,
)


TRACE_STATE_VALUE_PRED = 480
TRACE_STATE_TYPE_PRED = 481
TRACE_ASSIGN_EVENT_PRED = 482
TRACE_RETURN_EVENT_PRED = 483
TRACE_ERROR_EVENT_PRED = 484
TRACE_TRANSITION_PRED = 485
TRACE_SCOPE_PRED = 486
TRACE_PARAM_BIND_PRED = 487
TRACE_BINOP_EVENT_PRED = 488
TRACE_CALL_EVENT_PRED = 489
TRACE_COUNTEREXAMPLE_PRED = 490
TRACE_PRIMARY_PRED = 491
TRACE_COMPARE_EVENT_PRED = 492
TRACE_TEXT_TOKEN_PRED = 493
TRACE_TEXT_RELATION_PRED = 494
TRACE_TEXT_NEGATION_PRED = 495
TRACE_TEXT_STATE_PRED = 496
TRACE_TEXT_GOAL_PRED = 497


@dataclass(frozen=True)
class TraceTransitionFacts:
    before_facts: FrozenSet[Any]
    after_facts: FrozenSet[Any]
    label: str = ""
    counterexample: bool = False


@dataclass(frozen=True)
class SymbolicExecutionTraceBundle:
    language: str
    source_text: str
    observed_facts: FrozenSet[Any]
    target_facts: FrozenSet[Any]
    transitions: Tuple[TraceTransitionFacts, ...]
    counterexamples: Tuple[TraceTransitionFacts, ...]
    grounding_facts: FrozenSet[Any] = field(default_factory=frozenset)
    grounding_target_facts: FrozenSet[Any] = field(default_factory=frozenset)
    grounding_hypotheses: Tuple[Any, ...] = field(default_factory=tuple)
    grounding_verification_records: Tuple[Any, ...] = field(default_factory=tuple)
    grounding_validation_records: Tuple[Any, ...] = field(default_factory=tuple)
    grounding_repair_actions: Tuple[Any, ...] = field(default_factory=tuple)
    grounding_world_state_records: Tuple[Any, ...] = field(default_factory=tuple)
    grounding_ontology_records: Tuple[Any, ...] = field(default_factory=tuple)
    grounding_ontology_facts: FrozenSet[Any] = field(default_factory=frozenset)
    grounding_world_state_active_facts: FrozenSet[Any] = field(default_factory=frozenset)
    grounding_world_state_hypothetical_facts: FrozenSet[Any] = field(default_factory=frozenset)
    grounding_world_state_contradicted_facts: FrozenSet[Any] = field(default_factory=frozenset)
    grounding_graph_records: Tuple[Any, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GroundingRuntimeArtifacts:
    language: str
    source_text: str
    runtime_contract: GroundingRuntimeContract = field(default_factory=GroundingRuntimeContract)
    segment_spans: Dict[int, Any] = field(default_factory=dict)
    grounding_facts: FrozenSet[Any] = field(default_factory=frozenset)
    grounding_target_facts: FrozenSet[Any] = field(default_factory=frozenset)
    grounding_hypotheses: Tuple[Any, ...] = field(default_factory=tuple)
    grounding_verification_records: Tuple[Any, ...] = field(default_factory=tuple)
    grounding_validation_records: Tuple[Any, ...] = field(default_factory=tuple)
    grounding_repair_actions: Tuple[Any, ...] = field(default_factory=tuple)
    grounding_world_state_records: Tuple[Any, ...] = field(default_factory=tuple)
    grounding_ontology_records: Tuple[Any, ...] = field(default_factory=tuple)
    grounding_ontology_facts: FrozenSet[Any] = field(default_factory=frozenset)
    grounding_world_state_active_facts: FrozenSet[Any] = field(default_factory=frozenset)
    grounding_world_state_hypothetical_facts: FrozenSet[Any] = field(default_factory=frozenset)
    grounding_world_state_contradicted_facts: FrozenSet[Any] = field(default_factory=frozenset)
    grounding_graph_records: Tuple[Any, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def schema_version(self) -> str:
        return str(getattr(self.runtime_contract, "schema_version", ""))

    @property
    def source_profile(self) -> Optional[Any]:
        return getattr(self.runtime_contract, "source_profile", None)

    @property
    def document_summary(self) -> GroundingDocumentSummary:
        document_summary = getattr(self.runtime_contract, "document", None)
        if isinstance(document_summary, GroundingDocumentSummary):
            return document_summary
        return GroundingDocumentSummary()


def _has_grounding_runtime_artifacts(artifacts: Optional[GroundingRuntimeArtifacts]) -> bool:
    if artifacts is None:
        return False
    return any(
        (
            artifacts.grounding_facts,
            artifacts.grounding_target_facts,
            artifacts.grounding_hypotheses,
            artifacts.grounding_verification_records,
            artifacts.grounding_validation_records,
            artifacts.grounding_repair_actions,
            artifacts.grounding_world_state_records,
            artifacts.grounding_ontology_records,
            artifacts.grounding_graph_records,
        )
    )


def _build_grounding_runtime_artifacts(
    orchestrated: Any,
    *,
    language: str,
    source_text: str,
    max_steps: int,
) -> GroundingRuntimeArtifacts:
    pipeline = orchestrated.pipeline
    document = pipeline.document
    document_routing = getattr(document, "routing", None)
    document_summary = GroundingDocumentSummary(
        routing=document_routing,
        segment_count=len(tuple(getattr(document, "segments", ()) or ())),
        structural_unit_count=len(tuple(getattr(document, "structural_units", ()) or ())),
        semantic_authority=float(getattr(document, "metadata", {}).get("grounding_document_semantic_authority", 0.0)),
        multilingual=float(getattr(document, "metadata", {}).get("grounding_multilingual", 0.0)),
        source_id=str(getattr(document, "source_id", "") or ""),
        document_id=str(getattr(document, "document_id", "") or ""),
        episode_id=str(getattr(document, "episode_id", "") or ""),
        char_coverage=float(getattr(document, "metadata", {}).get("grounding_span_char_coverage", 0.0)),
        byte_coverage=float(getattr(document, "metadata", {}).get("grounding_span_byte_coverage", 0.0)),
    )
    grounding_facts, grounding_targets, grounding_symbolic_stats = compile_scene_symbolic_atoms(pipeline.scene)
    grounding_context_facts, grounding_context_symbolic_stats = compile_scene_context_symbolic_atoms(
        pipeline.scene
    )
    grounding_graph_records, grounding_graph_stats = compile_interlingua_graph_records(
        pipeline.interlingua,
        max_records=max(max_steps * 4, 16),
    )
    grounding_context_records, grounding_context_stats = compile_scene_context_graph_records(
        pipeline.scene,
        max_records=max(max_steps * 3, 12),
    )
    (
        grounding_world_state_active_facts,
        grounding_world_state_hypothetical_facts,
        grounding_world_state_contradicted_facts,
        grounding_world_state_symbolic_stats,
    ) = compile_world_state_symbolic_atoms(pipeline.world_state.records)
    grounding_ontology_facts, grounding_ontology_stats = compile_ontology_symbolic_atoms(
        pipeline.ontology.concepts
    )
    return GroundingRuntimeArtifacts(
        language=language,
        source_text=source_text,
        runtime_contract=GroundingRuntimeContract(
            source_profile=document_routing,
            document=document_summary,
            grounding_mode="semantic_scene_compiler",
            orchestrator_active=True,
        ),
        segment_spans=dict(orchestrated.segment_spans),
        grounding_facts=frozenset(set(grounding_facts).union(grounding_context_facts)),
        grounding_target_facts=frozenset(grounding_targets),
        grounding_hypotheses=tuple(pipeline.compiled.hypotheses),
        grounding_verification_records=tuple(pipeline.verification.records),
        grounding_validation_records=tuple(pipeline.verifier_stack.validation_records),
        grounding_repair_actions=tuple(pipeline.verifier_stack.repair_actions),
        grounding_world_state_records=tuple(pipeline.world_state.records),
        grounding_ontology_records=tuple(pipeline.ontology.concepts),
        grounding_ontology_facts=grounding_ontology_facts,
        grounding_world_state_active_facts=grounding_world_state_active_facts,
        grounding_world_state_hypothetical_facts=grounding_world_state_hypothetical_facts,
        grounding_world_state_contradicted_facts=grounding_world_state_contradicted_facts,
        grounding_graph_records=tuple(grounding_graph_records) + tuple(grounding_context_records),
        metadata={
            **dict(pipeline.document.metadata),
            **dict(pipeline.scene.metadata),
            **dict(pipeline.interlingua.metadata),
            **dict(pipeline.compiled.metadata),
            **dict(pipeline.verification.metadata),
            **dict(pipeline.verifier_stack.metadata),
            **dict(pipeline.world_state.metadata),
            **dict(pipeline.ontology.metadata),
            **dict(grounding_world_state_symbolic_stats),
            **dict(grounding_ontology_stats),
            **dict(grounding_context_symbolic_stats),
            **dict(grounding_symbolic_stats),
            **dict(grounding_context_stats),
            **dict(grounding_graph_stats),
            **dict(orchestrated.metadata),
            "grounding_schema_version_v1": 1.0,
            "grounding_contract_document_segments": float(document_summary.segment_count),
            "grounding_contract_document_structural_units": float(document_summary.structural_unit_count),
            "grounding_contract_document_char_coverage": float(document_summary.char_coverage),
            "grounding_contract_document_byte_coverage": float(document_summary.byte_coverage),
            "grounding_contract_identity_present": float(
                all((document_summary.source_id, document_summary.document_id, document_summary.episode_id))
            ),
            "grounding_mode": "semantic_scene_compiler",
            "grounding_orchestrator_active": 1.0,
        },
    )


@dataclass
class _FunctionValue:
    name: str
    node: ast.FunctionDef


@dataclass
class _EventRecord:
    trace_key: str
    step_idx: int
    scope_name: str
    before_env: Dict[str, Any]
    after_env: Dict[str, Any]
    event: str
    detail_name: Optional[str] = None
    value: Any = None
    lhs: Any = None
    rhs: Any = None
    op_name: Optional[str] = None
    error_name: Optional[str] = None
    counterexample: bool = False


class _TraceRuntimeError(RuntimeError):
    def __init__(self, error_name: str):
        super().__init__(error_name)
        self.error_name = error_name


class _UnknownValue:
    pass


UNKNOWN = _UnknownValue()


def _stable_hash(text: str) -> int:
    return int(zlib.adler32(text.encode("utf-8")) & 0x7FFFFFFF)


def _scope_symbol(name: str) -> int:
    return 300_000 + (_stable_hash(f"scope:{name}") % 100_000)


def _var_symbol(name: str) -> int:
    return 400_000 + (_stable_hash(f"var:{name}") % 100_000)


def _type_symbol(name: str) -> int:
    return 500_000 + (_stable_hash(f"type:{name}") % 100_000)


def _op_symbol(name: str) -> int:
    return 600_000 + (_stable_hash(f"op:{name}") % 100_000)


def _error_symbol(name: str) -> int:
    return 700_000 + (_stable_hash(f"err:{name}") % 100_000)


def _lexeme_symbol(name: str) -> int:
    return 800_000 + (_stable_hash(f"lex:{name}") % 100_000)


def _step_symbol(trace_key: str, step_idx: int) -> int:
    base = _stable_hash(f"trace:{trace_key}") % 10_000
    return 100_000 + base * 128 + min(step_idx, 127)


def _value_symbol(value: Any) -> int:
    if isinstance(value, bool):
        return 10_000 + int(value)
    if isinstance(value, int):
        return 20_000 + max(min(value + 5_000, 99_999), 0)
    if isinstance(value, float) and math.isfinite(value):
        scaled = int(round(value * 100.0))
        return 120_000 + max(min(scaled + 50_000, 99_999), 0)
    if value is None:
        return 220_000
    if isinstance(value, str):
        return 230_000 + (_stable_hash(f"str:{value}") % 100_000)
    if isinstance(value, tuple):
        return 330_000 + (_stable_hash(f"tuple:{repr(value)}") % 100_000)
    if isinstance(value, list):
        return 430_000 + (_stable_hash(f"list:{repr(value)}") % 100_000)
    if isinstance(value, dict):
        return 530_000 + (_stable_hash(f"dict:{repr(sorted(value.items(), key=repr))}") % 100_000)
    if isinstance(value, set):
        return 630_000 + (_stable_hash(f"set:{repr(sorted(value, key=repr))}") % 100_000)
    if isinstance(value, _FunctionValue):
        return 730_000 + (_stable_hash(f"fn:{value.name}") % 100_000)
    return 830_000 + (_stable_hash(repr(value)) % 100_000)


def _type_name(value: Any) -> str:
    if isinstance(value, _FunctionValue):
        return "function"
    if value is UNKNOWN:
        return "unknown"
    if value is None:
        return "none"
    return type(value).__name__.lower()


def sample_function_trace_bindings(
    fn_node: ast.FunctionDef,
    max_counterexamples: int = 4,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    params = [arg.arg for arg in fn_node.args.args]
    if not params:
        return [{}], []
    iterable_params = set()
    indexed_params = set()
    mapping_params = set()
    for node in ast.walk(fn_node):
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Name):
            iterable_params.add(node.iter.id)
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            indexed_params.add(node.value.id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                owner_name = node.func.value.id
                if node.func.attr in {"get", "keys", "values", "items", "setdefault"}:
                    mapping_params.add(owner_name)
                if node.func.attr in {"append", "extend", "pop"}:
                    iterable_params.add(owner_name)

    primary: Dict[str, Any] = {}
    for idx, name in enumerate(params):
        if name in mapping_params:
            primary[name] = {"a": idx + 1, "b": idx + 2}
        elif name in iterable_params or name in indexed_params:
            primary[name] = [idx + 1, idx + 2, idx + 3]
        else:
            primary[name] = idx + 1
    counterexamples: List[Dict[str, Any]] = []

    for node in ast.walk(fn_node):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            names = []
            if isinstance(node.left, ast.Name) and node.left.id in primary:
                names.append(node.left.id)
            if isinstance(node.right, ast.Name) and node.right.id in primary:
                names.append(node.right.id)
            if len(names) >= 2:
                bad = dict(primary)
                bad[names[0]] = "a"
                bad[names[1]] = 2
                counterexamples.append(bad)
                break
            if len(names) == 1:
                bad = dict(primary)
                bad[names[0]] = "a"
                counterexamples.append(bad)
                break
        if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Name):
            if node.operand.id in primary:
                bad = dict(primary)
                bad[node.operand.id] = "a"
                counterexamples.append(bad)
                break
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Name):
            if node.iter.id in primary:
                bad = dict(primary)
                bad[node.iter.id] = 7
                counterexamples.append(bad)
                break
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id in primary:
                bad = dict(primary)
                bad[node.value.id] = "a"
                counterexamples.append(bad)
                break
    if max_counterexamples >= 0:
        counterexamples = counterexamples[: max_counterexamples]
    return [primary], counterexamples


class _PythonTraceBuilder:
    def __init__(self, max_steps: int = 24, max_counterexamples: int = 4):
        self.max_steps = max(1, int(max_steps))
        self.max_counterexamples = max(0, int(max_counterexamples))
        self.max_loop_iters = max(4, min(self.max_steps, 16))
        self._functions: Dict[str, _FunctionValue] = {}
        self._events: List[_EventRecord] = []
        self._trace_steps: Dict[str, int] = {}
        self._call_counter = 0

    def build(self, code: str) -> Optional[SymbolicExecutionTraceBundle]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None

        module_env: Dict[str, Any] = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                fn_value = _FunctionValue(node.name, node)
                self._functions[node.name] = fn_value
                module_env[node.name] = fn_value
        self._execute_block(
            tree.body,
            env=module_env,
            scope_name="<module>",
            trace_key="module:primary",
            counterexample=False,
        )

        for fn_name, fn_value in list(self._functions.items())[:4]:
            primary_bindings, counter_bindings = self._sample_function_inputs(fn_value.node)
            for idx, bindings in enumerate(primary_bindings[:1]):
                self._execute_function(
                    fn_value,
                    bindings,
                    trace_key=f"{fn_name}:primary:{idx}",
                    counterexample=False,
                )
            for idx, bindings in enumerate(counter_bindings[: self.max_counterexamples]):
                self._execute_function(
                    fn_value,
                    bindings,
                    trace_key=f"{fn_name}:counter:{idx}",
                    counterexample=True,
                )

        transitions, counterexamples, observed, targets = self._materialize_fact_bundle()
        if not transitions and not counterexamples:
            return None
        return SymbolicExecutionTraceBundle(
            language="python",
            source_text=code,
            observed_facts=observed,
            target_facts=targets,
            transitions=transitions,
            counterexamples=counterexamples,
            grounding_facts=frozenset(),
            grounding_target_facts=frozenset(),
            metadata={
                "grounding_mode": "python_ast_trace",
                "grounding_segments": float(len(transitions)),
                "grounding_tokens": 0.0,
                "grounding_state_hints": 0.0,
                "grounding_relation_hints": 0.0,
                "grounding_goal_hints": 0.0,
                "grounding_counterexample_segments": float(len(counterexamples)),
                "grounding_multilingual": 0.0,
            },
        )

    def _sample_function_inputs(
        self,
        fn_node: ast.FunctionDef,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        return sample_function_trace_bindings(
            fn_node,
            max_counterexamples=self.max_counterexamples,
        )

    def _execute_function(
        self,
        fn_value: _FunctionValue,
        bindings: Dict[str, Any],
        trace_key: str,
        counterexample: bool,
    ) -> Any:
        self._call_counter += 1
        env = dict(bindings)
        self._record_event(
            trace_key=trace_key,
            scope_name=fn_value.name,
            before_env={},
            after_env=env,
            event="call",
            detail_name=fn_value.name,
            counterexample=counterexample,
        )
        for name, value in bindings.items():
            self._record_event(
                trace_key=trace_key,
                scope_name=fn_value.name,
                before_env=dict(env),
                after_env=dict(env),
                event="param",
                detail_name=name,
                value=value,
                counterexample=counterexample,
            )
        result, _ = self._execute_block(
            fn_value.node.body,
            env=env,
            scope_name=fn_value.name,
            trace_key=trace_key,
            counterexample=counterexample,
        )
        return result

    def _execute_block(
        self,
        statements: Sequence[ast.stmt],
        env: Dict[str, Any],
        scope_name: str,
        trace_key: str,
        counterexample: bool,
    ) -> Tuple[Any, Optional[str]]:
        for stmt in statements:
            result, control = self._execute_stmt(
                stmt,
                env=env,
                scope_name=scope_name,
                trace_key=trace_key,
                counterexample=counterexample,
            )
            if control is not None:
                return result, control
        return None, None

    def _execute_stmt(
        self,
        stmt: ast.stmt,
        env: Dict[str, Any],
        scope_name: str,
        trace_key: str,
        counterexample: bool,
    ) -> Tuple[Any, Optional[str]]:
        if self._trace_steps.get(trace_key, 0) >= self.max_steps:
            return None, "halt"

        if isinstance(stmt, ast.FunctionDef):
            fn_value = _FunctionValue(stmt.name, stmt)
            self._functions[stmt.name] = fn_value
            env[stmt.name] = fn_value
            return None, None

        before_env = dict(env)
        try:
            if isinstance(stmt, ast.Assign):
                value = self._eval_expr(stmt.value, env)
                target_name = self._primary_target_name(stmt.targets[0])
                self._assign_value(stmt.targets[0], value, env)
                op_name, lhs, rhs = self._expr_event_payload(stmt.value, env)
                self._record_event(
                    trace_key=trace_key,
                    scope_name=scope_name,
                    before_env=before_env,
                    after_env=dict(env),
                    event="assign",
                    detail_name=target_name,
                    value=value,
                    lhs=lhs,
                    rhs=rhs,
                    op_name=op_name,
                    counterexample=counterexample,
                )
                return None, None

            if isinstance(stmt, ast.AnnAssign):
                value = self._eval_expr(stmt.value, env) if stmt.value is not None else None
                target_name = self._primary_target_name(stmt.target)
                self._assign_value(stmt.target, value, env)
                op_name, lhs, rhs = self._expr_event_payload(stmt.value, env)
                self._record_event(
                    trace_key=trace_key,
                    scope_name=scope_name,
                    before_env=before_env,
                    after_env=dict(env),
                    event="assign",
                    detail_name=target_name,
                    value=value,
                    lhs=lhs,
                    rhs=rhs,
                    op_name=op_name,
                    counterexample=counterexample,
                )
                return None, None

            if isinstance(stmt, ast.AugAssign):
                target_name = self._primary_target_name(stmt.target)
                cur_value = self._eval_expr(stmt.target, env)
                rhs_value = self._eval_expr(stmt.value, env)
                value = self._apply_binop(stmt.op, cur_value, rhs_value)
                self._assign_value(stmt.target, value, env)
                self._record_event(
                    trace_key=trace_key,
                    scope_name=scope_name,
                    before_env=before_env,
                    after_env=dict(env),
                    event="assign",
                    detail_name=target_name,
                    value=value,
                    lhs=cur_value,
                    rhs=rhs_value,
                    op_name=type(stmt.op).__name__.lower(),
                    counterexample=counterexample,
                )
                return None, None

            if isinstance(stmt, ast.Return):
                value = self._eval_expr(stmt.value, env) if stmt.value is not None else None
                op_name, lhs, rhs = self._expr_event_payload(stmt.value, env)
                self._record_event(
                    trace_key=trace_key,
                    scope_name=scope_name,
                    before_env=before_env,
                    after_env=dict(env),
                    event="return",
                    value=value,
                    lhs=lhs,
                    rhs=rhs,
                    op_name=op_name,
                    counterexample=counterexample,
                )
                return value, "return"

            if isinstance(stmt, ast.If):
                cond_value = self._eval_expr(stmt.test, env)
                truth_value = bool(cond_value) if cond_value is not UNKNOWN else False
                op_name, lhs, rhs = self._expr_event_payload(stmt.test, env)
                self._record_event(
                    trace_key=trace_key,
                    scope_name=scope_name,
                    before_env=before_env,
                    after_env=dict(env),
                    event="compare",
                    value=truth_value,
                    lhs=lhs,
                    rhs=rhs,
                    op_name=op_name,
                    counterexample=counterexample,
                )
                branch = stmt.body if truth_value else stmt.orelse
                result, control = self._execute_block(
                    branch,
                    env=env,
                    scope_name=scope_name,
                    trace_key=trace_key,
                    counterexample=counterexample,
                )
                return result, control

            if isinstance(stmt, ast.For):
                iterable = self._normalize_iterable(self._eval_expr(stmt.iter, env))
                if iterable is UNKNOWN:
                    return None, None
                loop_items = list(iterable)[: self.max_loop_iters]
                for item in loop_items:
                    iter_before = dict(env)
                    target_name = self._primary_target_name(stmt.target)
                    self._assign_value(stmt.target, item, env)
                    self._record_event(
                        trace_key=trace_key,
                        scope_name=scope_name,
                        before_env=iter_before,
                        after_env=dict(env),
                        event="assign",
                        detail_name=target_name,
                        value=item,
                        lhs=item,
                        rhs=None,
                        op_name="for_iter",
                        counterexample=counterexample,
                    )
                    result, control = self._execute_block(
                        stmt.body,
                        env=env,
                        scope_name=scope_name,
                        trace_key=trace_key,
                        counterexample=counterexample,
                    )
                    if control in ("halt", "return"):
                        return result, control
                    if control == "break":
                        break
                    if control == "continue":
                        continue
                else:
                    if stmt.orelse:
                        return self._execute_block(
                            stmt.orelse,
                            env=env,
                            scope_name=scope_name,
                            trace_key=trace_key,
                            counterexample=counterexample,
                        )
                return None, None

            if isinstance(stmt, ast.While):
                n_iters = 0
                while n_iters < self.max_loop_iters:
                    cond_before = dict(env)
                    cond_value = self._eval_expr(stmt.test, env)
                    truth_value = bool(cond_value) if cond_value is not UNKNOWN else False
                    op_name, lhs, rhs = self._expr_event_payload(stmt.test, env)
                    self._record_event(
                        trace_key=trace_key,
                        scope_name=scope_name,
                        before_env=cond_before,
                        after_env=dict(env),
                        event="compare",
                        value=truth_value,
                        lhs=lhs,
                        rhs=rhs,
                        op_name=op_name or "while",
                        counterexample=counterexample,
                    )
                    if not truth_value:
                        if stmt.orelse:
                            return self._execute_block(
                                stmt.orelse,
                                env=env,
                                scope_name=scope_name,
                                trace_key=trace_key,
                                counterexample=counterexample,
                            )
                        break
                    result, control = self._execute_block(
                        stmt.body,
                        env=env,
                        scope_name=scope_name,
                        trace_key=trace_key,
                        counterexample=counterexample,
                    )
                    n_iters += 1
                    if control in ("halt", "return"):
                        return result, control
                    if control == "break":
                        break
                    if control == "continue":
                        continue
                return None, None

            if isinstance(stmt, ast.Break):
                return None, "break"

            if isinstance(stmt, ast.Continue):
                return None, "continue"

            if isinstance(stmt, ast.Assert):
                cond_value = self._eval_expr(stmt.test, env)
                truth_value = bool(cond_value) if cond_value is not UNKNOWN else False
                if not truth_value:
                    raise _TraceRuntimeError("AssertionError")
                return None, None

            if isinstance(stmt, ast.Raise):
                exc_value = self._eval_expr(stmt.exc, env) if stmt.exc is not None else None
                error_name = (
                    exc_value
                    if isinstance(exc_value, str) and exc_value
                    else self._call_name(stmt.exc) if stmt.exc is not None else "RuntimeError"
                )
                raise _TraceRuntimeError(str(error_name or "RuntimeError"))

            if isinstance(stmt, ast.Pass):
                return None, None

            if isinstance(stmt, ast.Expr):
                value = self._eval_expr(stmt.value, env)
                op_name, lhs, rhs = self._expr_event_payload(stmt.value, env)
                self._record_event(
                    trace_key=trace_key,
                    scope_name=scope_name,
                    before_env=before_env,
                    after_env=dict(env),
                    event="call" if isinstance(stmt.value, ast.Call) else "expr",
                    detail_name=self._call_name(stmt.value) if isinstance(stmt.value, ast.Call) else None,
                    value=value,
                    lhs=lhs,
                    rhs=rhs,
                    op_name=op_name,
                    counterexample=counterexample,
                )
                return None, None
        except _TraceRuntimeError as exc:
            self._record_event(
                trace_key=trace_key,
                scope_name=scope_name,
                before_env=before_env,
                after_env=dict(env),
                event="error",
                error_name=exc.error_name,
                counterexample=counterexample,
            )
            return None, "halt"
        return None, None

    def _record_event(
        self,
        trace_key: str,
        scope_name: str,
        before_env: Dict[str, Any],
        after_env: Dict[str, Any],
        event: str,
        detail_name: Optional[str] = None,
        value: Any = None,
        lhs: Any = None,
        rhs: Any = None,
        op_name: Optional[str] = None,
        error_name: Optional[str] = None,
        counterexample: bool = False,
    ) -> None:
        step_idx = self._trace_steps.get(trace_key, 0)
        if step_idx >= self.max_steps:
            return
        self._events.append(
            _EventRecord(
                trace_key=trace_key,
                step_idx=step_idx,
                scope_name=scope_name,
                before_env=before_env,
                after_env=after_env,
                event=event,
                detail_name=detail_name,
                value=value,
                lhs=lhs,
                rhs=rhs,
                op_name=op_name,
                error_name=error_name,
                counterexample=counterexample,
            )
        )
        self._trace_steps[trace_key] = step_idx + 1

    def _eval_expr(self, node: Optional[ast.AST], env: Dict[str, Any]) -> Any:
        if node is None:
            return None
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return env.get(node.id, UNKNOWN)
        if isinstance(node, ast.Tuple):
            return tuple(self._eval_expr(elt, env) for elt in node.elts)
        if isinstance(node, ast.List):
            return [self._eval_expr(elt, env) for elt in node.elts]
        if isinstance(node, ast.Set):
            return {self._eval_expr(elt, env) for elt in node.elts}
        if isinstance(node, ast.Dict):
            return {
                self._eval_expr(key, env): self._eval_expr(value, env)
                for key, value in zip(node.keys, node.values)
            }
        if isinstance(node, ast.UnaryOp):
            value = self._eval_expr(node.operand, env)
            if value is UNKNOWN:
                return UNKNOWN
            if isinstance(node.op, ast.USub) and isinstance(value, (int, float)):
                return -value
            if isinstance(node.op, ast.UAdd) and isinstance(value, (int, float)):
                return +value
            if isinstance(node.op, ast.Not):
                return not bool(value)
            raise _TraceRuntimeError("TypeError")
        if isinstance(node, ast.BoolOp):
            values = [self._eval_expr(v, env) for v in node.values]
            if any(v is UNKNOWN for v in values):
                return UNKNOWN
            if isinstance(node.op, ast.And):
                result = True
                for value in values:
                    result = result and bool(value)
                return result
            if isinstance(node.op, ast.Or):
                result = False
                for value in values:
                    result = result or bool(value)
                return result
            raise _TraceRuntimeError("TypeError")
        if isinstance(node, ast.IfExp):
            cond_value = self._eval_expr(node.test, env)
            if cond_value is UNKNOWN:
                return UNKNOWN
            branch = node.body if bool(cond_value) else node.orelse
            return self._eval_expr(branch, env)
        if isinstance(node, ast.Compare):
            left = self._eval_expr(node.left, env)
            if left is UNKNOWN:
                return UNKNOWN
            current = left
            result = True
            for op, comp in zip(node.ops, node.comparators):
                right = self._eval_expr(comp, env)
                if right is UNKNOWN:
                    return UNKNOWN
                result = result and self._apply_compare(op, current, right)
                current = right
            return result
        if isinstance(node, ast.BinOp):
            left = self._eval_expr(node.left, env)
            right = self._eval_expr(node.right, env)
            if left is UNKNOWN or right is UNKNOWN:
                return UNKNOWN
            return self._apply_binop(node.op, left, right)
        if isinstance(node, ast.Subscript):
            container = self._eval_expr(node.value, env)
            key = self._eval_subscript_key(node.slice, env)
            if container is UNKNOWN or key is UNKNOWN:
                return UNKNOWN
            try:
                if isinstance(container, dict):
                    return container[key]
                if isinstance(container, (list, tuple, str)):
                    if not isinstance(key, int):
                        raise _TraceRuntimeError("TypeError")
                    return container[key]
            except KeyError:
                raise _TraceRuntimeError("KeyError")
            except IndexError:
                raise _TraceRuntimeError("IndexError")
            raise _TraceRuntimeError("TypeError")
        if isinstance(node, ast.Attribute):
            owner = self._eval_expr(node.value, env)
            if owner is UNKNOWN:
                return UNKNOWN
            if isinstance(owner, dict) and node.attr in owner:
                return owner[node.attr]
            if hasattr(owner, node.attr):
                return getattr(owner, node.attr)
            return UNKNOWN
        if isinstance(node, ast.Call):
            return self._eval_call(node, env)
        return UNKNOWN

    def _eval_call(self, node: ast.Call, env: Dict[str, Any]) -> Any:
        try:
            fn_name = self._call_name(node)
            args = [self._eval_expr(arg, env) for arg in node.args]
            if any(arg is UNKNOWN for arg in args):
                return UNKNOWN

            if isinstance(node.func, ast.Attribute):
                owner = self._eval_expr(node.func.value, env)
                if owner is UNKNOWN:
                    return UNKNOWN
                method_name = node.func.attr
                if isinstance(owner, list):
                    if method_name == "append" and len(args) == 1:
                        owner.append(args[0])
                        return None
                    if method_name == "extend" and len(args) == 1:
                        owner.extend(list(self._normalize_iterable(args[0])))
                        return None
                    if method_name == "pop" and len(args) <= 1:
                        index = int(args[0]) if args else -1
                        return owner.pop(index)
                if isinstance(owner, dict):
                    if method_name == "get" and len(args) in (1, 2):
                        default = args[1] if len(args) == 2 else None
                        return owner.get(args[0], default)
                    if method_name == "keys" and not args:
                        return tuple(owner.keys())
                    if method_name == "values" and not args:
                        return tuple(owner.values())
                    if method_name == "items" and not args:
                        return tuple(owner.items())
                    if method_name == "pop" and len(args) in (1, 2):
                        if len(args) == 2:
                            return owner.pop(args[0], args[1])
                        return owner.pop(args[0])
                    if method_name == "setdefault" and len(args) in (1, 2):
                        default = args[1] if len(args) == 2 else None
                        return owner.setdefault(args[0], default)
                if isinstance(owner, set):
                    if method_name == "add" and len(args) == 1:
                        owner.add(args[0])
                        return None
                    if method_name == "discard" and len(args) == 1:
                        owner.discard(args[0])
                        return None
                if isinstance(owner, str):
                    if method_name == "upper" and not args:
                        return owner.upper()
                    if method_name == "lower" and not args:
                        return owner.lower()
                    if method_name == "strip" and len(args) <= 1:
                        return owner.strip(*args)
                    if method_name == "split" and len(args) <= 2:
                        return owner.split(*args)
                    if method_name == "startswith" and len(args) == 1:
                        return owner.startswith(args[0])
                    if method_name == "endswith" and len(args) == 1:
                        return owner.endswith(args[0])
                return UNKNOWN

            fn_value = env.get(fn_name) if fn_name is not None else None

            if fn_name == "len" and len(args) == 1 and isinstance(args[0], (str, list, tuple, dict, set)):
                return len(args[0])
            if fn_name == "int" and len(args) == 1:
                return int(args[0])
            if fn_name == "float" and len(args) == 1:
                return float(args[0])
            if fn_name == "str" and len(args) == 1:
                return str(args[0])
            if fn_name == "bool" and len(args) == 1:
                return bool(args[0])
            if fn_name == "range" and 1 <= len(args) <= 3:
                return tuple(range(*[int(arg) for arg in args]))
            if fn_name == "list" and len(args) <= 1:
                return list(self._normalize_iterable(args[0])) if args else []
            if fn_name == "tuple" and len(args) <= 1:
                return tuple(self._normalize_iterable(args[0])) if args else ()
            if fn_name == "set" and len(args) <= 1:
                return set(self._normalize_iterable(args[0])) if args else set()
            if fn_name == "dict" and len(args) <= 1:
                return dict(args[0]) if args else {}
            if fn_name == "sum" and len(args) in (1, 2):
                start = args[1] if len(args) == 2 else 0
                return sum(self._normalize_iterable(args[0]), start)
            if fn_name == "abs" and len(args) == 1:
                return abs(args[0])
            if fn_name == "min" and len(args) >= 1:
                return min(*args)
            if fn_name == "max" and len(args) >= 1:
                return max(*args)
            if fn_name == "enumerate" and len(args) in (1, 2):
                start = int(args[1]) if len(args) == 2 else 0
                return tuple(enumerate(self._normalize_iterable(args[0]), start=start))
            if fn_name == "sorted" and len(args) == 1:
                return sorted(self._normalize_iterable(args[0]))

            if isinstance(fn_value, _FunctionValue):
                param_names = [arg.arg for arg in fn_value.node.args.args]
                bindings = {name: value for name, value in zip(param_names, args)}
                return self._execute_function(
                    fn_value,
                    bindings,
                    trace_key=f"{fn_value.name}:call:{self._call_counter}",
                    counterexample=False,
                )
            return UNKNOWN
        except _TraceRuntimeError:
            raise
        except Exception as exc:
            raise _TraceRuntimeError(type(exc).__name__) from exc

    @staticmethod
    def _apply_compare(op: ast.cmpop, left: Any, right: Any) -> bool:
        if isinstance(op, ast.Eq):
            return left == right
        if isinstance(op, ast.NotEq):
            return left != right
        if isinstance(op, ast.Is):
            return left is right
        if isinstance(op, ast.IsNot):
            return left is not right
        if isinstance(op, ast.In):
            return left in right
        if isinstance(op, ast.NotIn):
            return left not in right
        if isinstance(left, str) or isinstance(right, str):
            raise _TraceRuntimeError("TypeError")
        if isinstance(op, ast.Lt):
            return left < right
        if isinstance(op, ast.LtE):
            return left <= right
        if isinstance(op, ast.Gt):
            return left > right
        if isinstance(op, ast.GtE):
            return left >= right
        raise _TraceRuntimeError("TypeError")

    @staticmethod
    def _apply_binop(op: ast.operator, left: Any, right: Any) -> Any:
        numeric = (int, float)
        if isinstance(op, ast.Add):
            if isinstance(left, numeric) and isinstance(right, numeric):
                return left + right
            if isinstance(left, str) and isinstance(right, str):
                return left + right
            if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
                return left + right
            raise _TraceRuntimeError("TypeError")
        if isinstance(op, ast.Sub):
            if isinstance(left, numeric) and isinstance(right, numeric):
                return left - right
            raise _TraceRuntimeError("TypeError")
        if isinstance(op, ast.Mult):
            if isinstance(left, numeric) and isinstance(right, numeric):
                return left * right
            if isinstance(left, str) and isinstance(right, int):
                return left * right
            if isinstance(right, str) and isinstance(left, int):
                return right * left
            raise _TraceRuntimeError("TypeError")
        if isinstance(op, ast.Div):
            if isinstance(left, numeric) and isinstance(right, numeric):
                if float(right) == 0.0:
                    raise _TraceRuntimeError("ZeroDivisionError")
                return left / right
            raise _TraceRuntimeError("TypeError")
        if isinstance(op, ast.FloorDiv):
            if isinstance(left, numeric) and isinstance(right, numeric):
                if float(right) == 0.0:
                    raise _TraceRuntimeError("ZeroDivisionError")
                return left // right
            raise _TraceRuntimeError("TypeError")
        if isinstance(op, ast.Mod):
            if isinstance(left, numeric) and isinstance(right, numeric):
                if float(right) == 0.0:
                    raise _TraceRuntimeError("ZeroDivisionError")
                return left % right
            raise _TraceRuntimeError("TypeError")
        if isinstance(op, ast.Pow):
            if isinstance(left, numeric) and isinstance(right, numeric):
                return left ** right
            raise _TraceRuntimeError("TypeError")
        raise _TraceRuntimeError("TypeError")

    @staticmethod
    def _assign_target_name(target: ast.AST) -> Optional[str]:
        if isinstance(target, ast.Name):
            return target.id
        return None

    def _primary_target_name(self, target: ast.AST) -> Optional[str]:
        if isinstance(target, ast.Name):
            return target.id
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            return target.value.id
        if isinstance(target, (ast.Tuple, ast.List)) and target.elts:
            return self._primary_target_name(target.elts[0])
        return self._assign_target_name(target)

    def _assign_value(self, target: ast.AST, value: Any, env: Dict[str, Any]) -> None:
        if isinstance(target, ast.Name):
            env[target.id] = value
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            values = list(self._normalize_iterable(value))
            for sub_target, sub_value in zip(target.elts, values):
                self._assign_value(sub_target, sub_value, env)
            return
        if isinstance(target, ast.Subscript):
            container = self._eval_expr(target.value, env)
            key = self._eval_subscript_key(target.slice, env)
            if container is UNKNOWN or key is UNKNOWN:
                raise _TraceRuntimeError("TypeError")
            try:
                container[key] = value
                return
            except Exception as exc:
                raise _TraceRuntimeError(type(exc).__name__) from exc
        raise _TraceRuntimeError("TypeError")

    def _eval_subscript_key(self, node: ast.AST, env: Dict[str, Any]) -> Any:
        if isinstance(node, ast.Slice):
            lower = self._eval_expr(node.lower, env) if node.lower is not None else None
            upper = self._eval_expr(node.upper, env) if node.upper is not None else None
            step = self._eval_expr(node.step, env) if node.step is not None else None
            return slice(lower, upper, step)
        return self._eval_expr(node, env)

    @staticmethod
    def _normalize_iterable(value: Any) -> Any:
        if value is UNKNOWN:
            return UNKNOWN
        if value is None:
            return tuple()
        if isinstance(value, dict):
            return tuple(value.keys())
        if isinstance(value, (list, tuple, str, set)):
            return value
        raise _TraceRuntimeError("TypeError")

    @staticmethod
    def _call_name(node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Call):
            return _PythonTraceBuilder._call_name(node.func)
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    def _expr_event_payload(
        self,
        node: Optional[ast.AST],
        env: Dict[str, Any],
    ) -> Tuple[Optional[str], Any, Any]:
        if isinstance(node, ast.BinOp):
            try:
                lhs = self._eval_expr(node.left, env)
                rhs = self._eval_expr(node.right, env)
            except _TraceRuntimeError:
                lhs, rhs = None, None
            return type(node.op).__name__.lower(), lhs, rhs
        if isinstance(node, ast.Compare):
            try:
                lhs = self._eval_expr(node.left, env)
                rhs = self._eval_expr(node.comparators[0], env) if node.comparators else None
            except _TraceRuntimeError:
                lhs, rhs = None, None
            op_name = type(node.ops[0]).__name__.lower() if node.ops else "compare"
            return op_name, lhs, rhs
        if isinstance(node, ast.Subscript):
            try:
                lhs = self._eval_expr(node.value, env)
                rhs = self._eval_subscript_key(node.slice, env)
            except _TraceRuntimeError:
                lhs, rhs = None, None
            return "subscript", lhs, rhs
        if isinstance(node, ast.Call):
            return self._call_name(node), None, None
        return None, None, None

    def _materialize_fact_bundle(
        self,
    ) -> Tuple[Tuple[TraceTransitionFacts, ...], Tuple[TraceTransitionFacts, ...], FrozenSet[Any], FrozenSet[Any]]:
        transitions: List[TraceTransitionFacts] = []
        counterexamples: List[TraceTransitionFacts] = []
        observed_facts = set()
        target_facts = set()

        for idx, event in enumerate(self._events):
            before_facts = frozenset(self._event_snapshot_facts(event, before=True))
            after_facts = frozenset(self._event_snapshot_facts(event, before=False))
            if idx > 0 and self._events[idx - 1].trace_key == event.trace_key:
                prev_step = _step_symbol(self._events[idx - 1].trace_key, self._events[idx - 1].step_idx)
                cur_step = _step_symbol(event.trace_key, event.step_idx)
                observed_facts.add(self._atom(TRACE_TRANSITION_PRED, prev_step, cur_step))
            observed_facts.update(before_facts)
            observed_facts.update(after_facts)
            transition = TraceTransitionFacts(
                before_facts=before_facts,
                after_facts=after_facts,
                label=event.event,
                counterexample=event.counterexample,
            )
            if event.counterexample:
                counterexamples.append(transition)
            else:
                transitions.append(transition)
                if event.event in ("assign", "return", "error"):
                    target_facts.update(self._targetable_after_facts(after_facts))
        return (
            tuple(transitions),
            tuple(counterexamples),
            frozenset(observed_facts),
            frozenset(target_facts),
        )

    @staticmethod
    def _targetable_after_facts(after_facts: FrozenSet[Any]) -> List[Any]:
        semantic_preds = {
            TRACE_ASSIGN_EVENT_PRED,
            TRACE_RETURN_EVENT_PRED,
            TRACE_ERROR_EVENT_PRED,
            TRACE_BINOP_EVENT_PRED,
            TRACE_COMPARE_EVENT_PRED,
            TRACE_STATE_VALUE_PRED,
            TRACE_PARAM_BIND_PRED,
        }
        return [fact for fact in after_facts if getattr(fact, "pred", None) in semantic_preds]

    def _event_snapshot_facts(
        self,
        event: _EventRecord,
        before: bool,
    ) -> List[Any]:
        env = event.before_env if before else event.after_env
        step = _step_symbol(event.trace_key, event.step_idx)
        scope = _scope_symbol(event.scope_name)
        facts = [self._atom(TRACE_SCOPE_PRED, step, scope)]
        facts.append(
            self._atom(
                TRACE_COUNTEREXAMPLE_PRED if event.counterexample else TRACE_PRIMARY_PRED,
                step,
                scope,
            )
        )
        for name, value in sorted(env.items()):
            if isinstance(value, _FunctionValue):
                continue
            facts.append(self._atom(TRACE_STATE_VALUE_PRED, step, _var_symbol(name), _value_symbol(value)))
            facts.append(self._atom(TRACE_STATE_TYPE_PRED, step, _var_symbol(name), _type_symbol(_type_name(value))))
        if not before:
            if event.event == "assign" and event.detail_name is not None:
                facts.append(
                    self._atom(
                        TRACE_ASSIGN_EVENT_PRED,
                        step,
                        scope,
                        _var_symbol(event.detail_name),
                        _value_symbol(event.value),
                    )
                )
            if event.event == "return":
                facts.append(self._atom(TRACE_RETURN_EVENT_PRED, step, scope, _value_symbol(event.value)))
            if event.event == "error" and event.error_name is not None:
                facts.append(self._atom(TRACE_ERROR_EVENT_PRED, step, scope, _error_symbol(event.error_name)))
            if event.event == "call" and event.detail_name is not None:
                facts.append(self._atom(TRACE_CALL_EVENT_PRED, step, scope, _scope_symbol(event.detail_name)))
            if event.event == "param" and event.detail_name is not None:
                facts.append(
                    self._atom(
                        TRACE_PARAM_BIND_PRED,
                        step,
                        scope,
                        _var_symbol(event.detail_name),
                        _value_symbol(event.value),
                    )
                )
            if event.event in ("assign", "return", "expr") and event.op_name and event.lhs is not None:
                facts.append(
                    self._atom(
                        TRACE_BINOP_EVENT_PRED,
                        step,
                        _op_symbol(event.op_name),
                        _value_symbol(event.lhs),
                        _value_symbol(event.rhs),
                        _value_symbol(event.value),
                    )
                )
            if event.event == "compare" and event.op_name and event.lhs is not None:
                facts.append(
                    self._atom(
                        TRACE_COMPARE_EVENT_PRED,
                        step,
                        _op_symbol(event.op_name),
                        _value_symbol(event.lhs),
                        _value_symbol(event.rhs),
                        _value_symbol(bool(event.value)),
                    )
                )
        return facts

    @staticmethod
    def _atom(pred: int, *args: int) -> Any:
        from omen_prolog import Const, HornAtom

        return HornAtom(pred=pred, args=tuple(Const(int(arg)) for arg in args))


class _ObservationTraceBuilder:
    def __init__(
        self,
        *,
        language: str,
        max_steps: int = 24,
        max_counterexamples: int = 4,
        semantic_backbone: Optional[SemanticGroundingBackbone] = None,
    ):
        self.language = language or "text"
        self.max_steps = max(1, int(max_steps))
        self.max_counterexamples = max(0, int(max_counterexamples))
        self.semantic_backbone = semantic_backbone

    def build(self, text: str) -> Optional[SymbolicExecutionTraceBundle]:
        bundle, _artifacts = self.build_with_artifacts(text)
        return bundle

    def build_with_artifacts(
        self,
        text: str,
        *,
        memory_records: Optional[Sequence[object]] = None,
    ) -> Tuple[Optional[SymbolicExecutionTraceBundle], Optional[GroundingRuntimeArtifacts]]:
        orchestrated = run_grounding_orchestrator(
            text,
            language=self.language,
            max_segments=self.max_steps,
            backbone=self.semantic_backbone,
            memory_records=memory_records,
        )
        pipeline = orchestrated.pipeline
        artifacts = _build_grounding_runtime_artifacts(
            orchestrated,
            language=self.language,
            source_text=text,
            max_steps=self.max_steps,
        )
        segments = list(pipeline.compiled.segments)
        if not segments:
            return None, artifacts if _has_grounding_runtime_artifacts(artifacts) else None
        if (
            len(segments) < 2
            and pipeline.compiled.metadata.get("compiled_relation_claims", 0.0) <= 0.0
            and pipeline.compiled.metadata.get("compiled_state_claims", 0.0) <= 0.0
            and pipeline.compiled.metadata.get("compiled_goal_claims", 0.0) <= 0.0
            and pipeline.compiled.metadata.get("grounding_counterexample_segments", 0.0) <= 0.0
        ):
            return None, artifacts if _has_grounding_runtime_artifacts(artifacts) else None

        transitions: List[TraceTransitionFacts] = []
        counterexamples: List[TraceTransitionFacts] = []
        observed_facts: Set[Any] = set()
        target_facts: Set[Any] = set()
        previous_after = frozenset()

        for step_idx, segment in enumerate(segments[: self.max_steps]):
            after_facts, targetable_facts = self._segment_facts(segment, step_idx)
            if not after_facts:
                continue
            transition = TraceTransitionFacts(
                before_facts=previous_after,
                after_facts=after_facts,
                label=f"{self.language}:{step_idx}",
                counterexample=bool(segment.counterexample),
            )
            observed_facts.update(previous_after)
            observed_facts.update(after_facts)
            observed_facts.add(self._atom(TRACE_PRIMARY_PRED, _step_symbol(self.language, step_idx)))
            if step_idx > 0:
                observed_facts.add(
                    self._atom(
                        TRACE_TRANSITION_PRED,
                        _step_symbol(self.language, step_idx - 1),
                        _step_symbol(self.language, step_idx),
                    )
                )
            transitions.append(transition)
            target_facts.update(targetable_facts)
            if transition.counterexample and len(counterexamples) < self.max_counterexamples:
                counterexamples.append(
                    TraceTransitionFacts(
                        before_facts=transition.before_facts,
                        after_facts=transition.after_facts,
                        label=transition.label,
                        counterexample=True,
                    )
                )
            previous_after = after_facts

        if not transitions:
            return None, artifacts if _has_grounding_runtime_artifacts(artifacts) else None

        if not target_facts:
            target_facts.update(transitions[-1].after_facts)
        if counterexamples:
            target_facts.update(counterexamples[-1].after_facts)
        bundle = SymbolicExecutionTraceBundle(
            language=self.language,
            source_text=text,
            observed_facts=frozenset(observed_facts),
            target_facts=frozenset(target_facts),
            transitions=tuple(transitions),
            counterexamples=tuple(counterexamples),
            grounding_facts=artifacts.grounding_facts,
            grounding_target_facts=artifacts.grounding_target_facts,
            grounding_hypotheses=artifacts.grounding_hypotheses,
            grounding_verification_records=artifacts.grounding_verification_records,
            grounding_validation_records=artifacts.grounding_validation_records,
            grounding_repair_actions=artifacts.grounding_repair_actions,
            grounding_world_state_records=artifacts.grounding_world_state_records,
            grounding_ontology_records=artifacts.grounding_ontology_records,
            grounding_ontology_facts=artifacts.grounding_ontology_facts,
            grounding_world_state_active_facts=artifacts.grounding_world_state_active_facts,
            grounding_world_state_hypothetical_facts=artifacts.grounding_world_state_hypothetical_facts,
            grounding_world_state_contradicted_facts=artifacts.grounding_world_state_contradicted_facts,
            grounding_graph_records=artifacts.grounding_graph_records,
            metadata=dict(artifacts.metadata),
        )
        return bundle, artifacts

    def _segments(self, text: str) -> List[str]:
        return split_text_segments(text, max_segments=self.max_steps)

    @staticmethod
    def _normalize_segment(segment: str) -> str:
        normalized = split_text_segments(segment, max_segments=1)
        return normalized[0] if normalized else segment.strip()

    def _segment_facts(self, segment: CompiledSymbolicSegment, step_idx: int) -> Tuple[FrozenSet[Any], FrozenSet[Any]]:
        facts: List[Any] = []
        targetable: List[Any] = []
        step_symbol = _step_symbol(self.language, step_idx)
        scope_symbol = _scope_symbol(f"{self.language}:observation")
        facts.append(self._atom(TRACE_SCOPE_PRED, step_symbol, scope_symbol))
        facts.append(self._atom(TRACE_PRIMARY_PRED, step_symbol))

        tokens = list(segment.tokens)
        for token in tokens[:8]:
            lex_symbol = _lexeme_symbol(token)
            facts.append(self._atom(TRACE_TEXT_TOKEN_PRED, step_symbol, lex_symbol))
            facts.append(self._atom(TRACE_STATE_VALUE_PRED, step_symbol, lex_symbol, _value_symbol(token)))
            facts.append(self._atom(TRACE_STATE_TYPE_PRED, step_symbol, lex_symbol, _type_symbol("token")))

        for key, value in segment.states:
            key_symbol = _lexeme_symbol(key)
            value_symbol = _value_symbol(value)
            state_fact = self._atom(
                TRACE_TEXT_STATE_PRED,
                step_symbol,
                key_symbol,
                value_symbol,
            )
            facts.append(state_fact)
            targetable.append(state_fact)
            assign_fact = self._atom(
                TRACE_ASSIGN_EVENT_PRED,
                step_symbol,
                key_symbol,
                value_symbol,
            )
            facts.append(assign_fact)
            targetable.append(assign_fact)

        for left, rel, right in segment.relations:
            relation_fact = self._atom(
                TRACE_TEXT_RELATION_PRED,
                step_symbol,
                _lexeme_symbol(left),
                _op_symbol(rel),
                _lexeme_symbol(right),
            )
            facts.append(relation_fact)
            targetable.append(relation_fact)
            assign_fact = self._atom(
                TRACE_ASSIGN_EVENT_PRED,
                step_symbol,
                _lexeme_symbol(left),
                _value_symbol(right),
            )
            facts.append(assign_fact)
            targetable.append(assign_fact)

        for goal_name, goal_value in segment.goals:
            goal_fact = self._atom(
                TRACE_TEXT_GOAL_PRED,
                step_symbol,
                _lexeme_symbol(goal_name),
                _value_symbol(goal_value),
            )
            facts.append(goal_fact)
            targetable.append(goal_fact)

        if segment.counterexample:
            negation_fact = self._atom(
                TRACE_TEXT_NEGATION_PRED,
                step_symbol,
                _value_symbol(segment.normalized_text.casefold()[:64]),
            )
            facts.append(negation_fact)
            targetable.append(negation_fact)

        return frozenset(facts), frozenset(targetable)

    @classmethod
    def _tokens(cls, segment: str) -> List[str]:
        return tokenize_semantic_words(segment)

    @classmethod
    def _normalize_symbol_text(cls, value: Any) -> Optional[str]:
        return normalize_symbol_text(value)

    @classmethod
    def _flatten_payload(cls, payload: Any, prefix: str = "") -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        if isinstance(payload, dict):
            for key, value in list(payload.items())[:8]:
                normalized_key = cls._normalize_symbol_text(key)
                if not normalized_key:
                    continue
                child_prefix = normalized_key if not prefix else f"{prefix}_{normalized_key}"
                pairs.extend(cls._flatten_payload(value, child_prefix))
            return pairs
        if isinstance(payload, (list, tuple)):
            for idx, value in enumerate(list(payload)[:6]):
                child_prefix = prefix or f"item_{idx}"
                pairs.extend(cls._flatten_payload(value, child_prefix))
            return pairs
        normalized_key = cls._normalize_symbol_text(prefix or "value")
        normalized_value = cls._normalize_symbol_text(payload)
        if normalized_key and normalized_value:
            pairs.append((normalized_key, normalized_value))
        return pairs

    @classmethod
    def _structured_pairs(cls, segment: str) -> List[Tuple[str, str]]:
        return extract_structured_pairs(segment)

    @classmethod
    def _relations(cls, segment: str) -> List[Tuple[str, str, str]]:
        return [
            (hint.left, hint.relation, hint.right)
            for hint in extract_relation_hints(segment)
        ]

    @classmethod
    def _goal_pairs(cls, segment: str) -> List[Tuple[str, str]]:
        return [
            (hint.goal_name, hint.goal_value)
            for hint in extract_goal_hints(segment)
        ]

    @classmethod
    def _is_counterexample(cls, segment: str) -> bool:
        return is_counterexample_text(segment)

    @staticmethod
    def _atom(pred: int, *args: int) -> Any:
        from omen_prolog import Const, HornAtom

        return HornAtom(pred=pred, args=tuple(Const(int(arg)) for arg in args))


def build_symbolic_trace_bundle(
    code: str,
    lang_hint: str = "python",
    max_steps: int = 24,
    max_counterexamples: int = 4,
    semantic_backbone: Optional[SemanticGroundingBackbone] = None,
    memory_records: Optional[Sequence[object]] = None,
) -> Optional[SymbolicExecutionTraceBundle]:
    bundle, _artifacts = build_symbolic_trace_bundle_with_artifacts(
        code,
        lang_hint=lang_hint,
        max_steps=max_steps,
        max_counterexamples=max_counterexamples,
        semantic_backbone=semantic_backbone,
        memory_records=memory_records,
    )
    return bundle


def build_symbolic_trace_bundle_with_artifacts(
    code: str,
    lang_hint: str = "python",
    max_steps: int = 24,
    max_counterexamples: int = 4,
    semantic_backbone: Optional[SemanticGroundingBackbone] = None,
    memory_records: Optional[Sequence[object]] = None,
) -> Tuple[Optional[SymbolicExecutionTraceBundle], Optional[GroundingRuntimeArtifacts]]:
    normalized_hint = (lang_hint or "python").lower()
    if not code.strip():
        return None, None
    if normalized_hint == "python":
        builder = _PythonTraceBuilder(
            max_steps=max_steps,
            max_counterexamples=max_counterexamples,
        )
        bundle = builder.build(code)
        if bundle is not None:
            return bundle, None
    observation_builder = _ObservationTraceBuilder(
        language=normalized_hint,
        max_steps=max_steps,
        max_counterexamples=max_counterexamples,
        semantic_backbone=semantic_backbone,
    )
    return observation_builder.build_with_artifacts(code, memory_records=memory_records)
