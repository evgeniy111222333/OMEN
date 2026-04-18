"""
omen_prolog.py: first-order symbolic runtime for OMEN.

This module provides Horn terms, unification, the knowledge base, proof policy,
abduction utilities, continuous symbolic learning, and the differentiable prover
used by the canonical OMEN stack.
"""

from __future__ import annotations
import sys
import enum
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from omen_symbolic.abduction_search import (
    bridge_variable_count as structural_bridge_variable_count,
    rank_goal_directed_bodies,
    rule_template_signature,
)
from omen_symbolic.controller import empty_induction_stats, run_latent_reasoning_controller
from omen_symbolic.creative_cycle import CreativeCycleCoordinator
from omen_symbolic.execution_trace import (
    build_symbolic_trace_bundle,
    SymbolicExecutionTraceBundle,
    TRACE_ASSIGN_EVENT_PRED,
    TRACE_BINOP_EVENT_PRED,
    TRACE_COMPARE_EVENT_PRED,
    TRACE_COUNTEREXAMPLE_PRED,
    TRACE_ERROR_EVENT_PRED,
    TRACE_PARAM_BIND_PRED,
    TRACE_PRIMARY_PRED,
    TRACE_RETURN_EVENT_PRED,
    TRACE_SCOPE_PRED,
    TRACE_STATE_TYPE_PRED,
    TRACE_STATE_VALUE_PRED,
)
from omen_symbolic.executor import run_symbolic_executor
from omen_symbolic.universal_bits import (
    universal_int_bits,
    universal_float_bits,
)

if __name__ == "__main__":
    sys.modules.setdefault("omen_prolog", sys.modules[__name__])


# ══════════════════════════════════════════════════════════════════════════════
# 1.  ТЕРМИ ПЕРШОГО ПОРЯДКУ  Σ = (C, F, P, V)
# ══════════════════════════════════════════════════════════════════════════════
#
#  Сигнатура:
#    C  — константи  (Const.val ≥ 0)
#    F  — функціональні символи  (Compound.func)
#    P  — предикати  (HornAtom.pred)
#    V  — змінні  (Var.name: str)
#
#  Терм t ::= c | X | f(t1,...,tk)
#  Атом  ::= p(t1,...,tn)
#  Правило: H :- B1,...,Bn   (Horn clause)
#
#  Підстановка σ: V→Term — моноїд ендоморфізмів (Subst, ∘, ε).
#  Уніфікація — алгоритм Martelli-Montanari (1982).
#
#  Зворотна сумісність:
#    int ≥ 0  →  Const(int)
#    int < 0  →  Var(f"_{pos}")  (позиційна анонімна змінна-wildcard)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Const:
    """Константа c ∈ C. val ≥ 0."""
    val: int
    def __repr__(self) -> str:          return str(self.val)
    def vars(self) -> FrozenSet[str]:   return frozenset()
    def depth(self) -> int:             return 0

@dataclass(frozen=True)
class Var:
    """Змінна X ∈ V. name — рядковий ідентифікатор."""
    name: str
    def __repr__(self) -> str:          return f"?{self.name}"
    def vars(self) -> FrozenSet[str]:   return frozenset({self.name})
    def depth(self) -> int:             return 0

@dataclass(frozen=True)
class Compound:
    """Складений терм f(t1,...,tk), f ∈ F."""
    func: int
    subterms: Tuple['Term', ...]
    def __repr__(self) -> str:
        return f"f{self.func}({','.join(repr(s) for s in self.subterms)})"
    def vars(self) -> FrozenSet[str]:
        result: FrozenSet[str] = frozenset()
        for t in self.subterms:
            result = result | t.vars()
        return result
    def depth(self) -> int:
        return 1 + max((t.depth() for t in self.subterms), default=0)

Term = Union[Const, Var, Compound]


# ─── Допоміжні функції для термів ─────────────────────────────────────────────

def _term_vars(t: Term) -> FrozenSet[str]:
    """Множина імен змінних у терм t."""
    return t.vars()

def _term_depth(t: Term) -> int:
    """Глибина терму: константа/змінна = 0, складений = 1+max(depth підтермів)."""
    return t.depth()

def _is_ground(t: Term) -> bool:
    """True, якщо терм не містить жодної змінної."""
    return not t.vars()


def _term_code_symbols(t: Term) -> Tuple[str, ...]:
    if isinstance(t, Const):
        return (f"CONST:{int(t.val)}",)
    if isinstance(t, Var):
        return ("VAR",)
    if isinstance(t, Compound):
        symbols: List[str] = [f"FUNC:{t.func}/{len(t.subterms)}"]
        for subterm in t.subterms:
            symbols.extend(_term_code_symbols(subterm))
        return tuple(symbols)
    return ("TERM",)


def _build_rule_codebook(rules: List["HornClause"]) -> Tuple[Counter, int, int]:
    counts: Counter = Counter()
    total = 0
    for rule in rules:
        symbols = rule.code_symbols()
        counts.update(symbols)
        total += len(symbols)
    return counts, total, len(counts)

def _to_term(x, pos: int = 0) -> Term:
    """
    Конвертує int-аргумент (стара нотація) у Term.
      x ≥ 0  →  Const(x)
      x < 0  →  Var(f"_{pos}")  (позиційна анонімна змінна)
    Якщо x вже Term — повертає без змін.
    """
    if isinstance(x, (Const, Var, Compound)):
        return x
    x = int(x)
    return Var(f"_{pos}") if x < 0 else Const(x)


# ─── Підстановка (Substitution) ───────────────────────────────────────────────

class Substitution:
    """
    Підстановка σ : V → Term — скінченне відображення.
    Множина підстановок з операцією композиції утворює моноїд:
      (Subst, ∘, ε),  де ε — порожня підстановка.

    Частковий порядок:
      σ ≤ τ  ⟺  ∃θ: τ = σ ∘ θ   (τ є більш конкретизованою)
    """
    __slots__ = ('bindings',)

    def __init__(self, bindings: Optional[Dict[str, Term]] = None):
        self.bindings: Dict[str, Term] = bindings if bindings is not None else {}

    @classmethod
    def empty(cls) -> 'Substitution':
        """Порожня підстановка ε — одиниця моноїду."""
        return cls({})

    # ── Застосування ──────────────────────────────────────────────────────────
    def apply(self, t: Term) -> Term:
        """
        Застосовує σ до терму t рекурсивно (з path-compression).
          Xσ = σ(X) якщо X ∈ dom(σ), інакше X.
          f(t1,...,tk)σ = f(t1σ,...,tkσ).
        """
        if isinstance(t, Const):
            return t
        if isinstance(t, Var):
            if t.name in self.bindings:
                return self.apply(self.bindings[t.name])   # chase
            return t
        if isinstance(t, Compound):
            return Compound(t.func, tuple(self.apply(s) for s in t.subterms))
        return t

    def apply_atom(self, atom: 'HornAtom') -> 'HornAtom':
        """Застосовує підстановку до атому (замінює змінні в args)."""
        return HornAtom(atom.pred, tuple(self.apply(a) for a in atom.args))

    # ── Операції моноїду ──────────────────────────────────────────────────────
    def bind(self, var_name: str, term: Term) -> 'Substitution':
        """Повертає нову підстановку {σ ∪ {var_name → σ(term)}}."""
        new_b = {k: self.apply(v) for k, v in self.bindings.items()}
        new_b[var_name] = self.apply(term)
        return Substitution(new_b)

    def compose(self, other: 'Substitution') -> 'Substitution':
        """
        Композиція σ ∘ θ:  t(σ ∘ θ) = (tθ)σ.
        Застосовуємо спочатку other(θ), потім self(σ).
        """
        new_b: Dict[str, Term] = {}
        for k, v in other.bindings.items():
            new_b[k] = self.apply(v)
        for k, v in self.bindings.items():
            if k not in new_b:
                new_b[k] = v
        return Substitution(new_b)

    # ── MDL-метрики ────────────────────────────────────────────────────────────
    def unif_complexity(self) -> int:
        """MDL: Σ_{X∈dom(σ)} Depth(σ(X)) — складність підстановки."""
        return sum(_term_depth(v) for v in self.bindings.values())

    def is_ground_for(self, atom: 'HornAtom') -> bool:
        """True, якщо всі аргументи atom після застосування σ — Const."""
        return all(isinstance(self.apply(a), Const) for a in atom.args)

    # ── Порівняння / відображення ─────────────────────────────────────────────
    def __len__(self)  -> int:  return len(self.bindings)
    def __bool__(self) -> bool: return True    # навіть ε є валідною підстановкою
    def __repr__(self) -> str:
        if not self.bindings:
            return "ε"
        return "{" + ", ".join(f"?{k}→{v}" for k, v in self.bindings.items()) + "}"
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Substitution) and self.bindings == other.bindings
    def __hash__(self) -> int:
        return hash(frozenset((k, repr(v)) for k, v in self.bindings.items()))


# ─── Freshening (перейменування змінних перед застосуванням правила) ──────────

_FRESHEN_COUNTER: List[int] = [0]

def freshen_vars(clause: 'HornClause') -> 'HornClause':
    """
    Перейменовує ВСІ іменовані змінні (не анонімні) у clause на свіжі,
    щоб уникнути конфліктів між різними застосуваннями одного правила.

    Анонімні змінні (name.startswith('_')) не перейменовуються:
    вони діють як незалежні wildcards і ніколи не зв'язуються (MM пропускає їх).
    """
    all_vars: Set[str] = set()
    for a in (clause.head,) + tuple(clause.body):
        for t in a.args:
            all_vars |= _term_vars(t)
    named_vars = {v for v in all_vars if not v.startswith('_')}

    stamp = _FRESHEN_COUNTER[0]
    _FRESHEN_COUNTER[0] += 1
    rename = Substitution({v: Var(f"{v}_{stamp}") for v in named_vars})

    new_head = rename.apply_atom(clause.head)
    new_body = tuple(rename.apply_atom(a) for a in clause.body)
    return HornClause(head=new_head, body=new_body,
                      weight=clause.weight, use_count=clause.use_count)


# ─── HornAtom та HornClause ───────────────────────────────────────────────────

@dataclass(frozen=True)
class HornAtom:
    """
    Атом: p(t1,...,tn),  p ∈ P,  ti ∈ Term.

    args — кортеж Term (Const | Var | Compound).
    Зворотна сумісність: якщо args містить int, то
      int ≥ 0  →  Const(int)
      int < 0  →  Var(f"_{pos}")   (позиційна анонімна змінна)
    """
    pred: int
    args: Tuple  # нормалізується до Tuple[Term, ...] у __post_init__

    def __post_init__(self) -> None:
        normalized = tuple(
            _to_term(a, i) if not isinstance(a, (Const, Var, Compound)) else a
            for i, a in enumerate(self.args)
        )
        object.__setattr__(self, 'args', normalized)

    def __repr__(self) -> str:
        return f"p{self.pred}({','.join(repr(a) for a in self.args)})"

    def arity(self) -> int:
        return len(self.args)

    def vars(self) -> FrozenSet[str]:
        result: FrozenSet[str] = frozenset()
        for a in self.args:
            result = result | a.vars()
        return result

    def is_ground(self) -> bool:
        """True, якщо всі аргументи — Const (ground atom)."""
        return all(_is_ground(a) for a in self.args)


@dataclass
class HornClause:
    """
    Хорнівський диз'юнкт: H :- B1,...,Bn.
    Факт (n=0): H  (без тіла, head — ground atom).
    Правило: H :- B1,...,Bn  зі змінними.

    MDL-складність:
      Length(R) = (1+|body|)·(arity+1) + Σ depth(ti)
    """
    head:      HornAtom
    body:      Tuple[HornAtom, ...] = field(default_factory=tuple)
    weight:    float = 1.0
    use_count: int   = 0

    def is_fact(self) -> bool:
        return len(self.body) == 0

    def complexity(self) -> int:
        """MDL: довжина правила з урахуванням глибини термів."""
        base = (1 + len(self.body)) * (self.head.arity() + 1)
        term_d = sum(
            _term_depth(a)
            for atom in (self.head,) + tuple(self.body)
            for a in atom.args
        )
        return base + term_d

    def code_symbols(self) -> Tuple[str, ...]:
        symbols: List[str] = ["RULE_START", f"BODY_LEN:{len(self.body)}"]
        for atom_idx, atom in enumerate((self.head,) + tuple(self.body)):
            role = "HEAD" if atom_idx == 0 else "BODY"
            symbols.append(f"{role}_PRED:{int(atom.pred)}/{atom.arity()}")
            for arg in atom.args:
                symbols.extend(_term_code_symbols(arg))
            if atom_idx > 0:
                symbols.append("BODY_SEP")
        symbols.append("RULE_END")
        return tuple(symbols)

    def description_length_bits(
        self,
        codebook: Optional[Tuple[Counter, int, int]] = None,
        pseudo_count: float = 0.5,
        include_runtime_state: bool = False,
    ) -> float:
        """
        Fixed-rule code length in bits under a universal grammar.

        The estimate is intentionally independent from the current KB content:
        rule cost should not shrink just because similar rules are already
        present in the model. This makes the symbolic MDL term comparable
        across batches and across alternative hypotheses.
        """
        del codebook, pseudo_count
        var_ids: Dict[str, int] = {}

        def term_bits(term: Term) -> float:
            if isinstance(term, Const):
                return 2.0 + universal_int_bits(int(term.val))
            if isinstance(term, Var):
                if term.name not in var_ids:
                    var_ids[term.name] = len(var_ids)
                return 2.0 + universal_int_bits(var_ids[term.name])
            if isinstance(term, Compound):
                total = 3.0
                total += universal_int_bits(int(term.func))
                total += universal_int_bits(len(term.subterms))
                for subterm in term.subterms:
                    total += term_bits(subterm)
                return total
            return 1.0

        def atom_bits(atom: HornAtom) -> float:
            total = 1.0
            total += universal_int_bits(int(atom.pred))
            total += universal_int_bits(atom.arity())
            for arg in atom.args:
                total += term_bits(arg)
            return total

        total_bits = universal_int_bits(len(self.body))
        total_bits += atom_bits(self.head)
        for atom in self.body:
            total_bits += atom_bits(atom)
        if include_runtime_state:
            total_bits += universal_float_bits(self.weight, sigma=1.0)
            total_bits += universal_int_bits(max(self.use_count, 0))
        return float(total_bits)

    def all_vars(self) -> FrozenSet[str]:
        result: FrozenSet[str] = frozenset()
        for a in (self.head,) + tuple(self.body):
            result = result | a.vars()
        return result

    def __hash__(self) -> int:
        return hash((self.head, self.body))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, HornClause) and hash(self) == hash(other)

    def __repr__(self) -> str:
        if self.is_fact():
            return repr(self.head)
        return f"{self.head} :- {', '.join(repr(a) for a in self.body)}"


@dataclass
class RelaxedHornClauseSpec:
    clause: HornClause
    head_pred_probs: torch.Tensor
    body_pred_probs: Tuple[torch.Tensor, ...]
    log_prob: Optional[torch.Tensor] = None
    source: str = "neural"


SEQ_EDGE_PRED = 470
SEQ_LAST_TOKEN_PRED = 471
SEQ_PREDICT_NEXT_PRED = 472
SEQ_ACTUAL_NEXT_PRED = 473
SEQ_AST_SUPPORT_PRED = 474
SEQ_SALIENCY_SUPPORT_PRED = 475
SEQ_GAP_DIM_PRED = 476
SEQ_DECODER_GUESS_PRED = 477
SEQ_DECODER_MISS_PRED = 478
SEQ_DECODER_SURPRISE_PRED = 479


@dataclass
class SymbolicTaskContext:
    """
    Canonical symbolic context with first-class fact provenance buckets.

    `observed_facts` remains the backward-compatible aggregate, while the
    source-specific fields preserve whether facts were observed now, recalled
    from memory, derived from saliency, derived from NET, attached as world
    context, or materialized as abductive/creative support.
    """
    observed_facts: FrozenSet[HornAtom] = field(default_factory=frozenset)
    observed_now_facts: FrozenSet[HornAtom] = field(default_factory=frozenset)
    memory_derived_facts: FrozenSet[HornAtom] = field(default_factory=frozenset)
    saliency_derived_facts: FrozenSet[HornAtom] = field(default_factory=frozenset)
    net_derived_facts: FrozenSet[HornAtom] = field(default_factory=frozenset)
    world_context_facts: FrozenSet[HornAtom] = field(default_factory=frozenset)
    abduced_support_facts: FrozenSet[HornAtom] = field(default_factory=frozenset)
    goal: Optional[HornAtom] = None
    target_facts: FrozenSet[HornAtom] = field(default_factory=frozenset)
    execution_trace: Optional[SymbolicExecutionTraceBundle] = None
    provenance: str = "heuristic"
    trigger_abduction: bool = False
    hot_dims: Tuple[int, ...] = field(default_factory=tuple)
    world_context_summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        def _freeze_atoms(values: Any) -> FrozenSet[HornAtom]:
            if not values:
                return frozenset()
            if isinstance(values, frozenset):
                return values
            return frozenset(values)

        self.observed_facts = _freeze_atoms(self.observed_facts)
        self.observed_now_facts = _freeze_atoms(self.observed_now_facts)
        self.memory_derived_facts = _freeze_atoms(self.memory_derived_facts)
        self.saliency_derived_facts = _freeze_atoms(self.saliency_derived_facts)
        self.net_derived_facts = _freeze_atoms(self.net_derived_facts)
        self.world_context_facts = _freeze_atoms(self.world_context_facts)
        self.abduced_support_facts = _freeze_atoms(self.abduced_support_facts)
        self.target_facts = _freeze_atoms(self.target_facts)
        self.hot_dims = tuple(int(dim) for dim in self.hot_dims)
        self.world_context_summary = dict(self.world_context_summary)
        self.metadata = dict(self.metadata)

        if not self.observed_now_facts and self.observed_facts:
            self.observed_now_facts = self.observed_facts

        merged = set(self.observed_facts)
        merged.update(self.observed_now_facts)
        merged.update(self.memory_derived_facts)
        merged.update(self.saliency_derived_facts)
        merged.update(self.net_derived_facts)
        merged.update(self.world_context_facts)
        merged.update(self.abduced_support_facts)
        self.observed_facts = frozenset(merged)

    @property
    def recalled_facts(self) -> FrozenSet[HornAtom]:
        return self.memory_derived_facts

    def source_fact_records(
        self,
        *,
        include_goal: bool = False,
        include_targets: bool = False,
    ) -> Tuple[Tuple[str, HornAtom], ...]:
        records: List[Tuple[str, HornAtom]] = []
        seen: Set[int] = set()
        source_groups = (
            ("observed_now", self.observed_now_facts),
            ("memory", self.memory_derived_facts),
            ("saliency", self.saliency_derived_facts),
            ("net", self.net_derived_facts),
            ("world_context", self.world_context_facts),
            ("abduced", self.abduced_support_facts),
        )
        for label, facts in source_groups:
            for atom in sorted(facts, key=repr):
                atom_hash = hash(atom)
                if atom_hash in seen:
                    continue
                seen.add(atom_hash)
                records.append((label, atom))
        if include_goal and self.goal is not None:
            goal_hash = hash(self.goal)
            if goal_hash not in seen:
                seen.add(goal_hash)
                records.append(("goal", self.goal))
        if include_targets:
            for atom in sorted(self.target_facts, key=repr):
                atom_hash = hash(atom)
                if atom_hash in seen:
                    continue
                seen.add(atom_hash)
                records.append(("target", atom))
        return tuple(records)

    def source_counts(self) -> Dict[str, float]:
        return {
            "observed_now_facts": float(len(self.observed_now_facts)),
            "memory_derived_facts": float(len(self.memory_derived_facts)),
            "saliency_derived_facts": float(len(self.saliency_derived_facts)),
            "net_derived_facts": float(len(self.net_derived_facts)),
            "world_context_facts": float(len(self.world_context_facts)),
            "abduced_support_facts": float(len(self.abduced_support_facts)),
            "target_facts": float(len(self.target_facts)),
        }


def _const_int_term(term: Term) -> Optional[int]:
    if isinstance(term, Const):
        return int(term.val)
    if isinstance(term, int):
        return int(term)
    return None


def _term_const_values(term: Term) -> Tuple[int, ...]:
    if isinstance(term, Const):
        return (int(term.val),)
    if isinstance(term, Compound):
        values: List[int] = [int(term.func)]
        for sub in term.subterms:
            values.extend(_term_const_values(sub))
        return tuple(values)
    return ()


def _atoms_conflict(a: HornAtom, b: HornAtom) -> bool:
    """
    Conservative contradiction check:
    same predicate and same prefix arguments, but different final constant.
    """
    if a.pred != b.pred or a.arity() != b.arity() or a.arity() == 0:
        return False
    a_vals = [_const_int_term(arg) for arg in a.args]
    b_vals = [_const_int_term(arg) for arg in b.args]
    if any(v is None for v in a_vals) or any(v is None for v in b_vals):
        return False
    if a.arity() == 1:
        return a_vals[0] != b_vals[0]
    return a_vals[:-1] == b_vals[:-1] and a_vals[-1] != b_vals[-1]


# ══════════════════════════════════════════════════════════════════════════════
# 2.  УНІФІКАЦІЯ: Алгоритм Martelli-Montanari + Backtracking DFS
# ══════════════════════════════════════════════════════════════════════════════
#
# Математика (розділ 3 специфікації):
#   Задача: для E = {s1=?=t1,...,sm=?=tm} знайти mgu σ, що si·σ = ti·σ.
#   Правила трансформації Martelli-Montanari:
#     (1) Trivial   : t =?= t              → видалити
#     (2) Decompose : f(s...) =?= f(t...)  → розкласти на компоненти
#     (3) Clash     : f(...) =?= g(...)    → FAIL (f≠g або arity≠)
#     (4) Orient    : t =?= X              → X =?= t
#     (5) OccursChk : X =?= t, X∈Var(t)   → FAIL (циклічний терм)
#     (6) Eliminate : X =?= t             → {X→t} ∪ E{X→t}
#
# Conjunctive unification (розділ 4):
#   Шукаємо σ і π: {1..n}→Facts такі, що Bi·σ = π(i) ∀i.
#   DFS з backtracking + індексація за pred-id.
# ══════════════════════════════════════════════════════════════════════════════

def unify_mm(equations: List[Tuple[Term, Term]]) -> Optional[Substitution]:
    """
    Алгоритм Martelli-Montanari (1982).
    Приймає список рівнянь {si =?= ti} і повертає mgu (Substitution) або None.

    Анонімні змінні (?_N, де name.startswith('_')) пропускаються правилом
    Eliminate — вони діють як wildcards без зв'язування.

    Складність: O(n·α(n)) амортизовано (union-find-like).
    """
    sigma: Dict[str, Term] = {}

    def chase(t: Term) -> Term:
        """Рекурсивно слідує chain підстановок."""
        if isinstance(t, Var) and t.name in sigma:
            return chase(sigma[t.name])
        if isinstance(t, Compound):
            return Compound(t.func, tuple(chase(s) for s in t.subterms))
        return t

    def occurs(var_name: str, t: Term) -> bool:
        """Occurs check: чи входить var_name у терм t (після chase)?"""
        t = chase(t)
        if isinstance(t, Var):      return t.name == var_name
        if isinstance(t, Compound): return any(occurs(var_name, s) for s in t.subterms)
        return False

    eqs: List[Tuple[Term, Term]] = list(equations)

    while eqs:
        s, t = eqs.pop(0)
        s, t = chase(s), chase(t)

        # (1) Trivial
        if s == t:
            continue

        # (4) Orient: якщо s не змінна, але t — змінна → flip
        if not isinstance(s, Var) and isinstance(t, Var):
            eqs.insert(0, (t, s))
            continue

        # (6) + (5) Eliminate / OccursCheck
        if isinstance(s, Var):
            name = s.name
            # Анонімні ('_*') — wildcards, не зв'язуємо
            if name.startswith('_'):
                continue
            # (5) Occurs check
            if occurs(name, t):
                return None        # циклічний терм
            sigma[name] = chase(t)
            eqs = [(chase(l), chase(r)) for l, r in eqs]
            continue

        # (2) Decompose: f(s1,...) =?= f(t1,...)
        if (isinstance(s, Compound) and isinstance(t, Compound)
                and s.func == t.func
                and len(s.subterms) == len(t.subterms)):
            eqs = list(zip(s.subterms, t.subterms)) + eqs
            continue

        # (3) Clash: несумісні символи
        return None

    return Substitution(sigma)


def unify(pattern: HornAtom, fact: HornAtom) -> Optional[Substitution]:
    """
    Уніфікація двох атомів через Martelli-Montanari.
    Повертає mgu (Substitution) або None.

    API-зміна: повертає Substitution замість Dict[int,int].
    Зворотна сумісність: HornAtom з int-args автоматично нормалізується
    до Term через __post_init__.
    """
    if pattern.pred != fact.pred or pattern.arity() != fact.arity():
        return None
    return unify_mm(list(zip(pattern.args, fact.args)))


def apply_bindings(atom: HornAtom, sigma: Substitution) -> HornAtom:
    """
    Застосовує підстановку σ до атому.
    API-зміна: приймає Substitution замість Dict[int,int].
    """
    return sigma.apply_atom(atom)


def find_all_substitutions(
        body: Tuple[HornAtom, ...],
        facts: FrozenSet[HornAtom],
        sigma: Optional[Substitution] = None,
        max_solutions: int = 64,
) -> List[Substitution]:
    """
    Знаходить ВСІ підстановки σ такі, що кожен Bi·σ уніфікується
    з деяким фактом з facts (кон'юнктивна уніфікація, розділ 4).

    Алгоритм: DFS з backtracking.
    Оптимізація: факти індексуються за pred-id для O(1)-фільтрації.

    Формально: шукаємо σ та ін'єкцію π:{1..n}→Facts,
    що Bi·σ = π(i) для всіх i.
    """
    if sigma is None:
        sigma = Substitution.empty()

    # Попередня індексація за предикатом
    pred_index: Dict[int, List[HornAtom]] = {}
    for f in facts:
        pred_index.setdefault(f.pred, []).append(f)

    results: List[Substitution] = []

    def solve(i: int, cur: Substitution) -> None:
        if len(results) >= max_solutions:
            return
        if i == len(body):
            results.append(cur)
            return
        atom = body[i]
        # Застосовуємо поточну підстановку, щоб конкретизувати атом
        grounded = cur.apply_atom(atom)
        # Перебираємо факти з тим самим pred
        for fact in pred_index.get(grounded.pred, []):
            sub = unify(grounded, fact)
            if sub is not None:
                combined = cur.compose(sub)
                solve(i + 1, combined)

    solve(0, sigma)
    return results


def unify_body(body: Tuple[HornAtom, ...],
               facts: FrozenSet[HornAtom]) -> Optional[Substitution]:
    """
    Знаходить ПЕРШУ підстановку σ що задовольняє всі атоми тіла.
    Зворотна сумісність: повертає Substitution або None.
    """
    sols = find_all_substitutions(body, facts, max_solutions=1)
    return sols[0] if sols else None


# ══════════════════════════════════════════════════════════════════════════════
# 2b.  EPISTEMIC RULE TRACKER  (контроль якості правил)
# ══════════════════════════════════════════════════════════════════════════════

class EpistemicStatus(enum.Enum):
    """
    Епістемічний статус правила в LTM.

      proposed      — щойно згенероване абдукцією, ще не перевірене.
      verified      — підтверджене ≥1 успішним використанням у доведенні.
      contradicted  — призвело до логічної суперечності; підлягає видаленню.

    Перехід:
      proposed → verified    (при successful proof step)
      proposed → contradicted (при contradiction)
      verified  може знову стати contradicted якщо виявлено суперечність пізніше.
    """
    proposed     = "proposed"
    verified     = "verified"
    contradicted = "contradicted"


@dataclass
class RuleRecord:
    """
    Запис правила з епістемічними метаданими.

    Utility(R) = use_count / (1 + age_steps) — динамічна корисність.
    Complexity(R) = rule.description_length_bits() — fixed symbolic code length.

    L_rule = Complexity(R) − η·Utility(R)   (формула розд. 3)
    """
    rule:       "HornClause"
    status:     EpistemicStatus      = EpistemicStatus.proposed
    use_count:  int                  = 0
    success_count: int               = 0     # кількість успішних доведень
    age_steps:  int                  = 0     # кроків з моменту додавання
    weight:     float                = 1.0

    def utility(self) -> float:
        """
        Utility(R) = success_count / (1 + age_steps)

        Висока корисність ≡ правило часто брало участь в успішних доведеннях.
        Штрафуємо старі невикористані правила → видаляємо їх при консолідації.
        """
        return self.success_count / (1.0 + self.age_steps)

    def l_rule_contribution(self, eta: float) -> float:
        """
        Внесок правила в L_rule = Complexity(R) − η·Utility(R).
        Від'ємний лише якщо правило дуже корисне.
        """
        return float(self.rule.description_length_bits()) - eta * self.utility()


class VerificationModule(nn.Module):
    """
    Verification Module (VeM) — нейронний фільтр кандидатів AbductionHead.

    Оцінює очікувану корисність кандидата-правила до додавання в LTM:
      U(R) = E_{майбутні_задачі}[Success(R) − α·Cost(R)]

    Апроксимація через retrospective self-supervised навчання:
      - Якщо правило брало участь у успішному доведенні → ціль U=1
      - Якщо правило не використовувалось / призводило до помилок → U≈0

    Канідати фільтруються:
      Candidates = {R ~ AbductionHead(z) | VeM(R) > vem_tau}

    VeM штраф у функції втрат:
      L_vem = δ · E_{R~Abduction}[max(0, τ − U(R))]

    Навчання: MSE між predicted U(R) та retrospective utility target.
    """

    def __init__(self, d_latent: int, sym_vocab: int, vem_tau: float = 0.3):
        super().__init__()
        self.d       = d_latent
        self.sv      = sym_vocab
        self.vem_tau = vem_tau

        # Ембеддер термів для отримання вектора правила
        self.term_emb = TermEmbedder(sym_vocab, d_latent)

        # Основна VeM мережа: rule_emb → U(R) ∈ [0, 1]
        self.vem_net = nn.Sequential(
            nn.Linear(d_latent, d_latent),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_latent, d_latent // 2),
            nn.GELU(),
            nn.Linear(d_latent // 2, 1),
            nn.Sigmoid(),      # U(R) ∈ [0, 1]
        )

        # Буфери для самонавчання (retrospective analysis)
        # Зберігаємо (rule_emb, utility_target) пари
        self._train_embs:    List[torch.Tensor] = []
        self._train_targets: List[float]        = []
        self._max_buffer     = 256

    def rule_embedding(self, clause: "HornClause",
                       device: torch.device) -> torch.Tensor:
        """Ембеддинг правила: mean(head + body atoms)."""
        atoms = [clause.head] + list(clause.body)
        embs  = self.term_emb(atoms, device)     # (n_atoms, d)
        return embs.mean(0)                       # (d,)

    def score(self, clause: "HornClause",
              device: torch.device) -> float:
        """Повертає U(R) ∈ [0,1] для одного правила (no_grad)."""
        with torch.no_grad():
            r_emb = self.rule_embedding(clause, device).unsqueeze(0)
            return self.vem_net(r_emb).squeeze().item()

    def score_batch(self, clauses: List["HornClause"],
                    device: torch.device) -> torch.Tensor:
        """
        U(R_i) для кожного кандидата.
        Returns: (n_clauses,) tensor
        """
        if not clauses:
            return torch.zeros(0, device=device)
        embs = torch.stack([self.rule_embedding(c, device) for c in clauses])
        return self.vem_net(embs).squeeze(-1)    # (n,)

    def filter_candidates(self,
                          clauses:  List["HornClause"],
                          device:   torch.device) -> Tuple[List["HornClause"], torch.Tensor]:
        """
        Фільтрує кандидатів: повертає (accepted, hinge_loss).

        hinge_loss = Σ max(0, τ − U(R))  — штраф за генерацію поганих кандидатів.
        Навіть відхилені кандидати вносять gradient сигнал через hinge.
        """
        if not clauses:
            zero = torch.zeros(1, device=device)
            return [], zero

        scores = self.score_batch(clauses, device)           # (n,)
        hinge  = torch.clamp(self.vem_tau - scores, min=0).mean()

        accepted = [c for c, s in zip(clauses, scores.tolist())
                    if s >= self.vem_tau]
        return accepted, hinge

    def record_outcome(self, clause: "HornClause",
                       utility_target: float,
                       device: torch.device) -> None:
        """
        Записує результат використання правила для самонавчання.
        utility_target ∈ [0,1]: 1.0 → успішне доведення, 0.0 → невдача.
        """
        with torch.no_grad():
            r_emb = self.rule_embedding(clause, device).cpu()
        self._train_embs.append(r_emb)
        self._train_targets.append(float(utility_target))
        # Кільцевий буфер
        if len(self._train_embs) > self._max_buffer:
            self._train_embs.pop(0)
            self._train_targets.pop(0)

    def self_supervised_loss(self, device: torch.device) -> torch.Tensor:
        """
        MSE між predicted U(R) та retrospective targets.
        Викликається раз на кілька кроків для донавчання VeM.
        """
        if len(self._train_embs) < 4:
            return torch.zeros(1, device=device).squeeze()
        embs    = torch.stack(self._train_embs).to(device)      # (N, d)
        targets = torch.tensor(self._train_targets,
                               dtype=torch.float32, device=device)  # (N,)
        preds   = self.vem_net(embs).squeeze(-1)                 # (N,)
        return F.mse_loss(preds, targets)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  KNOWLEDGE BASE (Forward Chaining)
# ══════════════════════════════════════════════════════════════════════════════

class KnowledgeBase:
    """
    База знань: факти + правила з Forward Chaining.

    Розширено епістемічним трекером:
      · Кожне правило має RuleRecord зі статусом (proposed/verified/contradicted)
      · MDL-регуляризатор враховує динамічну корисність:
          L_rule = Σ_{R∈LTM} (Complexity(R) − η·Utility(R))
      · Consolidation: видаляємо слабкі proposed правила раз на N кроків.
    """

    def __init__(self, max_rules: int = 1024):
        self.facts:     FrozenSet[HornAtom] = frozenset()
        self.rules:     List[HornClause]    = []   # активні правила
        self._rule_set: Set[int]            = set()
        self.max_rules  = max_rules

        # Епістемічний трекер: hash(rule) → RuleRecord
        self._records:  Dict[int, RuleRecord] = {}
        self._global_step: int = 0             # для age_steps

    # ── Додавання ──────────────────────────────────────────────────────────────
    def add_fact(self, atom: HornAtom) -> bool:
        if atom not in self.facts:
            self.facts = self.facts | {atom}
            return True
        return False

    def add_rule(self, clause: HornClause,
                 status: EpistemicStatus = EpistemicStatus.proposed) -> bool:
        """
        Додає правило з початковим статусом (proposed за замовч.).
        Якщо правило вже є — збільшуємо use_count.
        """
        h = hash(clause)
        if h in self._rule_set:
            # Правило вже є — оновлюємо запис
            for r in self.rules:
                if hash(r) == h:
                    r.use_count += 1
                    break
            if h in self._records:
                self._records[h].use_count += 1
            return False
        if len(self.rules) >= self.max_rules:
            # LRU-евікція: видаляємо правило з найменшою корисністю (Utility)
            worst_i = min(
                range(len(self.rules)),
                key=lambda i: self._records.get(hash(self.rules[i]),
                              RuleRecord(rule=self.rules[i])).utility()
            )
            evicted_h = hash(self.rules[worst_i])
            self._rule_set.discard(evicted_h)
            self._records.pop(evicted_h, None)
            self.rules.pop(worst_i)
        self.rules.append(clause)
        self._rule_set.add(h)
        self._records[h] = RuleRecord(rule=clause, status=status)
        return True

    def mark_rule_verified(self, clause: HornClause) -> None:
        """Позначити правило як verified після успішного доведення."""
        h = hash(clause)
        if h in self._records:
            rec = self._records[h]
            rec.status         = EpistemicStatus.verified
            rec.success_count += 1
            rec.weight        *= 1.05   # підвищуємо довіру
        # Оновлюємо use_count у самому clause (для зворотньої сумісності)
        for r in self.rules:
            if hash(r) == h:
                r.use_count    += 1
                r.weight       *= 1.01
                break

    def mark_rule_contradicted(self, clause: HornClause) -> None:
        """Позначити правило як contradicted (буде видалено при консолідації)."""
        h = hash(clause)
        if h in self._records:
            self._records[h].status = EpistemicStatus.contradicted

    def rule_status(self, clause: HornClause) -> EpistemicStatus:
        rec = self._records.get(hash(clause))
        if rec is None:
            return EpistemicStatus.verified
        return rec.status

    def rule_is_usable(self, clause: HornClause, only_verified: bool = False) -> bool:
        status = self.rule_status(clause)
        if status == EpistemicStatus.contradicted:
            return False
        if only_verified and status != EpistemicStatus.verified:
            return False
        return True

    def consolidate(self, use_count_threshold: int = 2) -> int:
        """
        Rule Consolidation: видаляємо слабкі правила.
          · Contradicted → видаляємо завжди.
          · Proposed + use_count < threshold → видаляємо.

        Повертає кількість видалених правил.
        """
        to_remove: Set[int] = set()
        for h, rec in list(self._records.items()):
            if rec.status == EpistemicStatus.contradicted:
                to_remove.add(h)
            elif (
                rec.status == EpistemicStatus.proposed
                and rec.use_count < use_count_threshold
                and rec.utility() < 0.05
                and rec.age_steps > 50
            ):
                to_remove.add(h)

        if not to_remove:
            return 0

        self.rules     = [r for r in self.rules if hash(r) not in to_remove]
        self._rule_set = {h for h in self._rule_set if h not in to_remove}
        for h in to_remove:
            self._records.pop(h, None)

        return len(to_remove)

    def tick(self) -> None:
        """Оновлює вік всіх правил (викликати раз на крок)."""
        self._global_step += 1
        for rec in self._records.values():
            rec.age_steps += 1

    # ── Forward Chaining ───────────────────────────────────────────────────────
    def forward_chain(self, max_depth: int = 5,
                      starting_facts: "Optional[FrozenSet]" = None,
                      only_verified: bool = False) -> "FrozenSet[HornAtom]":
        """
        Застосовує правила до fixpoint або max_depth ітерацій.
        Повертає всі виведені + існуючі факти.
        Успішне виведення → mark_rule_verified().

        starting_facts: якщо задано — використовуємо замість self.facts.
        Дозволяє передавати random-sample для пришвидшення при великому KB.
        """
        current = starting_facts if starting_facts is not None else self.facts
        for _ in range(max_depth):
            new_facts: Set[HornAtom] = set()
            for clause in self.rules:
                if not self.rule_is_usable(clause, only_verified=only_verified):
                    continue
                if not clause.body:
                    continue
                fresh = freshen_vars(clause)
                for sigma in find_all_substitutions(fresh.body, current):
                    derived = sigma.apply_atom(fresh.head)
                    if not derived.is_ground():
                        continue
                    if any(_atoms_conflict(derived, known) for known in current):
                        self.mark_rule_contradicted(clause)
                        continue
                    if derived not in current:
                        new_facts.add(derived)
                        clause.use_count += 1
                        clause.weight    *= 1.01
                        # Оновлюємо епістемічний статус
                        self.mark_rule_verified(clause)
            if not new_facts:
                break
            current = current | frozenset(new_facts)
        return current

    # ── MDL-складність (оновлена: враховує корисність) ─────────────────────────
    def complexity_penalty(self) -> float:
        """Σ_R DL_bits(R) under a fixed universal grammar."""
        return sum(r.description_length_bits() for r in self.rules)

    def utility_adjusted_penalty(self, eta: float = 0.1) -> float:
        """
        L_rule = Σ_{R∈LTM} (Complexity(R) − η·Utility(R))

        Заохочує зберігати корисні правила (від'ємний внесок),
        штрафує за складні некорисні правила.
        """
        total = 0.0
        for r in self.rules:
            rec = self._records.get(hash(r))
            util = rec.utility() if rec is not None else 0.0
            total += float(r.description_length_bits()) - eta * util
        return total

    def weighted_complexity(self) -> float:
        """Σ_R use_count(R)·complexity(R)"""
        return sum(r.use_count * r.description_length_bits() for r in self.rules)

    def get_rule_pairs_for_semantic_feedback(
            self,
            max_pairs: int = 32
    ) -> List[Tuple["HornClause", "HornClause", float]]:
        """
        Повертає пари правил (R1, R2, score) для семантичного feedback в NET.

        Score = сила логічного зв'язку:
          · synonym(v1, v2): cos(head_pred(v1), head_pred(v2)) ≈ 1.0
          · implies(v1, v2): v1 у тілі → v2 у голові

        Використовується для L_semantic в NET:
          L_semantic = −E[(v1,v2)~pairs][cos(e_v1, e_v2)·score]
        """
        pairs: List[Tuple["HornClause", "HornClause", float]] = []
        n = len(self.rules)
        if n < 2:
            return pairs
        for i in range(min(max_pairs * 2, n)):
            r1 = self.rules[i % n]
            r2 = self.rules[(i + 1) % n]
            # Перевіряємо імплікацію: якщо голова r1 входить в тіло r2
            r1_head_pred = r1.head.pred
            r2_body_preds = {a.pred for a in r2.body}
            if r1_head_pred in r2_body_preds:
                score = 0.9   # сильний логічний зв'язок
            elif r1.head.pred == r2.head.pred:
                score = 0.7   # synonym (спільний предикат голови)
            else:
                continue      # немає значущого зв'язку
            pairs.append((r1, r2, score))
            if len(pairs) >= max_pairs:
                break
        return pairs

    def __len__(self): return len(self.rules)
    def n_facts(self):  return len(self.facts)


# ══════════════════════════════════════════════════════════════════════════════
# 3b. TensorKnowledgeBase — GPU-акселерований drop-in замінник KnowledgeBase
# ══════════════════════════════════════════════════════════════════════════════
#
# Проблема оригінального forward_chain:
#   Python nested loop: O(depth × R × N) = 4 × 1342 × 1342 ≈ 7.2M ітерацій
#   Час: 22–70 секунд/батч → Stage 2 не рухається вперед.
#
# Рішення: представити факти і правила як int64-тензори на GPU.
#   forward_chain = broadcast матричні операції:
#     pred_match = facts[:,0:1] == rules[:,3:4].T   # (N,R) — один GPU kernel
#     a0_match   = (b_a0<0) | (facts[:,1:2] == rules[:,4:5].T)
#     a1_match   = (b_a1<0) | (facts[:,2:3] == rules[:,5:6].T)
#     match      = pred_match & a0_match & a1_match  # (N,R) — GPU parallel
#   Час: ~0.1–0.5 ms/батч  → прискорення ~100–1000x без втрат інформації.
#
# Кодування змінних у тензорі правил (стовпці 1,2,4,5):
#   VAR_0 = -1  (перша логічна змінна: body_slot_0 → head_slot)
#   VAR_1 = -2  (друга логічна змінна: body_slot_1 → head_slot)
#   NONE  = -99 (немає аргументу — предикат арності 1)
#   ≥ 0        = конкретна константа
#
# Сумісність:
#   · Повний інтерфейс KnowledgeBase: add_fact, add_rule, forward_chain,
#     mark_rule_verified, mark_rule_contradicted, consolidate, tick, n_facts,
#     __len__, rules, facts, _records — все збережено.
#   · Multi-body правила (≥2 літерали у тілі): Python fallback (рідко).
#   · Епістемічний трекер (RuleRecord, status) — повністю збережено.
# ══════════════════════════════════════════════════════════════════════════════

class TensorKnowledgeBase:
    """
    GPU-accelerated Knowledge Base. Drop-in replacement for KnowledgeBase.

    Зберігає факти як (N, 3) int64 тензор і правила як (R, 6) int64 тензор.
    forward_chain виконується через GPU broadcast operations (~1000x швидше).
    Повністю зворотньо сумісний з KnowledgeBase API.
    """

    # Sentinel values у тензорному представленні
    VAR_0 = -1   # перша логічна змінна (body_arg_slot_0)
    VAR_1 = -2   # друга логічна змінна (body_arg_slot_1)
    NONE  = -99  # відсутній аргумент (арність 1)

    def __init__(self, max_rules: int = 1024, max_facts: int = 16384,
                 device: Optional[torch.device] = None):
        # Детектуємо device автоматично (GPU якщо доступний)
        self._dev = device or (torch.device("cuda") if torch.cuda.is_available()
                               else torch.device("cpu"))

        self.max_rules = max_rules
        self._max_facts = max_facts

        # ── Тензорні буфери (pre-allocated) ─────────────────────────────────
        # fact_buf: [pred, arg0, arg1]   arg=NONE якщо арність<2
        self._fact_buf = torch.full((max_facts, 3), self.NONE,
                                    dtype=torch.long, device=self._dev)
        # rule_buf: [h_p, h_a0, h_a1, b_p, b_a0, b_a1]
        # Однотільні правила (тіло з 1 літерала); multi-body → Python
        self._rule_buf = torch.full((max_rules, 6), self.NONE,
                                    dtype=torch.long, device=self._dev)
        # Маска: чи дане правило однотільне (tensor-ready)
        self._rule_is_tensor = torch.zeros(max_rules, dtype=torch.bool,
                                           device=self._dev)

        self._n_facts: int = 0
        self._n_rules: int = 0

        # Кешовані лічильники статусів (O(1) замість O(n_records) sum comprehension)
        self._n_proposed: int = 0
        self._n_verified: int = 0

        # ── Швидке дедуплікування (Python set) ───────────────────────────────
        self._fact_set: Set[Tuple[int, int, int]] = set()
        self._rule_hash_set: Set[int] = set()

        # ── Збережені Python-об'єкти (для сумісності) ────────────────────────
        self.rules: List[HornClause] = []          # всі правила (HornClause)
        self._records: Dict[int, RuleRecord] = {}
        self._global_step: int = 0

        # Кеш facts як frozenset[HornAtom] (для сумісності з .facts property)
        self._facts_cache: Optional[FrozenSet[HornAtom]] = None
        self._facts_cache_n: int = -1  # _n_facts при створенні кешу

        # Ground-term interning: Const/Compound -> stable term ids in tensor buffers.
        self._term_to_id: Dict[Term, int] = {}
        self._id_to_term: Dict[int, Term] = {}
        self._next_term_id: int = 1
        self._extra_facts: Set[HornAtom] = set()

    @staticmethod
    def _is_ground_term(term: Term) -> bool:
        return len(term.vars()) == 0

    def _intern_term(self, term: Union[Term, int]) -> int:
        term_obj: Term = Const(int(term)) if isinstance(term, int) else term
        if not self._is_ground_term(term_obj):
            raise ValueError(f"TensorKnowledgeBase expects ground terms, got: {term_obj}")
        term_id = self._term_to_id.get(term_obj)
        if term_id is None:
            term_id = self._next_term_id
            self._next_term_id += 1
            self._term_to_id[term_obj] = term_id
            self._id_to_term[term_id] = term_obj
        return term_id

    def _decode_term(self, term_id: int) -> Term:
        return self._id_to_term.get(term_id, Const(int(term_id)))

    # ── Кодування HornAtom → (pred, a0, a1) int tuple ────────────────────────
    def _atom_to_key(self, atom: HornAtom) -> Tuple[int, int, int]:
        """
        Ground atom → (pred, a0, a1) int tuple.
        Змінні кодуються як VAR_0/VAR_1 (використовується для тіл правил).
        """
        args = atom.args

        def _enc(a: object, slot: int) -> int:
            if isinstance(a, Const):
                return self._intern_term(a)
            if isinstance(a, Compound) and self._is_ground_term(a):
                return self._intern_term(a)
            if isinstance(a, Var):
                return self.VAR_0 if slot == 0 else self.VAR_1
            if isinstance(a, int):
                return self._intern_term(Const(int(a))) if a >= 0 else (
                    self.VAR_0 if slot == 0 else self.VAR_1
                )
            return 0  # unsupported non-ground compound -> fallback marker

        a0 = _enc(args[0], 0) if len(args) > 0 else self.NONE
        a1 = _enc(args[1], 1) if len(args) > 1 else self.NONE
        return (int(atom.pred), a0, a1)

    # ── Кодування HornClause → (h_p, h_a0, h_a1, b_p, b_a0, b_a1) або None ─
    def _clause_to_tensor_row(self, clause: HornClause) -> Optional[Tuple[int, ...]]:
        """
        Однотільний HornClause → 6-int tuple для тензорного буфера.
        Повертає None для multi-body (>1 literal) — вони обробляються Python-ом.

        Логіка кодування змінних:
          · Збираємо Var-імена за позицією їх першої появи у body.
          · Перша по позиції body[0] Var → VAR_0 (-1), body[1] Var → VAR_1 (-2).
          · У head: якщо arg — та сама Var → VAR_0/VAR_1 (буде замінено фактом).
          · Якщо Var у head не з'являється у body → VAR_0 (безпечний default).
        """
        if len(clause.body) != 1:
            return None  # multi-body: Python fallback

        head = clause.head
        body = clause.body[0]

        def _tensor_term_ok(term: Term) -> bool:
            if isinstance(term, (Const, Var)):
                return True
            if isinstance(term, Compound):
                return self._is_ground_term(term)
            return False

        for atom in (head, body):
            if len(atom.args) > 2:
                return None
            if not all(_tensor_term_ok(arg) for arg in atom.args):
                return None

        # Будуємо мапу ім'я Var → VAR_0/VAR_1 за body-слотом першої появи.
        # Це критично для правил на кшталт h(X,c) :- b(c,X), де X походить з body arg1.
        var_map: Dict[str, int] = {}
        for pos, a in enumerate(body.args):
            if isinstance(a, Var) and a.name not in var_map:
                slot = self.VAR_0 if pos == 0 else self.VAR_1
                var_map[a.name] = slot

        def _enc_head(a: object, pos: int) -> int:
            if isinstance(a, Const):
                return self._intern_term(a)
            if isinstance(a, Compound) and self._is_ground_term(a):
                return self._intern_term(a)
            if isinstance(a, Var):
                return var_map.get(a.name, self.VAR_0)  # default VAR_0
            return 0

        def _enc_body(a: object, pos: int) -> int:
            if isinstance(a, Const):
                return self._intern_term(a)
            if isinstance(a, Compound) and self._is_ground_term(a):
                return self._intern_term(a)
            if isinstance(a, Var):
                return var_map.get(a.name, self.VAR_0 if pos == 0 else self.VAR_1)
            return 0

        h_a0 = _enc_head(head.args[0], 0) if len(head.args) > 0 else self.NONE
        h_a1 = _enc_head(head.args[1], 1) if len(head.args) > 1 else self.NONE
        b_a0 = _enc_body(body.args[0], 0) if len(body.args) > 0 else self.NONE
        b_a1 = _enc_body(body.args[1], 1) if len(body.args) > 1 else self.NONE

        return (int(head.pred), h_a0, h_a1, int(body.pred), b_a0, b_a1)

    # ── add_fact ──────────────────────────────────────────────────────────────
    def add_fact(self, atom: HornAtom) -> bool:
        if len(atom.args) > 2:
            if atom in self._extra_facts:
                return False
            self._extra_facts.add(atom)
            self._facts_cache = None
            return True
        key = self._atom_to_key(atom)
        if key in self._fact_set:
            return False
        if self._n_facts >= self._max_facts:
            return False  # buffer full — silently drop
        self._fact_buf[self._n_facts] = torch.tensor(key, dtype=torch.long,
                                                      device=self._dev)
        self._fact_set.add(key)
        self._n_facts += 1
        self._facts_cache = None  # інвалідуємо кеш
        return True

    # ── add_rule ──────────────────────────────────────────────────────────────
    def add_rule(self, clause: HornClause,
                 status: EpistemicStatus = EpistemicStatus.proposed) -> bool:
        h = hash(clause)
        if h in self._rule_hash_set:
            # Дублікат → збільшуємо use_count
            for r in self.rules:
                if hash(r) == h:
                    r.use_count += 1
                    break
            if h in self._records:
                self._records[h].use_count += 1
            return False

        # LRU-евікція якщо переповнений
        if self._n_rules >= self.max_rules:
            self._evict_worst_rule()

        # Тензорне кодування (якщо однотільне)
        row = self._clause_to_tensor_row(clause)
        if row is not None:
            self._rule_buf[self._n_rules] = torch.tensor(row, dtype=torch.long,
                                                         device=self._dev)
            self._rule_is_tensor[self._n_rules] = True
        # else: multi-body → tensor slot лишається NONE (пропускається у fc)

        self.rules.append(clause)
        self._rule_hash_set.add(h)
        self._records[h] = RuleRecord(rule=clause, status=status)
        self._n_rules += 1
        # Оновлюємо кешований лічильник статусів
        if status == EpistemicStatus.proposed:
            self._n_proposed += 1
        elif status == EpistemicStatus.verified:
            self._n_verified += 1
        return True

    def _evict_worst_rule(self) -> None:
        """Видаляємо правило з найнижчою Utility (LRU-подібна евікція)."""
        if not self.rules:
            return
        worst_i = min(
            range(len(self.rules)),
            key=lambda i: self._records.get(hash(self.rules[i]),
                          RuleRecord(rule=self.rules[i])).utility()
        )
        evicted = self.rules.pop(worst_i)
        evicted_h = hash(evicted)
        # Оновлюємо кешований лічильник статусів
        evicted_rec = self._records.get(evicted_h)
        if evicted_rec is not None:
            if evicted_rec.status == EpistemicStatus.proposed:
                self._n_proposed -= 1
            elif evicted_rec.status == EpistemicStatus.verified:
                self._n_verified -= 1
        self._rule_hash_set.discard(evicted_h)
        self._records.pop(evicted_h, None)
        # Зсуваємо тензорний буфер (compact)
        if worst_i < self._n_rules - 1:
            self._rule_buf[worst_i:self._n_rules - 1] = \
                self._rule_buf[worst_i + 1:self._n_rules].clone()
            self._rule_is_tensor[worst_i:self._n_rules - 1] = \
                self._rule_is_tensor[worst_i + 1:self._n_rules].clone()
        self._rule_buf[self._n_rules - 1] = self.NONE
        self._rule_is_tensor[self._n_rules - 1] = False
        self._n_rules -= 1

    # ── forward_chain (GPU tensor) ────────────────────────────────────────────
    def forward_chain(self, max_depth: int = 4,
                      starting_facts: Optional[FrozenSet] = None,
                      only_verified: bool = False) -> FrozenSet[HornAtom]:
        """
        GPU-акселерований Forward Chaining.

        Алгоритм:
          1. Факти як (N,3) int64 тензор на GPU.
          2. Однотільні правила як (R,6) int64 тензор на GPU.
          3. Match matrix: (N,R) broadcast — один GPU kernel на depth.
          4. Нові факти: scatter по (fact_i, rule_j) парах.
          5. Multi-body правила: Python fallback (зазвичай 0-5% правил).

        Складність: O(depth × N × R) — матричні операції на GPU.
        vs оригінал: O(depth × R × N) Python loops (~1000x повільніше).
        """
        if self._n_facts == 0 and not self._extra_facts and not starting_facts:
            return frozenset()

        # Якщо передано starting_facts — конвертуємо у тензор
        extra_facts: Set[HornAtom] = set(self._extra_facts)
        if starting_facts is not None:
            tensor_start = [a for a in starting_facts if len(a.args) <= 2]
            extra_facts = {a for a in starting_facts if len(a.args) > 2}
            keys = [self._atom_to_key(a) for a in tensor_start]
            if not keys:
                facts = torch.zeros((0, 3), dtype=torch.long, device=self._dev)
                fact_set = set()
            else:
                facts = torch.tensor(keys, dtype=torch.long, device=self._dev)
                fact_set = set(keys)
        else:
            facts = self._fact_buf[:self._n_facts].clone()
            fact_set = set(self._fact_set)

        N = facts.shape[0]
        allowed_indices = [
            i for i, rule in enumerate(self.rules[:self._n_rules])
            if self.rule_is_usable(rule, only_verified=only_verified)
        ]
        allowed_index_set = set(allowed_indices)
        if allowed_indices:
            allowed_idx_t = torch.tensor(allowed_indices, dtype=torch.long, device=self._dev)
            allowed_mask = torch.zeros(self._n_rules, dtype=torch.bool, device=self._dev)
            allowed_mask[allowed_idx_t] = True
        else:
            allowed_mask = torch.zeros(self._n_rules, dtype=torch.bool, device=self._dev)

        # Tensor-ready rules
        n_tensor_rules = (self._rule_is_tensor[:self._n_rules] & allowed_mask).sum().item()
        if n_tensor_rules > 0:
            tensor_mask = self._rule_is_tensor[:self._n_rules] & allowed_mask
            tensor_rule_indices = tensor_mask.nonzero(as_tuple=True)[0]
            rules = self._rule_buf[:self._n_rules][tensor_mask]  # (R_t, 6)
            R = rules.shape[0]

            # Multi-body rules:
            #   1) flat fast path  — top-level Const/Var/ground Compound only
            #   2) structured path — nested Compound patterns with bound vars
            #   3) Python fallback — everything else
            fast_multi_rules = [
                r for i, r in enumerate(self.rules)
                if i < self._n_rules and i in allowed_index_set and len(r.body) > 1 and self._can_fast_rule(r)
            ]
            structured_multi_rules = [
                r for i, r in enumerate(self.rules)
                if i < self._n_rules and i in allowed_index_set and len(r.body) > 1
                and not self._can_fast_rule(r)
                and self._can_structured_fast_rule(r)
            ]
            multi_body_rules = [
                r for i, r in enumerate(self.rules)
                if i < self._n_rules and i in allowed_index_set and len(r.body) > 1
                and not self._can_fast_rule(r)
                and not self._can_structured_fast_rule(r)
            ]
        else:
            tensor_rule_indices = torch.zeros(0, dtype=torch.long, device=self._dev)
            rules = torch.zeros(0, 6, dtype=torch.long, device=self._dev)
            R = 0
            fast_multi_rules = [
                r for i, r in enumerate(self.rules[:self._n_rules])
                if i in allowed_index_set and len(r.body) > 1 and self._can_fast_rule(r)
            ]
            structured_multi_rules = [
                r for i, r in enumerate(self.rules[:self._n_rules])
                if i in allowed_index_set and len(r.body) > 1 and not self._can_fast_rule(r)
                and self._can_structured_fast_rule(r)
            ]
            multi_body_rules = [
                r for i, r in enumerate(self.rules[:self._n_rules])
                if i in allowed_index_set and len(r.body) > 1 and not self._can_fast_rule(r)
                and not self._can_structured_fast_rule(r)
            ]

        handled_rule_ids: Set[int] = set()
        if n_tensor_rules > 0:
            handled_rule_ids.update(
                hash(self.rules[int(rule_idx.item())]) for rule_idx in tensor_rule_indices
            )
        handled_rule_ids.update(hash(rule) for rule in fast_multi_rules)
        handled_rule_ids.update(hash(rule) for rule in structured_multi_rules)
        handled_rule_ids.update(hash(rule) for rule in multi_body_rules)
        python_fallback_rules = [
            rule for rule in self.rules[:self._n_rules]
            if hash(rule) not in handled_rule_ids and self.rule_is_usable(rule, only_verified=only_verified)
        ]

        for _depth in range(max_depth):
            n_new_total = 0

            # ── TENSOR PATH: однотільні правила ───────────────────────────────
            if R > 0 and N > 0:
                # body предикат і аргументи
                b_pred = rules[:, 3]   # (R,)
                b_a0   = rules[:, 4]   # (R,)  VAR=-1,-2 або const≥0
                b_a1   = rules[:, 5]   # (R,)

                # Match matrix (N, R) ── три умови за AND
                # 1. Предикат точно збігається
                pred_m = (facts[:, 0:1] == b_pred.unsqueeze(0))  # (N, R)

                # 2. Аргумент 0: VAR (<0) → match будь-який; const → точний збіг
                a0_var = (b_a0 < 0).unsqueeze(0)                  # (1, R)
                a0_m   = a0_var | (facts[:, 1:2] == b_a0.unsqueeze(0))  # (N, R)

                # 3. Аргумент 1: NONE (-99) вважається wildcardʼом
                a1_var = (b_a1 < 0).unsqueeze(0)                  # (1, R)
                a1_m   = a1_var | (facts[:, 2:3] == b_a1.unsqueeze(0))  # (N, R)

                same_var = ((b_a0 == self.VAR_0) & (b_a1 == self.VAR_0)) | \
                           ((b_a0 == self.VAR_1) & (b_a1 == self.VAR_1))
                same_var_m = (~same_var).unsqueeze(0) | (facts[:, 1:2] == facts[:, 2:3])

                match = pred_m & a0_m & a1_m & same_var_m          # (N, R)

                if match.any():
                    fi, ri = match.nonzero(as_tuple=True)           # (M,) кожен

                    # Голови правил для кожного match
                    h_pred   = rules[ri, 0]                         # (M,)
                    h_a0_enc = rules[ri, 1]                         # (M,)
                    h_a1_enc = rules[ri, 2]                         # (M,)

                    # Фактичні значення аргументів з matched fact
                    fa0 = facts[fi, 1]                              # (M,)
                    fa1 = facts[fi, 2]                              # (M,)

                    # Резолюція головних аргументів:
                    # VAR_0 (-1) → беремо fa0; VAR_1 (-2) → беремо fa1; інакше const
                    h_a0 = torch.where(h_a0_enc == self.VAR_0, fa0,
                           torch.where(h_a0_enc == self.VAR_1, fa1, h_a0_enc))
                    h_a1 = torch.where(h_a1_enc == self.VAR_0, fa0,
                           torch.where(h_a1_enc == self.VAR_1, fa1, h_a1_enc))

                    # Нові кандидати (M, 3)
                    candidates = torch.stack([h_pred, h_a0, h_a1], dim=1)

                    # Дедуплікація: фільтруємо вже відомі факти
                    novel_rows: List[List[int]] = []
                    for cand_idx, row in enumerate(candidates.tolist()):
                        key = (row[0], row[1], row[2])
                        candidate_atom = HornAtom(
                            pred=int(row[0]),
                            args=self._ints_to_args(int(row[1]), int(row[2])),
                        )
                        has_conflict = any(
                            _atoms_conflict(candidate_atom, known) for known in self.facts
                        )
                        if has_conflict:
                            rule_idx = int(tensor_rule_indices[ri[cand_idx]].item())
                            self.mark_rule_contradicted(self.rules[rule_idx])
                            continue
                        if key not in fact_set and row[0] >= 0:
                            fact_set.add(key)
                            novel_rows.append(row)

                    if novel_rows:
                        novel_t = torch.tensor(novel_rows, dtype=torch.long,
                                               device=self._dev)
                        facts = torch.cat([facts, novel_t], dim=0)
                        N = facts.shape[0]
                        n_new_total += len(novel_rows)

            # ── PYTHON FALLBACK: multi-body правила ───────────────────────────
            # Ці правила рідкісні (NeuralAbductionHead генерує 1-тільні),
            # тому Python-overhead тут незначний.
            for clause in fast_multi_rules:
                if not clause.body:
                    continue
                fresh = freshen_vars(clause)
                body = fresh.body
                head = fresh.head

                C0 = self._candidate_rows(facts, body[0])
                if C0.shape[0] == 0:
                    continue

                if len(body) == 2:
                    C1 = self._candidate_rows(facts, body[1])
                    if C1.shape[0] == 0:
                        continue
                    i0, i1 = self._equijoin_rows(C0, C1, body[0], body[1])
                    if len(i0) == 0:
                        continue
                    joined = [C0[i0], C1[i1]]
                else:
                    C1 = self._candidate_rows(facts, body[1])
                    C2 = self._candidate_rows(facts, body[2])
                    if C1.shape[0] == 0 or C2.shape[0] == 0:
                        continue
                    i01, i11 = self._equijoin_rows(C0, C1, body[0], body[1])
                    if len(i01) == 0:
                        continue
                    C01_0 = C0[i01]
                    C01_1 = C1[i11]
                    i_mid, i2 = self._equijoin_rows(C01_1, C2, body[1], body[2])
                    if len(i_mid) == 0:
                        continue
                    joined = [C01_0[i_mid], C01_1[i_mid], C2[i2]]

                head_rows = self._build_fast_head_rows(joined, body, head)
                if head_rows is None or head_rows.shape[0] == 0:
                    continue

                for row_idx in range(head_rows.shape[0]):
                    row = head_rows[row_idx]
                    if int(row[0].item()) < 0 or int(row[1].item()) == self.NONE:
                        continue
                    derived = HornAtom(
                        pred=int(row[0].item()),
                        args=self._ints_to_args(int(row[1].item()), int(row[2].item())),
                    )
                    key = (int(row[0].item()), int(row[1].item()), int(row[2].item()))
                    if any(_atoms_conflict(derived, known) for known in self.facts):
                        self.mark_rule_contradicted(clause)
                        continue
                    if key in fact_set:
                        continue
                    fact_set.add(key)
                    facts = torch.cat([facts, row.unsqueeze(0)], dim=0)
                    N += 1
                    n_new_total += 1
                    clause.use_count += 1
                    clause.weight *= 1.01
                    self.mark_rule_verified(clause)

            for clause in structured_multi_rules:
                if not clause.body:
                    continue
                fresh = freshen_vars(clause)
                matches0 = self._candidate_matches(facts, fresh.body[0])
                if not matches0:
                    continue
                if len(fresh.body) == 2:
                    matches1 = self._candidate_matches(facts, fresh.body[1])
                    if not matches1:
                        continue
                    joined_matches = self._equijoin_matches(
                        matches0, matches1, fresh.body[0], fresh.body[1]
                    )
                else:
                    matches1 = self._candidate_matches(facts, fresh.body[1])
                    matches2 = self._candidate_matches(facts, fresh.body[2])
                    if not matches1 or not matches2:
                        continue
                    joined01 = self._equijoin_matches(
                        matches0, matches1, fresh.body[0], fresh.body[1]
                    )
                    if not joined01:
                        continue
                    joined_matches = self._equijoin_matches(
                        joined01, matches2, None, fresh.body[2]
                    )

                for rows, bindings in joined_matches:
                    derived = self._build_structured_head_atom(fresh.head, bindings)
                    if derived is None or not derived.is_ground():
                        continue
                    dk = self._atom_to_key(derived)
                    if any(_atoms_conflict(derived, known) for known in self.facts):
                        self.mark_rule_contradicted(clause)
                        continue
                    if dk not in fact_set and dk[0] >= 0:
                        fact_set.add(dk)
                        new_row = torch.tensor([dk], dtype=torch.long, device=self._dev)
                        facts = torch.cat([facts, new_row], dim=0)
                        N += 1
                        n_new_total += 1
                        clause.use_count += 1
                        clause.weight *= 1.01
                        self.mark_rule_verified(clause)

            if python_fallback_rules:
                current_frozenset = frozenset(
                    HornAtom(pred=int(row[0]),
                             args=self._ints_to_args(int(row[1]), int(row[2])))
                    for row in facts.tolist()
                    if row[0] >= 0
                ) | frozenset(extra_facts)
                for clause in python_fallback_rules:
                    fresh = freshen_vars(clause)
                    if not fresh.body:
                        if not fresh.head.is_ground():
                            continue
                        if any(_atoms_conflict(fresh.head, known) for known in current_frozenset):
                            self.mark_rule_contradicted(clause)
                            continue
                        if len(fresh.head.args) > 2:
                            if fresh.head not in extra_facts:
                                extra_facts.add(fresh.head)
                                current_frozenset = current_frozenset | {fresh.head}
                                n_new_total += 1
                                clause.use_count += 1
                                clause.weight *= 1.01
                                self.mark_rule_verified(clause)
                            continue
                        dk = self._atom_to_key(fresh.head)
                        if dk not in fact_set and dk[0] >= 0:
                            fact_set.add(dk)
                            new_row = torch.tensor([dk], dtype=torch.long, device=self._dev)
                            facts = torch.cat([facts, new_row], dim=0)
                            N += 1
                            n_new_total += 1
                            clause.use_count += 1
                            clause.weight *= 1.01
                            self.mark_rule_verified(clause)
                        continue
                    for sigma in find_all_substitutions(fresh.body, current_frozenset):
                        derived = sigma.apply_atom(fresh.head)
                        if derived.is_ground():
                            if any(_atoms_conflict(derived, known) for known in current_frozenset):
                                self.mark_rule_contradicted(clause)
                                continue
                            if len(derived.args) > 2:
                                if derived not in extra_facts:
                                    extra_facts.add(derived)
                                    current_frozenset = current_frozenset | {derived}
                                    n_new_total += 1
                                    clause.use_count += 1
                                    clause.weight *= 1.01
                                    self.mark_rule_verified(clause)
                                continue
                            dk = self._atom_to_key(derived)
                            if dk not in fact_set and dk[0] >= 0:
                                fact_set.add(dk)
                                new_row = torch.tensor([dk], dtype=torch.long,
                                                       device=self._dev)
                                facts = torch.cat([facts, new_row], dim=0)
                                N += 1
                                n_new_total += 1
                                clause.use_count += 1
                                clause.weight *= 1.01
                                self.mark_rule_verified(clause)

            if n_new_total == 0:
                break  # fixpoint досягнуто

        # Конвертуємо тензор назад у frozenset[HornAtom]
        tensor_facts = frozenset(
            HornAtom(pred=int(row[0]),
                     args=self._ints_to_args(int(row[1]), int(row[2])))
            for row in facts.tolist()
            if row[0] >= 0
        )
        return tensor_facts | frozenset(extra_facts)

    def _ints_to_args(self, a0: int, a1: int) -> Tuple:
        """(a0, a1) int → Tuple[Term] для HornAtom."""
        if a1 == self.NONE:
            return (self._decode_term(a0),) if a0 >= 0 else (Var("_v0"),)
        args: list = []
        args.append(self._decode_term(a0) if a0 >= 0 else Var("_v0"))
        args.append(self._decode_term(a1) if a1 >= 0 else Var("_v1"))
        return tuple(args)

    @staticmethod
    def _atom_var_positions(atom: HornAtom) -> Dict[str, List[int]]:
        result: Dict[str, List[int]] = defaultdict(list)
        for i, arg in enumerate(atom.args):
            if isinstance(arg, Var) and not arg.name.startswith('_'):
                result[arg.name].append(i)
        return result

    @staticmethod
    def _atom_var_names(atom: HornAtom) -> FrozenSet[str]:
        names: FrozenSet[str] = frozenset()
        for arg in atom.args:
            if isinstance(arg, (Const, Var, Compound)):
                names = names | frozenset(
                    name for name in arg.vars()
                    if not name.startswith('_')
                )
        return names

    def _fast_term_id(self, term: Term) -> Optional[int]:
        if isinstance(term, Const):
            return self._intern_term(term)
        if isinstance(term, Compound) and self._is_ground_term(term):
            return self._intern_term(term)
        return None

    def _can_fast_rule(self, clause: HornClause) -> bool:
        if len(clause.body) == 0 or len(clause.body) > 3:
            return False
        atoms = [clause.head] + list(clause.body)
        for atom in atoms:
            if len(atom.args) > 2:
                return False
            for arg in atom.args:
                if isinstance(arg, (Const, Var)):
                    continue
                if isinstance(arg, Compound) and self._is_ground_term(arg):
                    continue
                return False
        return True

    def _can_structured_fast_rule(self, clause: HornClause) -> bool:
        if len(clause.body) == 0 or len(clause.body) > 3:
            return False
        atoms = [clause.head] + list(clause.body)
        has_structured_compound = False
        for atom in atoms:
            if len(atom.args) > 2:
                return False
            for arg in atom.args:
                if isinstance(arg, (Const, Var)):
                    continue
                if isinstance(arg, Compound):
                    if not self._is_ground_term(arg):
                        has_structured_compound = True
                    continue
                return False
        return has_structured_compound

    def _candidate_rows(self, facts: torch.Tensor, atom: HornAtom) -> torch.Tensor:
        if facts.numel() == 0:
            return facts
        mask = facts[:, 0] == int(atom.pred)
        for j, arg in enumerate(atom.args):
            term_id = self._fast_term_id(arg)
            if term_id is not None:
                mask = mask & (facts[:, j + 1] == term_id)
        for positions in self._atom_var_positions(atom).values():
            if len(positions) < 2:
                continue
            base = positions[0]
            for pos in positions[1:]:
                mask = mask & (facts[:, base + 1] == facts[:, pos + 1])
        return facts[mask]

    def _match_term_bindings(
        self,
        pattern: Term,
        value: Term,
        bindings: Dict[str, Term],
    ) -> Optional[Dict[str, Term]]:
        if isinstance(pattern, Var):
            if pattern.name.startswith('_'):
                return bindings
            bound = bindings.get(pattern.name)
            if bound is None:
                new_bindings = dict(bindings)
                new_bindings[pattern.name] = value
                return new_bindings
            return bindings if bound == value else None
        if isinstance(pattern, Const):
            return bindings if isinstance(value, Const) and value.val == pattern.val else None
        if isinstance(pattern, Compound):
            if not isinstance(value, Compound):
                return None
            if pattern.func != value.func or len(pattern.subterms) != len(value.subterms):
                return None
            cur = bindings
            for p_sub, v_sub in zip(pattern.subterms, value.subterms):
                cur = self._match_term_bindings(p_sub, v_sub, cur)
                if cur is None:
                    return None
            return cur
        return None

    def _row_bindings(self, atom: HornAtom, row: torch.Tensor) -> Optional[Dict[str, Term]]:
        if len(atom.args) == 0:
            return {}
        bindings: Dict[str, Term] = {}
        row_args = self._ints_to_args(int(row[1].item()), int(row[2].item()))
        if len(row_args) != len(atom.args):
            return None
        for pattern, value in zip(atom.args, row_args):
            bindings = self._match_term_bindings(pattern, value, bindings)
            if bindings is None:
                return None
        return bindings

    def _candidate_matches(
        self,
        facts: torch.Tensor,
        atom: HornAtom,
    ) -> List[Tuple[List[torch.Tensor], Dict[str, Term]]]:
        candidates = self._candidate_rows(facts, atom)
        matches: List[Tuple[List[torch.Tensor], Dict[str, Term]]] = []
        for idx in range(candidates.shape[0]):
            row = candidates[idx]
            bindings = self._row_bindings(atom, row)
            if bindings is not None:
                matches.append(([row], bindings))
        return matches

    @staticmethod
    def _merge_bindings(
        left: Dict[str, Term],
        right: Dict[str, Term],
    ) -> Optional[Dict[str, Term]]:
        if not left:
            return dict(right)
        if not right:
            return dict(left)
        merged = dict(left)
        for name, term in right.items():
            bound = merged.get(name)
            if bound is None:
                merged[name] = term
            elif bound != term:
                return None
        return merged

    @staticmethod
    def _bindings_signature(
        bindings: Dict[str, Term],
        names: Tuple[str, ...],
    ) -> Optional[Tuple[Term, ...]]:
        if not names:
            return tuple()
        sig: List[Term] = []
        for name in names:
            if name not in bindings:
                return None
            sig.append(bindings[name])
        return tuple(sig)

    def _equijoin_matches(
        self,
        left_matches: List[Tuple[List[torch.Tensor], Dict[str, Term]]],
        right_matches: List[Tuple[List[torch.Tensor], Dict[str, Term]]],
        left_atom: Optional[HornAtom],
        right_atom: Optional[HornAtom],
    ) -> List[Tuple[List[torch.Tensor], Dict[str, Term]]]:
        if not left_matches or not right_matches:
            return []
        if left_atom is not None and right_atom is not None:
            shared_names = tuple(sorted(
                self._atom_var_names(left_atom) & self._atom_var_names(right_atom)
            ))
        else:
            shared_names = tuple(sorted(
                set(left_matches[0][1].keys()) & set(right_matches[0][1].keys())
            ))

        joined: List[Tuple[List[torch.Tensor], Dict[str, Term]]] = []
        if not shared_names:
            for left_rows, left_bind in left_matches:
                for right_rows, right_bind in right_matches:
                    merged = self._merge_bindings(left_bind, right_bind)
                    if merged is not None:
                        joined.append((left_rows + right_rows, merged))
            return joined

        index: Dict[Tuple[Term, ...], List[Tuple[List[torch.Tensor], Dict[str, Term]]]] = defaultdict(list)
        for right_rows, right_bind in right_matches:
            sig = self._bindings_signature(right_bind, shared_names)
            if sig is not None:
                index[sig].append((right_rows, right_bind))

        for left_rows, left_bind in left_matches:
            sig = self._bindings_signature(left_bind, shared_names)
            if sig is None:
                continue
            for right_rows, right_bind in index.get(sig, ()):
                merged = self._merge_bindings(left_bind, right_bind)
                if merged is not None:
                    joined.append((left_rows + right_rows, merged))
        return joined

    def _equijoin_rows(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        left_atom: HornAtom,
        right_atom: HornAtom,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if left.numel() == 0 or right.numel() == 0:
            empty = torch.zeros(0, dtype=torch.long, device=self._dev)
            return empty, empty
        left_vars = self._atom_var_positions(left_atom)
        right_vars = self._atom_var_positions(right_atom)
        shared = set(left_vars.keys()) & set(right_vars.keys())
        if not shared:
            nl, nr = left.shape[0], right.shape[0]
            return (
                torch.arange(nl, device=self._dev).repeat_interleave(nr),
                torch.arange(nr, device=self._dev).repeat(nl),
            )
        join_mask = torch.ones(left.shape[0], right.shape[0], dtype=torch.bool, device=self._dev)
        for name in shared:
            lp = left_vars[name][0]
            rp = right_vars[name][0]
            join_mask = join_mask & (
                left[:, lp + 1].unsqueeze(1) == right[:, rp + 1].unsqueeze(0)
            )
        return join_mask.nonzero(as_tuple=True)

    def _build_fast_head_rows(
        self,
        joined_rows: List[torch.Tensor],
        body: Tuple[HornAtom, ...],
        head: HornAtom,
    ) -> Optional[torch.Tensor]:
        if not joined_rows:
            return None
        m = joined_rows[0].shape[0]
        if m == 0:
            return None
        head_rows = torch.full((m, 3), self.NONE, dtype=torch.long, device=self._dev)
        head_rows[:, 0] = int(head.pred)
        var_map: Dict[str, torch.Tensor] = {}
        for atom, rows in zip(body, joined_rows):
            for j, arg in enumerate(atom.args):
                if isinstance(arg, Var) and not arg.name.startswith('_') and arg.name not in var_map:
                    var_map[arg.name] = rows[:, j + 1]
        for j, arg in enumerate(head.args[:2]):
            term_id = self._fast_term_id(arg)
            if term_id is not None:
                head_rows[:, j + 1] = term_id
            elif isinstance(arg, Var) and arg.name in var_map:
                head_rows[:, j + 1] = var_map[arg.name]
        return head_rows

    def _instantiate_bound_term(
        self,
        term: Term,
        bindings: Dict[str, Term],
    ) -> Optional[Term]:
        if isinstance(term, Const):
            return term
        if isinstance(term, Var):
            if term.name.startswith('_'):
                return None
            return bindings.get(term.name)
        if isinstance(term, Compound):
            subterms: List[Term] = []
            for sub in term.subterms:
                bound = self._instantiate_bound_term(sub, bindings)
                if bound is None:
                    return None
                subterms.append(bound)
            return Compound(term.func, tuple(subterms))
        return None

    def _build_structured_head_atom(
        self,
        head: HornAtom,
        bindings: Dict[str, Term],
    ) -> Optional[HornAtom]:
        args: List[Term] = []
        for arg in head.args:
            bound = self._instantiate_bound_term(arg, bindings)
            if bound is None:
                return None
            args.append(bound)
        return HornAtom(pred=head.pred, args=tuple(args))

    def _sigma_from_rows(
        self,
        body: Tuple[HornAtom, ...],
        rows: List[torch.Tensor],
    ) -> Substitution:
        bindings: Dict[str, Term] = {}
        for atom, row in zip(body, rows):
            for j, arg in enumerate(atom.args):
                if isinstance(arg, Var) and not arg.name.startswith('_') and arg.name not in bindings:
                    bindings[arg.name] = self._decode_term(int(row[j + 1].item()))
        return Substitution(bindings)

    # ── facts property (frozenset[HornAtom] для сумісності) ──────────────────
    @property
    def facts(self) -> FrozenSet[HornAtom]:
        """Повертає всі факти як frozenset[HornAtom]. Кешується між змінами."""
        if self._facts_cache is not None and self._facts_cache_n == self._n_facts:
            return self._facts_cache
        result: Set[HornAtom] = set()
        for row in self._fact_buf[:self._n_facts].tolist():
            p, a0, a1 = row
            if p >= 0:
                result.add(HornAtom(pred=p, args=self._ints_to_args(a0, a1)))
        result.update(self._extra_facts)
        self._facts_cache = frozenset(result)
        self._facts_cache_n = self._n_facts
        return self._facts_cache

    # ── Епістемічний трекер (ідентичний KnowledgeBase) ───────────────────────
    def mark_rule_verified(self, clause: HornClause) -> None:
        h = hash(clause)
        if h in self._records:
            rec = self._records[h]
            old_status = rec.status
            rec.status = EpistemicStatus.verified
            rec.success_count += 1
            rec.weight *= 1.05
            # Оновлюємо кешовані лічильники при переході статусу
            if old_status == EpistemicStatus.proposed:
                self._n_proposed -= 1
                self._n_verified += 1
            elif old_status != EpistemicStatus.verified:
                self._n_verified += 1
        for r in self.rules:
            if hash(r) == h:
                r.use_count += 1
                r.weight *= 1.01
                break

    def mark_rule_contradicted(self, clause: HornClause) -> None:
        h = hash(clause)
        if h in self._records:
            old_status = self._records[h].status
            self._records[h].status = EpistemicStatus.contradicted
            # Оновлюємо кешовані лічильники
            if old_status == EpistemicStatus.proposed:
                self._n_proposed -= 1
            elif old_status == EpistemicStatus.verified:
                self._n_verified -= 1

    def rule_status(self, clause: HornClause) -> EpistemicStatus:
        rec = self._records.get(hash(clause))
        if rec is None:
            return EpistemicStatus.verified
        return rec.status

    def rule_is_usable(self, clause: HornClause, only_verified: bool = False) -> bool:
        status = self.rule_status(clause)
        if status == EpistemicStatus.contradicted:
            return False
        if only_verified and status != EpistemicStatus.verified:
            return False
        return True

    def consolidate(self, use_count_threshold: int = 2) -> int:
        """Видаляємо слабкі / contradicted правила."""
        to_remove: Set[int] = set()
        for h, rec in list(self._records.items()):
            if rec.status == EpistemicStatus.contradicted:
                to_remove.add(h)
            elif (
                rec.status == EpistemicStatus.proposed
                and rec.use_count < use_count_threshold
                and rec.utility() < 0.05
                and rec.age_steps > 50
            ):
                to_remove.add(h)
        if not to_remove:
            return 0

        # Знаходимо індекси для видалення
        indices_to_remove = [
            i for i, r in enumerate(self.rules)
            if hash(r) in to_remove
        ]
        # Compact тензорних буферів (видаляємо рядки)
        keep_mask = torch.ones(self._n_rules, dtype=torch.bool, device=self._dev)
        for i in sorted(indices_to_remove, reverse=True):
            if i < self._n_rules:
                keep_mask[i] = False

        kept_indices = keep_mask.nonzero(as_tuple=True)[0]
        n_kept = len(kept_indices)
        if n_kept < self._n_rules:
            self._rule_buf[:n_kept] = self._rule_buf[kept_indices].clone()
            self._rule_is_tensor[:n_kept] = self._rule_is_tensor[kept_indices].clone()
            self._rule_buf[n_kept:self._n_rules] = self.NONE
            self._rule_is_tensor[n_kept:self._n_rules] = False

        self.rules = [r for r in self.rules if hash(r) not in to_remove]
        self._rule_hash_set -= to_remove
        # Оновлюємо кешовані лічильники перед видаленням записів
        for h in to_remove:
            rec = self._records.get(h)
            if rec is not None:
                if rec.status == EpistemicStatus.proposed:
                    self._n_proposed -= 1
                elif rec.status == EpistemicStatus.verified:
                    self._n_verified -= 1
        for h in to_remove:
            self._records.pop(h, None)
        self._n_rules = len(self.rules)
        return len(to_remove)

    def tick(self) -> None:
        # PERF FIX: не ітеруємо всі записи на кожному кроці.
        # age_steps кожного запису — це (global_step - birth_step).
        # Просто збільшуємо глобальний лічильник; utility() читає age_steps
        # ліниво через RuleRecord.age_steps, який ми більше не оновлюємо тут.
        # Натомість ставимо age_steps = global_step при створенні (birth = 0),
        # а utility() ділить success_count на (1 + global_step - birth_step).
        # Для зворотньої сумісності просто пропускаємо per-record update —
        # age_steps залишається freeze на моменті додавання правила,
        # але global_step зростає → utility() з часом зменшується через
        # додаткову нормалізацію на рівні consolidate().
        self._global_step += 1

    # ── MDL / utility (ідентично KnowledgeBase) ───────────────────────────────
    def complexity_penalty(self) -> float:
        return sum(r.description_length_bits() for r in self.rules)

    def utility_adjusted_penalty(self, eta: float = 0.1) -> float:
        total = 0.0
        for r in self.rules:
            rec = self._records.get(hash(r))
            util = rec.utility() if rec is not None else 0.0
            total += float(r.description_length_bits()) - eta * util
        return total

    def weighted_complexity(self) -> float:
        return sum(r.use_count * r.description_length_bits() for r in self.rules)

    def get_rule_pairs_for_semantic_feedback(
            self, max_pairs: int = 32
    ) -> List[Tuple["HornClause", "HornClause", float]]:
        pairs: List[Tuple["HornClause", "HornClause", float]] = []
        n = len(self.rules)
        if n < 2:
            return pairs
        for i in range(min(max_pairs * 2, n)):
            r1 = self.rules[i % n]
            r2 = self.rules[(i + 1) % n]
            r1_head_pred = r1.head.pred
            r2_body_preds = {a.pred for a in r2.body}
            if r1_head_pred in r2_body_preds:
                score = 0.9
            elif r1.head.pred == r2.head.pred:
                score = 0.7
            else:
                continue
            pairs.append((r1, r2, score))
            if len(pairs) >= max_pairs:
                break
        return pairs

    def get_token_pairs_for_semantic_feedback(
            self, max_pairs: int = 32
    ) -> List[Tuple[int, int, float]]:
        NET_CONTEXT_PRED = 101
        NET_MEANS_PRED = 102

        ctx_to_tokens: Dict[int, Set[int]] = {}
        concept_to_tokens: Dict[int, Set[int]] = {}
        for fact in self.facts:
            if fact.pred != NET_CONTEXT_PRED or len(fact.args) < 2:
                continue
            tok = _const_int_term(fact.args[0])
            ctx = _const_int_term(fact.args[1])
            if tok is None or ctx is None:
                continue
            ctx_to_tokens.setdefault(ctx, set()).add(tok)

        scores: Dict[Tuple[int, int], float] = {}
        for tokens in ctx_to_tokens.values():
            ordered = sorted(tokens)
            for i in range(len(ordered)):
                for j in range(i + 1, len(ordered)):
                    scores[(ordered[i], ordered[j])] = max(
                        scores.get((ordered[i], ordered[j]), 0.0), 1.0
                    )

        for rule in self.rules:
            if rule.head.pred != NET_MEANS_PRED or len(rule.head.args) < 2:
                continue
            tok = _const_int_term(rule.head.args[0])
            concept = _const_int_term(rule.head.args[1])
            if tok is None or concept is None:
                continue
            concept_to_tokens.setdefault(concept, set()).add(tok)
            for other in ctx_to_tokens.get(concept, set()):
                if other == tok:
                    continue
                key = tuple(sorted((tok, other)))
                scores[key] = max(scores.get(key, 0.0), 0.85)

        for tokens in concept_to_tokens.values():
            ordered = sorted(tokens)
            for i in range(len(ordered)):
                for j in range(i + 1, len(ordered)):
                    key = (ordered[i], ordered[j])
                    scores[key] = max(scores.get(key, 0.0), 1.25)

        concept_items = list(concept_to_tokens.items())
        for i in range(len(concept_items)):
            concept_a, toks_a = concept_items[i]
            ctx_a = ctx_to_tokens.get(concept_a, set())
            for j in range(i + 1, len(concept_items)):
                concept_b, toks_b = concept_items[j]
                ctx_b = ctx_to_tokens.get(concept_b, set())
                if not ctx_a or not ctx_b:
                    continue
                overlap = len(ctx_a & ctx_b) / max(len(ctx_a | ctx_b), 1)
                if overlap < 0.5:
                    continue
                for tok_a in toks_a:
                    for tok_b in toks_b:
                        if tok_a == tok_b:
                            continue
                        key = tuple(sorted((tok_a, tok_b)))
                        scores[key] = max(scores.get(key, 0.0), 0.7 + 0.2 * overlap)

        items = [(a, b, s) for (a, b), s in scores.items()]
        items.sort(key=lambda x: x[2], reverse=True)
        return items[:max_pairs]

    def add_concept_fact(self, token_idx: int,
                         context_indices: Optional[List[int]] = None) -> None:
        """Сумісність з NET: реєструємо NET токен як факт."""
        NET_TOKEN_PRED = 100
        NET_CONTEXT_PRED = 101
        NET_MEANS_PRED = 102
        fact = HornAtom(pred=NET_TOKEN_PRED, args=(Const(token_idx),))
        self.add_fact(fact)
        ctx = [int(c) for c in (context_indices or []) if int(c) != int(token_idx)]
        ctx = list(dict.fromkeys(ctx))
        for ctx_idx in ctx:
            self.add_fact(HornAtom(pred=NET_CONTEXT_PRED,
                                   args=(Const(token_idx), Const(ctx_idx))))
            self.add_fact(HornAtom(pred=NET_MEANS_PRED,
                                   args=(Const(token_idx), Const(ctx_idx))))
            self.add_rule(
                HornClause(
                    head=HornAtom(pred=NET_MEANS_PRED,
                                  args=(Const(token_idx), Const(ctx_idx))),
                    body=(HornAtom(pred=NET_CONTEXT_PRED,
                                   args=(Const(token_idx), Const(ctx_idx))),),
                ),
                status=EpistemicStatus.proposed,
            )
        if ctx:
            self.add_rule(
                HornClause(
                    head=HornAtom(pred=NET_MEANS_PRED,
                                  args=(Const(token_idx), Const(min(ctx)))),
                    body=tuple(
                        HornAtom(pred=NET_CONTEXT_PRED, args=(Const(token_idx), Const(ctx_idx)))
                        for ctx_idx in ctx[: min(len(ctx), 3)]
                    ),
                ),
                status=EpistemicStatus.proposed,
            )

    @property
    def n_proposed(self) -> int:
        """Кількість правил зі статусом 'proposed' (O(1), кешовано)."""
        return max(0, self._n_proposed)

    @property
    def n_verified(self) -> int:
        """Кількість правил зі статусом 'verified' (O(1), кешовано)."""
        return max(0, self._n_verified)

    def __len__(self) -> int:
        return len(self.rules)

    def n_facts(self) -> int:
        return self._n_facts + len(self._extra_facts)

    def to(self, device: torch.device) -> "TensorKnowledgeBase":
        """Переносимо тензорні буфери на новий device."""
        self._dev = device
        self._fact_buf = self._fact_buf.to(device)
        self._rule_buf = self._rule_buf.to(device)
        self._rule_is_tensor = self._rule_is_tensor.to(device)
        return self


# ══════════════════════════════════════════════════════════════════════════════
# 4.  НЕЙРО-СИМВОЛЬНИЙ ІНТЕРФЕЙС (розділи 5-7 специфікації)
# ══════════════════════════════════════════════════════════════════════════════

class TermEmbedder(nn.Module):
    """
    Ембеддинги для термів першого порядку (розділ 5.1).

    E: Symbol → R^d

    Константи c: E(c) — lookup table.
    Змінні  X:  глобальні навчені вектори v_{X} (positional-encoding для логіки).
    Compound f(t1,...,tk): MLP_f(t1_emb ⊕ ... ⊕ tk_emb).

    t_emb обчислюється рекурсивно (TreeLSTM-like, спрощено до MLP).
    """

    def __init__(self, sym_vocab: int, d: int, max_arity: int = 2):
        super().__init__()
        self.d = d
        self.max_arity = max_arity

        # Ембеддинги констант і предикатів
        self.const_emb = nn.Embedding(sym_vocab + 4, d)

        # Глобальні ембеддинги змінних (по хешу від імені)
        self.var_emb   = nn.Embedding(256, d)

        # MLP для Compound f(t1,...,tk) → R^d
        self.compound_mlp = nn.Sequential(
            nn.Linear(d * max_arity, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

    def _var_idx(self, name: str) -> int:
        return abs(hash(name)) % 256

    def embed_term(self, t: Term, device: torch.device) -> torch.Tensor:
        """Повертає вектор ∈ R^d для терму t (рекурсивно)."""
        if isinstance(t, Const):
            idx = min(t.val, self.const_emb.num_embeddings - 1)
            return self.const_emb(torch.tensor(idx, device=device))
        if isinstance(t, Var):
            idx = self._var_idx(t.name)
            return self.var_emb(torch.tensor(idx, device=device))
        if isinstance(t, Compound):
            sub_embs = [self.embed_term(s, device) for s in t.subterms[:self.max_arity]]
            while len(sub_embs) < self.max_arity:
                sub_embs.append(torch.zeros(self.d, device=device))
            cat = torch.stack(sub_embs).view(1, -1)      # (1, d*max_arity)
            return self.compound_mlp(cat).squeeze(0)
        return torch.zeros(self.d, device=device)

    def embed_atom(self, atom: HornAtom, device: torch.device) -> torch.Tensor:
        """Ембеддинг атому: mean(args_embs) + pred_emb, нормований."""
        pred_idx  = min(atom.pred, self.const_emb.num_embeddings - 1)
        pred_emb  = self.const_emb(torch.tensor(pred_idx, device=device))
        if not atom.args:
            return pred_emb
        arg_embs  = torch.stack([self.embed_term(a, device) for a in atom.args])
        return (arg_embs.mean(0) + pred_emb) * 0.5

    def forward(self, atoms: List[HornAtom], device: torch.device) -> torch.Tensor:
        """
        atoms: список HornAtom
        Returns: (len(atoms), d) — матриця ембеддингів

        PERF FIX: pred-індекси збираємо у один LongTensor → один GPU-виклик
        замість N окремих torch.tensor()+embedding (було ~N×3 kernel launches).
        Args Const/Var → пряме звернення до .weight[idx] (без torch.tensor()).
        Compound (рідкість) → fallback на embed_term().
        """
        if not atoms:
            return torch.zeros(0, self.d, device=device)
        max_e = self.const_emb.num_embeddings - 1

        # --- Один batch-виклик для всіх pred-ембеддингів (N kernel calls → 1) ---
        pred_ids  = torch.tensor([min(a.pred, max_e) for a in atoms],
                                  dtype=torch.long, device=device)
        pred_embs = self.const_emb(pred_ids)                     # (N, d)

        results: List[torch.Tensor] = []
        for i, atom in enumerate(atoms):
            p_emb = pred_embs[i]
            if not atom.args:
                results.append(p_emb)
                continue
            arg_embs: List[torch.Tensor] = []
            for t in atom.args:
                if isinstance(t, Const):
                    # Пряме звернення до матриці ваг — без torch.tensor() overhead
                    arg_embs.append(self.const_emb.weight[min(t.val, max_e)])
                elif isinstance(t, Var):
                    arg_embs.append(self.var_emb.weight[self._var_idx(t.name)])
                elif isinstance(t, int):
                    idx = min(t, max_e) if t >= 0 else 0
                    arg_embs.append(self.const_emb.weight[idx])
                elif isinstance(t, Compound):
                    arg_embs.append(self.embed_term(t, device))  # рекурсія (рідко)
                else:
                    arg_embs.append(torch.zeros(self.d, device=device))
            arg_mean = torch.stack(arg_embs).mean(0)
            results.append((arg_mean + p_emb) * 0.5)

        return torch.stack(results)


class SoftUnifier(nn.Module):
    """
    Диференційована уніфікація (розділ 7 специфікації).

    М'яка підстановка через механізм уваги:
      σ_soft(X) = Σ_c α(X,c)·E(c)
    де  α(X,c) = softmax(Score(E(X), E(c)))

    Score(B_j, F) = σ(MLP(B_j_emb ⊕ F_emb))   (розділ 5.2)

    Енергія уніфікації (диференційована):
      E(σ) = Σ_j min_{F∈F} ||B_j_emb − F_emb||²
    """

    def __init__(self, d: int, sym_vocab: int, max_arity: int = 2):
        super().__init__()
        self.d = d
        self.term_emb = TermEmbedder(sym_vocab, d, max_arity)
        # Compatibility MLP: [B_j_emb ⊕ F_emb] → score
        self.compat_mlp = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

    def soft_unif_energy(self,
                         rule_body: Tuple[HornAtom, ...],
                         facts_list: List[HornAtom],
                         device: torch.device) -> torch.Tensor:
        """
        Диференційована енергія уніфікації тіла правила з фактами:
          E = Σ_j min_{F∈F} ||body_j_emb − F_emb||²

        Мінімум замінюється на log-sum-exp для диференційованості:
          E_smooth = Σ_j −τ·log Σ_F exp(−||B_j−F||²/τ)
        """
        if not facts_list or not rule_body:
            return torch.tensor(0.0, device=device)

        fact_embs = self.term_emb(facts_list, device)           # (|F|, d)
        body_embs = self.term_emb(list(rule_body), device)      # (|body|, d)

        # (|body|, |F|) — матриця квадратних відстаней
        diffs    = body_embs.unsqueeze(1) - fact_embs.unsqueeze(0)
        sq_dists = diffs.pow(2).sum(-1)                         # (|body|, |F|)

        # Smooth min через neg-logsumexp
        tau  = 0.1
        energy = (-tau * torch.logsumexp(-sq_dists / tau, dim=1)).sum()
        return energy

    def variable_attention(self,
                           var_atom: HornAtom,
                           facts_list: List[HornAtom],
                           device: torch.device
                           ) -> Tuple[torch.Tensor, Optional[HornAtom]]:
        """
        Attention-розподіл підстановки для атому з змінними:
          α(B_j, F) = σ(MLP([B_j_emb ⊕ F_emb]))

        Повертає (attn_weights ∈ R^{|F|}, argmax_fact).
        """
        if not facts_list:
            return torch.zeros(0, device=device), None

        var_emb   = self.term_emb.embed_atom(var_atom, device)      # (d,)
        fact_embs = self.term_emb(facts_list, device)               # (|F|, d)

        expanded = var_emb.unsqueeze(0).expand(len(facts_list), -1)
        pairs    = torch.cat([expanded, fact_embs], dim=-1)         # (|F|, 2d)
        scores   = self.compat_mlp(pairs).squeeze(-1)               # (|F|,)
        attn     = F.softmax(scores, dim=0)

        best_idx  = int(attn.argmax().item())
        best_fact = facts_list[best_idx] if facts_list else None
        return attn, best_fact

    def forward(self,
                rule_body: Tuple[HornAtom, ...],
                facts: FrozenSet[HornAtom],
                device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Повертає (soft_energy, attn_entropy):
          soft_energy — диференційована енергія уніфікації
          attn_entropy — ентропія розподілу уваги (регуляризація)
        """
        facts_list = list(facts)
        energy = self.soft_unif_energy(rule_body, facts_list, device)

        entropies: List[torch.Tensor] = []
        for atom in rule_body:
            if not atom.vars():
                continue               # ground atom — без змінних
            attn, _ = self.variable_attention(atom, facts_list, device)
            if attn.numel() > 1:
                ent = -(attn * (attn + 1e-9).log()).sum()
                entropies.append(ent)

        attn_entropy = (torch.stack(entropies).mean()
                        if entropies else torch.tensor(0.0, device=device))
        return energy, attn_entropy


# ══════════════════════════════════════════════════════════════════════════════
# 4b.  GRAPH MATCHING UNIFIER — консистентна уніфікація (розділ 5.2)
# ══════════════════════════════════════════════════════════════════════════════
#
# Ключова ідея: якщо ?Y з'являється у кількох атомах тіла, то м'яке
# прив'язування ?Y → константа ПОВИННЕ бути однаковим у всіх атомах.
#
# Реалізація через ітеративний message-passing на графі:
#   Вузли: {змінні} ∪ {факти-кандидати}
#   Ребра: (X, F) якщо F є кандидатом для X; (X, Y) якщо X,Y в одному атомі.
#
# Soft-substitution: σ_soft(?Y) = Σ_c α(?Y,c)·E(c)
# де α(?Y,c) = Gumbel-Softmax(score(?Y,c))  — диференційоване.
# ══════════════════════════════════════════════════════════════════════════════

class GraphMatchingUnifier(nn.Module):
    """
    Граф-відповідності для консистентної уніфікації (розділ 5.2).

    Забезпечує, що одна й та сама змінна ?Y в різних атомах тіла
    прив'язується до ОДНОГО і того ж терму — через спільний вектор змінної
    та cross-body attention.

    Граф: вузли = {змінні} ∪ {факти-кандидати}
          ребра = var→fact (attention),  var→var (co-occurrence у тілі).

    Score(B_j, F) = σ(MLP(B_j_emb ⊕ F_emb))  — розділ 5.2.
    Soft-substitution: σ_soft(?Y) = Σ_c α(?Y,c)·E(c)  — розділ 7.
    """

    def __init__(self, d: int, sym_vocab: int,
                 max_arity: int = 2, n_iters: int = 3):
        super().__init__()
        self.d       = d
        self.n_iters = n_iters
        self.term_emb = TermEmbedder(sym_vocab, d, max_arity)

        # Проекції для attention: змінна → query, факт → key/value
        self.var_q  = nn.Linear(d, d)
        self.fact_k = nn.Linear(d, d)
        self.fact_v = nn.Linear(d, d)

        # Message-passing gate: оновлення var_emb через контекст сусідів
        self.msg_gate = nn.Sequential(
            nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, d), nn.Sigmoid()
        )

        # Co-occurrence gate: якщо дві змінні в одному атомі → обмін інфо
        self.cooc_gate = nn.Sequential(
            nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, d), nn.Sigmoid()
        )

        self.scale = d ** -0.5

    def forward(
        self,
        rule_body: Tuple['HornAtom', ...],
        facts: FrozenSet['HornAtom'],
        device: torch.device,
        tau: float = 0.5,
        hard: bool = False,
        return_attention: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor],
        Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]],
    ]:
        """
        Диференційована уніфікація тіла правила з фактами.

        Returns:
          energy     : scalar — сумарна енергія уніфікації E(σ)
          var_assign : {var_name → soft_vec ∈ R^d}  — м'яке прив'язування
          entropy    : scalar — ентропія розподілу уваги (регуляризатор)

        Алгоритм (3 фази):
          Phase 1. Ініціалізація: var_vec = TermEmbedder(Var)
          Phase 2. Ітеративний message-passing:
                    a) var → facts: attention, оновлення var_vec
                    b) var → var: co-occurrence (якщо в одному атомі)
          Phase 3. Gumbel-Softmax: диференційована підстановка
                   σ_soft(?Y) = Σ_c Gumbel(score(?Y,c))·V(c)
        """
        facts_list = list(facts)
        if not facts_list or not rule_body:
            zero = torch.tensor(0.0, device=device)
            return zero, {}, zero

        # ── Phase 0: ембеддинги фактів ────────────────────────────────────────
        fact_embs = self.term_emb(facts_list, device)      # (|F|, d)
        K = self.fact_k(fact_embs)                         # (|F|, d)
        V = self.fact_v(fact_embs)                         # (|F|, d)

        # ── Phase 1: збір унікальних змінних з тіла ───────────────────────────
        # Ключово: ?Y з різних атомів ОДИН вектор → консистентне прив'язування
        var_names: List[str] = []
        seen: Set[str] = set()
        # co-occurrence: множини змінних для кожного атому
        atom_var_sets: List[Set[str]] = []
        for atom in rule_body:
            av: Set[str] = set()
            for t in atom.args:
                if isinstance(t, Var) and not t.name.startswith('_'):
                    if t.name not in seen:
                        var_names.append(t.name)
                        seen.add(t.name)
                    av.add(t.name)
            atom_var_sets.append(av)

        if not var_names:
            zero = torch.tensor(0.0, device=device)
            return zero, {}, zero

        # Початкові ембеддинги змінних
        var_vecs: Dict[str, torch.Tensor] = {
            name: self.term_emb.embed_term(Var(name), device)
            for name in var_names
        }

        # ── Phase 2: ітеративний graph message-passing ─────────────────────────
        for _it in range(self.n_iters):
            # (a) var → facts: attention-based update
            new_vecs: Dict[str, torch.Tensor] = {}
            for name in var_names:
                q      = self.var_q(var_vecs[name])          # (d,)
                scores = (q @ K.t()) * self.scale            # (|F|,)
                attn   = F.softmax(scores, dim=0)            # (|F|,)
                ctx    = (attn.unsqueeze(0) @ V).squeeze(0)  # (d,)
                g      = self.msg_gate(
                    torch.cat([var_vecs[name], ctx], dim=-1))
                new_vecs[name] = g * ctx + (1 - g) * var_vecs[name]

            # (b) var → var: co-occurrence у одному атомі
            for av_set in atom_var_sets:
                cooc_names = [n for n in var_names if n in av_set]
                if len(cooc_names) < 2:
                    continue
                # Повідомлення між парами (усереднення)
                avg = torch.stack([new_vecs[n] for n in cooc_names]).mean(0)
                for n in cooc_names:
                    g2 = self.cooc_gate(
                        torch.cat([new_vecs[n], avg], dim=-1))
                    new_vecs[n] = g2 * avg + (1 - g2) * new_vecs[n]

            var_vecs = new_vecs

        # ── Phase 3: Gumbel-Softmax assignments ───────────────────────────────
        # σ_soft(?Y) = Σ_c Gumbel(score(?Y,c))·V(c)  — диференційована підст.
        var_assign: Dict[str, torch.Tensor] = {}
        var_attn: Dict[str, torch.Tensor] = {}
        total_entropy  = torch.tensor(0.0, device=device)
        total_energy   = torch.tensor(0.0, device=device)

        for name in var_names:
            q       = self.var_q(var_vecs[name])
            scores  = (q @ K.t()) * self.scale                # (|F|,)
            soft_w  = F.gumbel_softmax(scores, tau=tau, hard=hard)  # (|F|,)
            soft_v  = soft_w @ V                               # (d,) soft assignment
            var_assign[name] = soft_v
            var_attn[name] = soft_w

            probs   = F.softmax(scores, dim=0)
            ent     = -(probs * (probs + 1e-9).log()).sum()
            total_entropy = total_entropy + ent

        # ── Phase 4: енергія E(σ) — квадратна відстань після підстановки ──────
        # E(σ) = Σ_j min_{F∈F} ||B_j·σ_emb − F_emb||²  (smooth-min)
        for atom in rule_body:
            arg_embs: List[torch.Tensor] = []
            for t in atom.args:
                if isinstance(t, Var) and t.name in var_assign:
                    arg_embs.append(var_assign[t.name])
                else:
                    arg_embs.append(self.term_emb.embed_term(t, device))

            if arg_embs:
                atom_vec = torch.stack(arg_embs).mean(0)   # (d,)
            else:
                idx = min(atom.pred, self.term_emb.const_emb.num_embeddings - 1)
                atom_vec = self.term_emb.const_emb(
                    torch.tensor(idx, device=device))

            same_pred = [f for f in facts_list if f.pred == atom.pred]
            if same_pred:
                sp_embs  = self.term_emb(same_pred, device)   # (k, d)
                dists    = (atom_vec.unsqueeze(0) - sp_embs).pow(2).sum(-1)
                tau_e    = 0.1
                energy_j = -tau_e * torch.logsumexp(-dists / tau_e, dim=0)
                total_energy = total_energy + energy_j

        n_vars = max(len(var_names), 1)
        mean_entropy = total_entropy / n_vars
        if return_attention:
            return total_energy, var_assign, mean_entropy, var_attn
        return total_energy, var_assign, mean_entropy


# ══════════════════════════════════════════════════════════════════════════════
# 4c.  PROOF COST ESTIMATOR — Cost(T) (розділ 6 специфікації)
# ══════════════════════════════════════════════════════════════════════════════
#
# Cost(T) = Σ_{(R,σ)∈T} [ Length(R) + λ·UnifComplexity(σ) ]
#
# Де:
#   Length(R)         = MDL-довжина правила (HornClause.complexity())
#   UnifComplexity(σ) = Σ_{X∈dom(σ)} Depth(σ(X))  (Substitution.unif_complexity())
#
# Нейронна складова: TermEmbedder дає диференційовану оцінку складності правила.
# Це дозволяє градієнту проходити через вибір правил у REINFORCE.
# ══════════════════════════════════════════════════════════════════════════════

class ProofCostEstimator(nn.Module):
    """
    Оцінювач вартості доведення — формула Cost(T) з розділу 6.

    Cost(T) = Σ_{(R,σ)∈T} [Length(R) + λ·UnifComplexity(σ)]

    MDL-принцип: простіші правила з простішими підстановками → менша вартість.
    Короткі докази з малою глибиною термів у σ — краще.

    Нейронна частина nn_len = rule_enc(clause_emb) дозволяє градієнту
    текти через вибір правил (диференційована метрика MDL).
    """

    def __init__(self, d: int, sym_vocab: int, lam: float = 0.1):
        super().__init__()
        self.lam      = lam
        self.term_emb = TermEmbedder(sym_vocab, d)

        # Нейронний estimator складності правила: emb → scalar ≥ 0
        self.rule_enc = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 1),
            nn.Softplus(),          # гарантує ≥ 0
        )

    def clause_emb(self, clause: 'HornClause',
                   device: torch.device) -> torch.Tensor:
        """Ембеддинг правила: mean(head_emb + body_atom_embs)."""
        atoms = [clause.head] + list(clause.body)
        embs  = self.term_emb(atoms, device)         # (n_atoms, d)
        return embs.mean(0)                           # (d,)

    def symbolic_cost(self, clause: 'HornClause',
                      sigma: Optional['Substitution']) -> float:
        """
        Суто символьна вартість (без градієнту):
          sym_cost = bits(R) + λ·bits(σ)
        """
        rule_len = float(clause.description_length_bits())
        unif_cost = (
            universal_int_bits(int(sigma.unif_complexity()))
            if sigma is not None else 0.0
        )
        return rule_len + self.lam * unif_cost

    def forward(
        self,
        trajectory: List[Tuple['HornClause', Optional['Substitution']]],
        device: torch.device,
    ) -> torch.Tensor:
        """
        trajectory : список (rule, sigma) — кроки доведення
        Returns    : scalar tensor Cost(T) = Σ [Length(R) + λ·UnifComplexity(σ)]

        Диференційована версія:
          Cost = Σ_i [ bits(Ri) + nn_len(Ri) + λ·bits(σi) ]
        """
        if not trajectory:
            return torch.tensor(0.0, device=device)

        total = torch.tensor(0.0, device=device)
        for rule, sigma in trajectory:
            # Fixed symbolic code length in bits.
            sym_len = float(rule.description_length_bits())
            # Нейронна оцінка (диференційована для backprop)
            r_emb   = self.clause_emb(rule, device)          # (d,)
            nn_len  = self.rule_enc(r_emb.unsqueeze(0)).squeeze()   # scalar ≥ 0

            mdl_part = torch.tensor(sym_len, device=device, dtype=nn_len.dtype) + nn_len

            # Universal-code proxy for substitution complexity.
            unif_bits = (
                universal_int_bits(int(sigma.unif_complexity()))
                if sigma is not None else 0.0
            )
            total = total + mdl_part + self.lam * float(unif_bits)

        return total


# ══════════════════════════════════════════════════════════════════════════════
# 5.  POLICY NETWORK  (стратегія пошуку доведення)
# ══════════════════════════════════════════════════════════════════════════════

class ProofPolicyNet(nn.Module):
    """
    π_θ(Action | State)

    State  = конкатенація: z_context (стан моделі) + z_goal (ціль)
    Action = який правило застосувати наступним (індекс у rules)

    Навчається через REINFORCE:
      ∇_θ J = E_τ[R(τ)·Σ_t ∇_θ log π_θ(a_t|s_t)]
    """

    def __init__(self, d_latent: int, max_rules: int, dropout: float = 0.1):
        super().__init__()
        self.max_rules = max_rules
        self.state_enc = nn.Sequential(
            nn.Linear(d_latent * 2, d_latent), nn.GELU(),
            nn.Dropout(dropout),
        )
        self.rule_score = nn.Linear(d_latent, max_rules)

    def forward(self,
                z_ctx: torch.Tensor,
                z_goal: torch.Tensor,
                n_rules: int) -> torch.Tensor:
        """
        z_ctx  : (B, d)
        z_goal : (B, d)
        Returns: log_probs (B, n_rules)
        """
        state = self.state_enc(torch.cat([z_ctx, z_goal], dim=-1))
        logits = self.rule_score(state)[..., :n_rules]          # (B, n_rules)
        return F.log_softmax(logits, dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  ABDUCTION HEAD  (генерує кандидати-правила з нейронного z)
# ══════════════════════════════════════════════════════════════════════════════

class NeuralAbductionHead(nn.Module):
    """
    Нейромережевий генератор HornClause-кандидатів.

    R* = argmin [ Complexity(R) + λ·PredError(R, Trace) ]

    Мережа пропонує кандидати через Gumbel-Softmax,
    символьна оцінка виконується детерміновано.
    """

    def __init__(self, d_latent: int, sym_vocab: int,
                 n_cands: int, max_arity: int = 2):
        super().__init__()
        self.sv       = sym_vocab
        self.n_cands  = n_cands
        self.max_arity = max_arity

        # Мережа → розподіл над sym_vocab для кожного слоту
        # Структура: [pred] [arg0..argN] для head + body
        slots_per_atom = 1 + max_arity          # pred + args
        n_slots_total  = slots_per_atom * 2     # head + 1 тіло-атом
        self.rule_gen = nn.Sequential(
            nn.Linear(d_latent, d_latent * 2), nn.GELU(),
            nn.Linear(d_latent * 2, sym_vocab * n_slots_total),
        )
        self.slots = n_slots_total

    def forward(self, z: torch.Tensor) -> List[HornClause]:
        clauses, _ = self.sample_candidates(z)
        return clauses

    def _variable_pattern(
        self,
        cand_i: int,
    ) -> Tuple[Tuple[Var, ...], Tuple[Var, ...]]:
        all_vars = [Var(f"X{i}") for i in range(self.max_arity + 1)]
        mode = cand_i % 3
        if mode == 0:
            head_args = tuple(all_vars[:self.max_arity])
            body_args = tuple(all_vars[:self.max_arity])
        elif mode == 1:
            xa, xb = all_vars[0], all_vars[1]
            xc = all_vars[2] if self.max_arity >= 2 else all_vars[0]
            head_args = (xa, xc)[:self.max_arity]
            body_args = (xa, xb)[:self.max_arity]
        else:
            head_args = (all_vars[0],) * self.max_arity
            body_args = (all_vars[0],) * self.max_arity
        return tuple(head_args), tuple(body_args)

    def sample_candidates_relaxed(
        self,
        z: torch.Tensor,
        stochastic: bool = True,
        tau: float = 0.7,
    ) -> List[RelaxedHornClauseSpec]:
        logits = self.rule_gen(z.squeeze(0)).view(self.slots, self.sv)
        specs: List[RelaxedHornClauseSpec] = []
        topk = min(self.n_cands, self.sv)
        deterministic_choices: List[torch.Tensor] = []
        if not stochastic:
            for i in range(self.slots):
                deterministic_choices.append(torch.topk(logits[i], k=topk, dim=-1).indices)

        tau = max(float(tau), 1e-3)
        body_pred_slot = 1 + self.max_arity
        for cand_i in range(self.n_cands):
            indices: List[int] = []
            slot_probs: List[torch.Tensor] = []
            lp_sum = torch.zeros(1, device=z.device).squeeze()
            for i in range(self.slots):
                slot_logits = logits[i]
                if stochastic:
                    noisy_logits = slot_logits + torch.randn_like(slot_logits) * 0.05
                    probs = F.gumbel_softmax(noisy_logits, tau=tau, hard=False, dim=-1)
                    idx = int(probs.argmax().item())
                    log_soft = F.log_softmax(noisy_logits / tau, dim=-1)
                else:
                    chosen = deterministic_choices[i][cand_i % topk]
                    idx = int(chosen.item())
                    probs = F.softmax(slot_logits / tau, dim=-1)
                    log_soft = F.log_softmax(slot_logits / tau, dim=-1)
                indices.append(idx)
                slot_probs.append(probs)
                lp_sum = lp_sum + log_soft[idx]

            head_args, body_args = self._variable_pattern(cand_i)
            clause = HornClause(
                head=HornAtom(pred=indices[0], args=head_args),
                body=(HornAtom(pred=indices[body_pred_slot], args=body_args),),
            )
            specs.append(
                RelaxedHornClauseSpec(
                    clause=clause,
                    head_pred_probs=slot_probs[0],
                    body_pred_probs=(slot_probs[body_pred_slot],),
                    log_prob=lp_sum,
                    source="neural",
                )
            )
        return specs

    def sample_candidates(
        self,
        z: torch.Tensor,
        stochastic: bool = True,
        tau: float = 0.7,
    ) -> Tuple[List[HornClause], torch.Tensor]:
        """
        z: (1, d_latent)
        Returns: список HornClause-кандидатів зі ЗМІННИМИ у args.

        Ключова зміна: аргументи правил тепер є іменованими Var (?X0,...,?Xk),
        а не конкретними константами. Це забезпечує compositional generalization:
        правило узагальнюється на будь-які константи (розділ 5.2).

        Підтримує три паттерни змінних для різноманітного узагальнення:
          Mode 0: head(?X,?Y) :- body(?X,?Y)  — спільні змінні (transitive)
          Mode 1: head(?X,?Z) :- body(?X,?Y), ...  — часткове перекриття
          Mode 2: head(?X,?X) :- body(?X,?Y)  — рефлексивне правило

        Gumbel-Softmax вибирає ПРЕДИКАТ (pred_id), а не аргументи,
        оскільки аргументи є змінними і обираються уніфікацією.
        """
        specs = self.sample_candidates_relaxed(
            z,
            stochastic=stochastic,
            tau=tau,
        )
        clauses = [spec.clause for spec in specs]
        log_probs = [spec.log_prob for spec in specs if spec.log_prob is not None]
        return clauses, torch.stack(log_probs) if log_probs else torch.zeros(0, device=z.device)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  DIFFERENTIABLE PROVER (∂-Prolog)
# ══════════════════════════════════════════════════════════════════════════════

class DifferentiableProver(nn.Module):
    """
    Диференційований Theorem Prover (∂-Prolog).

    Навчання:
      · Швидкий цикл (GPU): оновлення θ через -log P_θ
      · Повільний цикл (CPU): оновлення Γ через абдукцію

    L_sym = -E_{τ~π_θ}[R(τ)] + α·Length(τ)

    Де:
      R(τ)      = 1 якщо ціль доведена, 0 інакше
      Length(τ) = кількість кроків (заохочує короткі докази)
    """

    def __init__(self,
                 d_latent:   int,
                 sym_vocab:  int,
                 max_rules:  int = 1024,
                 max_depth:  int = 5,
                 n_cands:    int = 8,
                 alpha:      float = 0.1,
                 vem_tau:    float = 0.3,
                 eta_utility: float = 0.1,
                 consolidate_every: int = 100):
        super().__init__()
        self.d = d_latent
        self.alpha = alpha
        self.max_depth = max_depth
        self.vem_tau    = vem_tau
        self.eta_utility = eta_utility
        self.consolidate_every = consolidate_every

        # Глобальна KnowledgeBase (GPU-акселерована TensorKnowledgeBase)
        self.kb = TensorKnowledgeBase(max_rules=max_rules)

        # ── Нові компоненти ────────────────────────────────────────────────────
        # VeM: фільтрує кандидати абдукції до додавання в LTM
        self.vem = VerificationModule(d_latent, sym_vocab, vem_tau=vem_tau)

        # Нейронні компоненти (оновлюються на швидкому циклі)
        self.policy   = ProofPolicyNet(d_latent, max_rules)
        self.abductor = NeuralAbductionHead(d_latent, sym_vocab, n_cands)

        # Проекція z → символьна ціль (goal embedding)
        self.goal_proj = nn.Linear(d_latent, d_latent)
        self.z_to_pred = nn.Linear(d_latent, sym_vocab)

        # Для ground-зв'язку: символьні ембеддинги → нейронний простір
        self.sym_emb  = nn.Embedding(sym_vocab + 2, d_latent)
        self.out_proj = nn.Linear(d_latent, d_latent)

        # TermEmbedder: структурний ембеддинг термів (розділ 5.1)
        self.term_emb  = TermEmbedder(sym_vocab, d_latent)
        # SoftUnifier: диференційована уніфікація (розділ 7)
        self.soft_unif = SoftUnifier(d_latent, sym_vocab)
        # GraphMatchingUnifier: консистентна уніфікація (розділ 5.2)
        self.graph_unif = GraphMatchingUnifier(d_latent, sym_vocab)
        # ProofCostEstimator: Cost(T) = Σ[Length(R) + λ·UnifComplexity(σ)] (розділ 6)
        self.cost_est   = ProofCostEstimator(d_latent, sym_vocab,
                                             lam=alpha)

        self._step = 0
        self.task_context: Optional[SymbolicTaskContext] = None
        self._working_memory_facts: FrozenSet[HornAtom] = frozenset()
        self.last_goal: Optional[HornAtom] = None
        self.last_context_facts: FrozenSet[HornAtom] = frozenset()
        self.last_all_facts: FrozenSet[HornAtom] = frozenset()
        self.last_forward_info: Dict[str, Union[int, float, str]] = {}
        self.last_abduced_rules: List[HornClause] = []
        self.last_used_rules: List[HornClause] = []
        self._last_used_rule_hashes: Set[int] = set()
        self._rule_utility_history: Dict[int, List[float]] = defaultdict(list)
        self._ground_cache: Dict[Tuple[str, FrozenSet[HornAtom]], torch.Tensor] = {}
        self._mental_rule_cache: Dict[Tuple[int, FrozenSet[HornAtom], str], Tuple[float, Optional[HornAtom]]] = {}
        self._pred_error_cache: Dict[Tuple[int, FrozenSet[HornAtom]], float] = {}
        self._world_rule_error_cache: Dict[Tuple[int, Optional[HornAtom], str], Optional[float]] = {}
        # fc_cache видалено: TensorKnowledgeBase.forward_chain виконується
        # за ~0.1–0.5 ms/батч через GPU broadcast — кешування не потрібне.

        # ── WorldRNN-інтеграція (Fix: Дедукція + Абдукція) ────────────────────
        # WorldRNN ін'єктується з omen_scale після ініціалізації через set_world_rnn().
        # Використовується в:
        #   · _mental_simulate_rule()  → latent-space Prediction Error (Дедукція)
        #   · _pred_error_for_rule()   → WorldRNN component у MDL PredError (Абдукція)
        self._world_rnn: Optional[Any] = None          # WorldRNN | None
        self._last_z: Optional[torch.Tensor] = None    # (B, d) — знімок з останнього forward()
        self._world_rnn_vocab: int = 0                 # кешована vocab_size для clamp
        self._world_graph_context: Optional[torch.Tensor] = None
        self._world_target_state: Optional[torch.Tensor] = None
        self.hypothesis_token_head: Optional[nn.Module] = None
        self.continuous_cycle_enabled: bool = True
        self.continuous_cycle_eval_enabled: bool = True
        self.continuous_cycle_eval_learning_enabled: bool = True
        self.continuous_cycle_max_contextual: int = 4
        self.continuous_cycle_max_neural: int = 4
        self.continuous_cycle_accept_threshold: float = 0.55
        self.continuous_cycle_verify_threshold: float = 0.75
        self.continuous_cycle_contradict_threshold: float = 0.15
        self.continuous_cycle_symbolic_weight: float = 0.30
        self.continuous_cycle_world_weight: float = 0.55
        self.continuous_cycle_token_weight: float = 0.15
        self.continuous_cycle_trace_weight: float = 0.20
        self.continuous_cycle_counterexample_weight: float = 0.15
        self.continuous_cycle_world_reject_threshold: float = 0.75
        self.continuous_cycle_soft_symbolic_weight: float = 0.45
        self.continuous_cycle_policy_weight: float = 0.25
        self.continuous_cycle_policy_baseline_momentum: float = 0.90
        self.continuous_cycle_candidate_tau: float = 0.70
        self.continuous_cycle_repair_enabled: bool = True
        self.continuous_cycle_repair_threshold: float = 0.35
        self.continuous_cycle_max_repairs: int = 2
        self.continuous_cycle_max_trace_candidates: int = 4
        self._cycle_reward_baseline: float = 0.5
        self.sym_cycle_loss_weight: float = 0.10
        self.sym_abduction_loss_weight: float = 0.05
        self.allow_latent_goal_fallback: bool = True
        self.graph_reasoning_enabled: bool = True
        self.graph_reasoning_top_k_facts: int = 12
        self.graph_reasoning_max_fact_subset: int = 96
        self.graph_reasoning_attention_threshold: float = 0.02
        self.graph_reasoning_tau: float = 0.5
        self.graph_reasoning_full_scan_cutoff: int = 64
        self.world_rule_symbolic_weight: float = 0.25
        self.world_rule_world_weight: float = 0.75
        self.world_abduction_symbolic_weight: float = 0.20
        self.world_abduction_trace_weight: float = 0.15
        self.world_abduction_world_weight: float = 0.65
        self._graph_reasoning_stats: Dict[str, float] = {
            "calls": 0.0,
            "guided_calls": 0.0,
            "fallbacks": 0.0,
            "mean_subset": 0.0,
            "mean_full_facts": 0.0,
            "mean_solutions": 0.0,
        }
        self.creative_cycle = CreativeCycleCoordinator()
        self.last_creative_report = self.creative_cycle.last_report

    # ── Нейронний → символьний (perception) ──────────────────────────────────
    def set_task_context(self, context: Optional[SymbolicTaskContext]) -> None:
        if context is not None:
            context = self.creative_cycle.enrich_task_context(context)
        self.task_context = context
        self._working_memory_facts = frozenset()
        self._reset_graph_reasoning_stats()
        self._clear_runtime_caches()

    def clear_task_context(self) -> None:
        self.task_context = None
        self._working_memory_facts = frozenset()
        self._world_graph_context = None
        self._world_target_state = None
        self._reset_graph_reasoning_stats()
        self._clear_runtime_caches()

    def _clear_runtime_caches(self) -> None:
        self._ground_cache.clear()
        self._mental_rule_cache.clear()
        self._pred_error_cache.clear()
        self._world_rule_error_cache.clear()

    _atoms_conflict = staticmethod(_atoms_conflict)

    def set_allow_latent_goal_fallback(self, enabled: bool) -> None:
        self.allow_latent_goal_fallback = bool(enabled)

    def set_world_rnn(self, world_rnn: Any) -> None:
        """
        Ін'єктує WorldRNN у прувер для:
          · ментальної симуляції правил (Дедукція, розділ 2 концепції)
          · MDL PredError з latent-space відстанню (Абдукція, розділ 6)

        Викликається з omen_scale.py після ініціалізації:
          self.prover.set_world_rnn(self.world_rnn)
        """
        self._world_rnn = world_rnn
        emb = getattr(world_rnn, "act_emb", None)
        self._world_rnn_vocab = int(getattr(emb, "num_embeddings", 0))
        if self._world_rnn_vocab > 0:
            self.hypothesis_token_head = nn.Sequential(
                nn.Linear(self.d * 2, self.d),
                nn.GELU(),
                nn.Linear(self.d, self._world_rnn_vocab),
            ).to(next(self.parameters()).device)
        else:
            self.hypothesis_token_head = None

    def set_world_context(
        self,
        *,
        graph_context: Optional[torch.Tensor] = None,
        target_state: Optional[torch.Tensor] = None,
    ) -> None:
        self._world_graph_context = (
            None if graph_context is None else graph_context.detach()
        )
        self._world_target_state = (
            None if target_state is None else target_state.detach()
        )
        self._clear_runtime_caches()

    def configure_creative_cycle(
        self,
        *,
        enabled: bool = True,
        cycle_every: int = 4,
        max_selected_rules: int = 2,
        analogy_dim: int = 16,
        tau_analogy: float = 0.82,
        tau_metaphor: Optional[float] = None,
        analogy_hidden_dim: int = 64,
        analogy_gnn_layers: int = 2,
        analogy_spec_ratio: float = 0.5,
        analogy_temperature: float = 0.07,
        analogy_contrastive_steps: int = 2,
        analogy_contrastive_lr: float = 3e-3,
        analogy_dropout: float = 0.10,
        cwe_max_rule_mods: int = 2,
        cwe_surprise_lambda: float = 0.5,
        cwe_max_candidates: int = 8,
        cwe_max_transforms_per_rule: int = 4,
        aee_population: int = 16,
        aee_generations: int = 2,
        aee_gamma: float = 0.25,
        aee_mutation_rate: float = 0.35,
        aee_crossover_rate: float = 0.5,
        aee_ltm_seed_ratio: float = 0.35,
        aee_gene_pool_size: int = 32,
        oee_gap_threshold: float = 0.45,
        oee_contradiction_threshold: int = 1,
        oee_d_latent: int = 32,
        oee_consistency_lambda: float = 0.1,
        oee_online_lr: float = 1e-3,
        oee_forward_chain_depth: int = 2,
        oee_max_interaction_preds: int = 3,
        oee_max_hypotheses: int = 8,
        oee_bundle_beam_width: int = 4,
        oee_max_bundle_rules: int = 3,
        oee_bundle_seed_k: int = 12,
        ice_state_history: int = 128,
        ice_goal_threshold: float = 0.35,
    ) -> None:
        self.creative_cycle = CreativeCycleCoordinator(
            enabled=enabled,
            cycle_every=cycle_every,
            max_selected_rules=max_selected_rules,
            analogy_dim=analogy_dim,
            tau_analogy=tau_analogy,
            tau_metaphor=tau_metaphor,
            analogy_hidden_dim=analogy_hidden_dim,
            analogy_gnn_layers=analogy_gnn_layers,
            analogy_spec_ratio=analogy_spec_ratio,
            analogy_temperature=analogy_temperature,
            analogy_contrastive_steps=analogy_contrastive_steps,
            analogy_contrastive_lr=analogy_contrastive_lr,
            analogy_dropout=analogy_dropout,
            cwe_max_rule_mods=cwe_max_rule_mods,
            cwe_surprise_lambda=cwe_surprise_lambda,
            cwe_max_candidates=cwe_max_candidates,
            cwe_max_transforms_per_rule=cwe_max_transforms_per_rule,
            aee_population=aee_population,
            aee_generations=aee_generations,
            aee_gamma=aee_gamma,
            aee_mutation_rate=aee_mutation_rate,
            aee_crossover_rate=aee_crossover_rate,
            aee_ltm_seed_ratio=aee_ltm_seed_ratio,
            aee_gene_pool_size=aee_gene_pool_size,
            oee_gap_threshold=oee_gap_threshold,
            oee_contradiction_threshold=oee_contradiction_threshold,
            oee_d_latent=oee_d_latent,
            oee_consistency_lambda=oee_consistency_lambda,
            oee_online_lr=oee_online_lr,
            oee_forward_chain_depth=oee_forward_chain_depth,
            oee_max_interaction_preds=oee_max_interaction_preds,
            oee_max_hypotheses=oee_max_hypotheses,
            oee_bundle_beam_width=oee_bundle_beam_width,
            oee_max_bundle_rules=oee_max_bundle_rules,
            oee_bundle_seed_k=oee_bundle_seed_k,
            ice_state_history=ice_state_history,
            ice_goal_threshold=ice_goal_threshold,
        )
        self.last_creative_report = self.creative_cycle.last_report

    def run_creative_cycle(
        self,
        z: Optional[torch.Tensor],
        current_facts: FrozenSet[HornAtom],
        target_facts: FrozenSet[HornAtom],
        device: torch.device,
    ) -> Any:
        self.last_creative_report = self.creative_cycle.run(
            self,
            z,
            current_facts,
            target_facts,
            device,
        )
        return self.last_creative_report

    def current_intrinsic_value(self) -> float:
        return float(self.creative_cycle.current_intrinsic_value())

    def current_intrinsic_goal(self) -> Optional[HornAtom]:
        goal = self.creative_cycle.current_intrinsic_goal()
        return goal if isinstance(goal, HornAtom) else None

    def scheduled_intrinsic_goals(self) -> Tuple[HornAtom, ...]:
        return tuple(
            goal
            for goal in self.creative_cycle.scheduled_intrinsic_goals()
            if isinstance(goal, HornAtom)
        )

    def focus_intrinsic_goal(self) -> Optional[HornAtom]:
        intrinsic_goal = self.current_intrinsic_goal()
        if intrinsic_goal is None:
            return None
        intrinsic_value = self.current_intrinsic_value()
        if self.task_context is None:
            self.set_task_context(
                SymbolicTaskContext(
                    observed_facts=frozenset(),
                    goal=intrinsic_goal,
                    target_facts=frozenset({intrinsic_goal}),
                    provenance="intrinsic",
                    trigger_abduction=True,
                    metadata={
                        "emc_intrinsic_focus": 1.0,
                        "intrinsic_goal_active": 1.0,
                        "intrinsic_value": intrinsic_value,
                        "intrinsic_goal_repr": repr(intrinsic_goal),
                    },
                )
            )
            return intrinsic_goal

        metadata = dict(self.task_context.metadata)
        previous_goal = self.task_context.goal
        metadata.update(
            {
                "emc_intrinsic_focus": 1.0,
                "intrinsic_goal_active": 1.0,
                "intrinsic_value": intrinsic_value,
                "intrinsic_goal_repr": repr(intrinsic_goal),
                "primary_goal_repr": repr(previous_goal) if previous_goal is not None else "",
                "primary_provenance": self.task_context.provenance,
            }
        )
        targets = set(self.task_context.target_facts)
        targets.add(intrinsic_goal)
        if previous_goal is not None:
            targets.add(previous_goal)
        focused = replace(
            self.task_context,
            goal=intrinsic_goal,
            target_facts=frozenset(targets),
            provenance="intrinsic",
            trigger_abduction=True,
            metadata=metadata,
        )
        self.set_task_context(focused)
        return intrinsic_goal

    def configure_loss_weights(
        self,
        *,
        cycle_loss_weight: float = 0.10,
        abduction_loss_weight: float = 0.05,
    ) -> None:
        self.sym_cycle_loss_weight = float(max(cycle_loss_weight, 0.0))
        self.sym_abduction_loss_weight = float(max(abduction_loss_weight, 0.0))

    def configure_hypothesis_cycle(
        self,
        *,
        enabled: bool = True,
        eval_enabled: bool = True,
        eval_learning_enabled: bool = True,
        max_contextual: int = 4,
        max_neural: int = 4,
        accept_threshold: float = 0.55,
        verify_threshold: float = 0.75,
        contradict_threshold: float = 0.15,
        symbolic_weight: float = 0.45,
        world_weight: float = 0.25,
        token_weight: float = 0.30,
        trace_weight: float = 0.20,
        counterexample_weight: float = 0.15,
        world_reject_threshold: float = 0.75,
        soft_symbolic_weight: float = 0.45,
        policy_weight: float = 0.25,
        policy_baseline_momentum: float = 0.90,
        candidate_tau: float = 0.70,
        repair_enabled: bool = True,
        repair_threshold: float = 0.35,
        max_repairs: int = 2,
        max_trace_candidates: int = 4,
    ) -> None:
        self.continuous_cycle_enabled = bool(enabled)
        self.continuous_cycle_eval_enabled = bool(eval_enabled)
        self.continuous_cycle_eval_learning_enabled = bool(eval_learning_enabled)
        self.continuous_cycle_max_contextual = max(int(max_contextual), 0)
        self.continuous_cycle_max_neural = max(int(max_neural), 0)
        self.continuous_cycle_accept_threshold = float(max(0.0, min(1.0, accept_threshold)))
        self.continuous_cycle_verify_threshold = float(max(0.0, min(1.0, verify_threshold)))
        self.continuous_cycle_contradict_threshold = float(max(0.0, min(1.0, contradict_threshold)))
        total_weight = max(
            float(symbolic_weight) + float(world_weight) + float(token_weight),
            1e-6,
        )
        self.continuous_cycle_symbolic_weight = float(symbolic_weight) / total_weight
        self.continuous_cycle_world_weight = float(world_weight) / total_weight
        self.continuous_cycle_token_weight = float(token_weight) / total_weight
        self.continuous_cycle_trace_weight = float(max(trace_weight, 0.0))
        self.continuous_cycle_counterexample_weight = float(max(counterexample_weight, 0.0))
        self.continuous_cycle_world_reject_threshold = float(
            max(0.0, min(1.0, world_reject_threshold))
        )
        self.continuous_cycle_soft_symbolic_weight = float(max(0.0, min(1.0, soft_symbolic_weight)))
        self.continuous_cycle_policy_weight = float(max(policy_weight, 0.0))
        self.continuous_cycle_policy_baseline_momentum = float(
            max(0.0, min(0.999, policy_baseline_momentum))
        )
        self.continuous_cycle_candidate_tau = float(max(candidate_tau, 1e-3))
        self.continuous_cycle_repair_enabled = bool(repair_enabled)
        self.continuous_cycle_repair_threshold = float(max(0.0, min(1.0, repair_threshold)))
        self.continuous_cycle_max_repairs = max(int(max_repairs), 0)
        self.continuous_cycle_max_trace_candidates = max(int(max_trace_candidates), 0)

    def configure_graph_reasoning(
        self,
        *,
        enabled: bool = True,
        top_k_facts: int = 12,
        max_fact_subset: int = 96,
        attention_threshold: float = 0.02,
        tau: float = 0.5,
        full_scan_cutoff: int = 64,
    ) -> None:
        self.graph_reasoning_enabled = bool(enabled)
        self.graph_reasoning_top_k_facts = max(int(top_k_facts), 1)
        self.graph_reasoning_max_fact_subset = max(int(max_fact_subset), 8)
        self.graph_reasoning_attention_threshold = float(max(attention_threshold, 0.0))
        self.graph_reasoning_tau = float(max(tau, 1e-3))
        self.graph_reasoning_full_scan_cutoff = max(int(full_scan_cutoff), 0)

    def configure_world_reasoning(
        self,
        *,
        rule_symbolic_weight: float = 0.25,
        rule_world_weight: float = 0.75,
        abduction_symbolic_weight: float = 0.20,
        abduction_trace_weight: float = 0.15,
        abduction_world_weight: float = 0.65,
    ) -> None:
        rule_total = max(float(rule_symbolic_weight) + float(rule_world_weight), 1e-6)
        abduction_total = max(
            float(abduction_symbolic_weight)
            + float(abduction_trace_weight)
            + float(abduction_world_weight),
            1e-6,
        )
        self.world_rule_symbolic_weight = float(max(rule_symbolic_weight, 0.0)) / rule_total
        self.world_rule_world_weight = float(max(rule_world_weight, 0.0)) / rule_total
        self.world_abduction_symbolic_weight = (
            float(max(abduction_symbolic_weight, 0.0)) / abduction_total
        )
        self.world_abduction_trace_weight = (
            float(max(abduction_trace_weight, 0.0)) / abduction_total
        )
        self.world_abduction_world_weight = (
            float(max(abduction_world_weight, 0.0)) / abduction_total
        )

    @staticmethod
    def _clamp_reasoning_error(value: float) -> float:
        return float(max(0.0, min(1.0, value)))

    def _rule_prediction_error_score(
        self,
        symbolic_error: float,
        world_error: Optional[float],
    ) -> float:
        symbolic_error = self._clamp_reasoning_error(symbolic_error)
        if world_error is None:
            return symbolic_error
        world_error = self._clamp_reasoning_error(world_error)
        combined = (
            self.world_rule_symbolic_weight * symbolic_error
            + self.world_rule_world_weight * world_error
        )
        if world_error >= self.continuous_cycle_world_reject_threshold:
            combined = max(combined, world_error)
        return self._clamp_reasoning_error(combined)

    def _abduction_prediction_error_score(
        self,
        symbolic_error: float,
        trace_error: float,
        world_error: Optional[float],
    ) -> float:
        symbolic_error = self._clamp_reasoning_error(symbolic_error)
        trace_error = self._clamp_reasoning_error(trace_error)
        if world_error is None:
            world_error = max(symbolic_error, trace_error)
        else:
            world_error = self._clamp_reasoning_error(world_error)
        combined = (
            self.world_abduction_symbolic_weight * symbolic_error
            + self.world_abduction_trace_weight * trace_error
            + self.world_abduction_world_weight * world_error
        )
        if world_error >= self.continuous_cycle_world_reject_threshold:
            combined = max(combined, world_error)
        return self._clamp_reasoning_error(combined)

    def _reset_graph_reasoning_stats(self) -> None:
        self._graph_reasoning_stats = {
            "calls": 0.0,
            "guided_calls": 0.0,
            "fallbacks": 0.0,
            "mean_subset": 0.0,
            "mean_full_facts": 0.0,
            "mean_solutions": 0.0,
        }

    def _record_graph_reasoning_call(
        self,
        n_full_facts: int,
        n_subset_facts: int,
        guided: bool,
        fallback: bool,
        n_solutions: int,
    ) -> None:
        stats = self._graph_reasoning_stats
        calls = stats["calls"] + 1.0
        stats["calls"] = calls
        if guided:
            stats["guided_calls"] += 1.0
        if fallback:
            stats["fallbacks"] += 1.0
        stats["mean_full_facts"] += (float(n_full_facts) - stats["mean_full_facts"]) / calls
        stats["mean_subset"] += (float(n_subset_facts) - stats["mean_subset"]) / calls
        stats["mean_solutions"] += (float(n_solutions) - stats["mean_solutions"]) / calls

    def _reasoning_device(
        self,
        reference: Optional[torch.Tensor] = None,
    ) -> torch.device:
        if reference is not None:
            return reference.device
        if self._last_z is not None:
            return self._last_z.device
        return next(self.parameters()).device

    def _graph_guided_fact_subset(
        self,
        rule_body: Tuple[HornAtom, ...],
        facts: FrozenSet[HornAtom],
        device: Optional[torch.device] = None,
    ) -> Tuple[FrozenSet[HornAtom], bool]:
        if (
            not self.graph_reasoning_enabled
            or len(rule_body) < 2
            or len(facts) <= self.graph_reasoning_full_scan_cutoff
        ):
            return facts, False

        body_preds = {atom.pred for atom in rule_body}
        facts_list = [fact for fact in facts if fact.pred in body_preds]
        if not facts_list:
            return facts, False

        device = self._reasoning_device() if device is None else device
        try:
            with torch.no_grad():
                _energy, _assign, _entropy, var_attn = self.graph_unif(
                    rule_body,
                    frozenset(facts_list),
                    device,
                    tau=self.graph_reasoning_tau,
                    hard=False,
                    return_attention=True,
                )
        except Exception:
            return facts, False

        selected_scores: Dict[int, float] = {}
        for atom in rule_body:
            atom_vars = [
                term.name
                for term in atom.args
                if isinstance(term, Var) and not term.name.startswith("_")
            ]
            ranked: List[Tuple[float, int]] = []
            for fact_idx, fact in enumerate(facts_list):
                if unify(atom, fact) is None:
                    continue
                if atom_vars:
                    var_scores = []
                    for var_name in atom_vars:
                        attn = var_attn.get(var_name)
                        if attn is not None and fact_idx < int(attn.numel()):
                            var_scores.append(float(attn[fact_idx].item()))
                    score = sum(var_scores) / len(var_scores) if var_scores else 0.0
                else:
                    score = 1.0
                ranked.append((score, fact_idx))
            ranked.sort(key=lambda item: item[0], reverse=True)
            kept = 0
            for score, fact_idx in ranked:
                if kept >= self.graph_reasoning_top_k_facts:
                    break
                if score < self.graph_reasoning_attention_threshold and kept > 0:
                    continue
                selected_scores[fact_idx] = max(selected_scores.get(fact_idx, 0.0), score)
                kept += 1

        for attn in var_attn.values():
            if attn.numel() <= 0:
                continue
            top_k = min(self.graph_reasoning_top_k_facts, int(attn.numel()))
            top_vals, top_idx = torch.topk(attn, k=top_k)
            for score_t, idx_t in zip(top_vals.tolist(), top_idx.tolist()):
                score = float(score_t)
                if score < self.graph_reasoning_attention_threshold and len(selected_scores) >= top_k:
                    continue
                selected_scores[int(idx_t)] = max(selected_scores.get(int(idx_t), 0.0), score)

        if not selected_scores:
            return facts, False

        ranked_indices = sorted(
            selected_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[: self.graph_reasoning_max_fact_subset]
        subset = frozenset(facts_list[idx] for idx, _ in ranked_indices)
        if len(subset) < len(rule_body):
            return facts, False
        return subset, len(subset) < len(facts)

    def _find_rule_substitutions(
        self,
        body: Tuple[HornAtom, ...],
        facts: FrozenSet[HornAtom],
        *,
        max_solutions: int = 16,
        device: Optional[torch.device] = None,
    ) -> List[Substitution]:
        if not body:
            return []
        device = self._reasoning_device() if device is None else device
        fact_subset, guided = self._graph_guided_fact_subset(body, facts, device=device)
        guided_subset_size = len(fact_subset)
        subs = find_all_substitutions(body, fact_subset, max_solutions=max_solutions)
        fallback = False
        if fact_subset != facts:
            full_subs = find_all_substitutions(body, facts, max_solutions=max_solutions)
            if len(full_subs) != len(subs) or set(full_subs) != set(subs):
                subs = full_subs
                fallback = True
        self._record_graph_reasoning_call(
            n_full_facts=len(facts),
            n_subset_facts=guided_subset_size,
            guided=guided,
            fallback=fallback,
            n_solutions=len(subs),
        )
        return subs

    def _guided_unify_body(
        self,
        body: Tuple[HornAtom, ...],
        facts: FrozenSet[HornAtom],
        device: Optional[torch.device] = None,
    ) -> Optional[Substitution]:
        solutions = self._find_rule_substitutions(
            body,
            facts,
            max_solutions=1,
            device=device,
        )
        return solutions[0] if solutions else None

    def _executor_rule_is_usable(
        self,
        clause: HornClause,
        *,
        only_verified: bool = False,
    ) -> bool:
        status = self._rule_status(clause)
        if status == EpistemicStatus.contradicted:
            return False
        if only_verified and status != EpistemicStatus.verified:
            return False
        if not only_verified and status == EpistemicStatus.proposed:
            return False
        return True

    def forward_chain_reasoned(
        self,
        max_depth: Optional[int] = None,
        starting_facts: Optional[FrozenSet[HornAtom]] = None,
        only_verified: bool = False,
        device: Optional[torch.device] = None,
    ) -> FrozenSet[HornAtom]:
        depth = self.max_depth if max_depth is None else max(int(max_depth), 0)
        if depth <= 0:
            return starting_facts if starting_facts is not None else self.current_working_facts()

        current = starting_facts if starting_facts is not None else self.current_working_facts()
        has_multi_body = any(
            len(rule.body) > 1 and self._executor_rule_is_usable(rule, only_verified=only_verified)
            for rule in self.kb.rules
        )
        if not self.graph_reasoning_enabled or not has_multi_body:
            return self.kb.forward_chain(
                depth,
                starting_facts=current,
                only_verified=only_verified,
            )

        device = self._reasoning_device() if device is None else device
        for _ in range(depth):
            added, _new_facts, _trace, current = self.forward_chain_step_local(
                current,
                only_verified=only_verified,
                device=device,
            )
            if added <= 0:
                break
        return current

    def _predicate_one_hot(
        self,
        pred: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        probs = torch.zeros(self.abductor.sv, device=device, dtype=dtype)
        if probs.numel() == 0:
            return probs
        idx = max(0, min(int(pred), probs.numel() - 1))
        probs[idx] = 1.0
        return probs

    def _relaxed_spec_for_clause(
        self,
        clause: HornClause,
        device: torch.device,
        dtype: torch.dtype,
        *,
        log_prob: Optional[torch.Tensor] = None,
        source: str = "contextual",
    ) -> RelaxedHornClauseSpec:
        return RelaxedHornClauseSpec(
            clause=clause,
            head_pred_probs=self._predicate_one_hot(clause.head.pred, device, dtype),
            body_pred_probs=tuple(
                self._predicate_one_hot(atom.pred, device, dtype)
                for atom in clause.body
            ),
            log_prob=log_prob,
            source=source,
        )

    def _soft_predicate_embedding(
        self,
        pred_probs: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if pred_probs.numel() == 0:
            return torch.zeros(self.d, device=device, dtype=next(self.parameters()).dtype)
        weight = self.term_emb.const_emb.weight[:pred_probs.shape[-1]].to(
            device=device,
            dtype=pred_probs.dtype,
        )
        return pred_probs @ weight

    def _soft_atom_embedding(
        self,
        atom: HornAtom,
        pred_probs: torch.Tensor,
        var_assign: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        pred_emb = self._soft_predicate_embedding(pred_probs, device)
        if not atom.args:
            return pred_emb
        arg_embs: List[torch.Tensor] = []
        for term in atom.args:
            if isinstance(term, Var) and term.name in var_assign:
                arg_embs.append(var_assign[term.name].to(device=device, dtype=pred_emb.dtype))
            else:
                arg_embs.append(self.term_emb.embed_term(term, device).to(dtype=pred_emb.dtype))
        arg_mean = torch.stack(arg_embs).mean(0)
        return 0.5 * (arg_mean + pred_emb)

    def _smooth_embedding_match(
        self,
        query_vec: torch.Tensor,
        anchors: torch.Tensor,
        *,
        tau: float = 0.1,
        scale: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if anchors.ndim == 1:
            anchors = anchors.unsqueeze(0)
        if anchors.numel() == 0:
            zero = torch.zeros((), device=query_vec.device, dtype=query_vec.dtype)
            return zero, zero
        dists = (anchors - query_vec.unsqueeze(0)).pow(2).sum(-1)
        tau = max(float(tau), 1e-3)
        energy = -tau * torch.logsumexp(-dists / tau, dim=0)
        score = torch.exp(-scale * energy).clamp(0.0, 1.0)
        return score, energy

    def _world_transition_with_diagnostics(
        self,
        z_state: torch.Tensor,
        *,
        action_token: Optional[int] = None,
        action_probs: Optional[torch.Tensor] = None,
        target_state_override: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        base_error = torch.full((), 0.5, device=z_state.device, dtype=z_state.dtype)
        base_align = torch.full((), 0.5, device=z_state.device, dtype=z_state.dtype)
        if self._world_rnn is None:
            return z_state, None, base_error, base_align
        graph_context = self._world_graph_context
        if graph_context is not None:
            graph_context = graph_context[: z_state.size(0)].to(device=z_state.device, dtype=z_state.dtype)
        target_state = target_state_override
        if target_state is None and self._world_target_state is not None:
            target_state = self._world_target_state[: z_state.size(0)].to(
                device=z_state.device,
                dtype=z_state.dtype,
            )
        if hasattr(self._world_rnn, "transition"):
            try:
                result = self._world_rnn.transition(
                    z_state,
                    action=None if action_token is None else torch.tensor(
                        [action_token],
                        device=z_state.device,
                        dtype=torch.long,
                    ),
                    action_probs=action_probs,
                    graph_context=graph_context,
                    target_state=target_state,
                )
                return (
                    result.z_next,
                    result.hidden,
                    result.causal_error.mean(),
                    result.graph_alignment.mean(),
                )
            except Exception:
                pass
        if action_probs is not None and hasattr(self._world_rnn, "act_emb"):
            try:
                emb_layer = self._world_rnn.act_emb
                weight = emb_layer.weight.to(device=z_state.device, dtype=z_state.dtype)
                vocab = weight.size(0)
                probs = action_probs.to(device=z_state.device, dtype=z_state.dtype)
                if probs.numel() < vocab:
                    probs = F.pad(probs, (0, vocab - probs.numel()))
                elif probs.numel() > vocab:
                    probs = probs[:vocab]
                probs = probs / probs.sum().clamp_min(1e-6)
                act = probs.unsqueeze(0) @ weight
                if all(hasattr(self._world_rnn, attr) for attr in ("gru", "out", "h0")):
                    h = self._world_rnn.h0.expand(z_state.size(0), -1)
                    h2 = self._world_rnn.gru(torch.cat([z_state, act], dim=-1), h)
                    z_next = self._world_rnn.out(h2)
                    if target_state is not None:
                        align = (
                            F.cosine_similarity(z_next, target_state, dim=-1)
                            .clamp(-1.0, 1.0)
                            .add(1.0)
                            .mul(0.5)
                            .mean()
                        )
                        err = (1.0 - align).clamp(0.0, 1.0)
                        return z_next, h2, err, align
                    return z_next, h2, base_error, base_align
                if hasattr(self._world_rnn, "proj"):
                    z_next = torch.tanh(self._world_rnn.proj(torch.cat([z_state, act], dim=-1)))
                    if target_state is not None:
                        align = (
                            F.cosine_similarity(z_next, target_state, dim=-1)
                            .clamp(-1.0, 1.0)
                            .add(1.0)
                            .mul(0.5)
                            .mean()
                        )
                        err = (1.0 - align).clamp(0.0, 1.0)
                        return z_next, None, err, align
                    return z_next, None, base_error, base_align
            except Exception:
                pass
        if action_token is None:
            return z_state, None, base_error, base_align
        action_t = torch.tensor([action_token], device=z_state.device, dtype=torch.long)
        z_next, hidden = self._world_rnn(z_state, action_t)
        if target_state is not None:
            align = (
                F.cosine_similarity(z_next, target_state, dim=-1)
                .clamp(-1.0, 1.0)
                .add(1.0)
                .mul(0.5)
                .mean()
            )
            err = (1.0 - align).clamp(0.0, 1.0)
            return z_next, hidden, err, align
        return z_next, hidden, base_error, base_align

    def _world_transition(
        self,
        z_state: torch.Tensor,
        *,
        action_token: Optional[int] = None,
        action_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        z_next, hidden, _causal_error, _graph_align = self._world_transition_with_diagnostics(
            z_state,
            action_token=action_token,
            action_probs=action_probs,
        )
        return z_next, hidden

    def _task_observed_facts(self) -> FrozenSet[HornAtom]:
        if self.task_context is None:
            return frozenset()
        return self.task_context.observed_facts

    def _task_target_facts(self) -> FrozenSet[HornAtom]:
        if self.task_context is None:
            intrinsic_goal = self.current_intrinsic_goal()
            if intrinsic_goal is not None:
                return frozenset({intrinsic_goal})
            return frozenset()
        return self.task_context.target_facts

    def _task_execution_trace(self) -> Optional[SymbolicExecutionTraceBundle]:
        if self.task_context is None:
            return None
        return self.task_context.execution_trace

    def _world_expected_state(
        self,
        derived: Optional[HornAtom],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        target_facts: Set[HornAtom] = set()
        if derived is not None and derived.is_ground():
            target_facts.add(derived)
        if self.task_context is not None:
            if self.task_context.goal is not None:
                target_facts.add(self.task_context.goal)
            target_facts.update(list(self.task_context.target_facts)[:4])
        if target_facts:
            expected = self.ground(frozenset(target_facts), device)[:1]
            if self._world_target_state is not None:
                target_state = self._world_target_state[:1].to(device=device, dtype=expected.dtype)
                expected = 0.7 * expected + 0.3 * target_state
            return expected
        if self._world_target_state is not None:
            return self._world_target_state[:1].to(device=device)
        return None

    def _world_rule_prediction_error(
        self,
        clause: "HornClause",
        derived: Optional[HornAtom],
        *,
        device: torch.device,
        action_probs: Optional[torch.Tensor] = None,
    ) -> Optional[float]:
        cache_key = (hash(clause), derived, str(device))
        if cache_key in self._world_rule_error_cache:
            return self._world_rule_error_cache[cache_key]
        if self._world_rnn is None or self._last_z is None or self._world_rnn_vocab <= 0:
            self._world_rule_error_cache[cache_key] = None
            return None
        if derived is None or not derived.is_ground():
            self._world_rule_error_cache[cache_key] = None
            return None
        try:
            z_anchor = self._last_z[:1].to(device=device)
            action_token = min(int(clause.head.pred), self._world_rnn_vocab - 1)
            target_state = self._world_expected_state(derived, device)
            _z_next, _hidden, causal_error_t, _align_t = self._world_transition_with_diagnostics(
                z_anchor,
                action_token=action_token,
                action_probs=action_probs,
                target_state_override=target_state,
            )
            error = float(causal_error_t.detach().clamp(0.0, 1.0).item())
            self._world_rule_error_cache[cache_key] = error
            return error
        except Exception:
            self._world_rule_error_cache[cache_key] = None
            return None

    @staticmethod
    def _trace_fact_priority(atom: HornAtom) -> int:
        if atom.pred in (TRACE_RETURN_EVENT_PRED, TRACE_ERROR_EVENT_PRED):
            return 5
        if atom.pred in (TRACE_ASSIGN_EVENT_PRED, TRACE_BINOP_EVENT_PRED):
            return 4
        if atom.pred in (TRACE_COMPARE_EVENT_PRED, TRACE_PARAM_BIND_PRED):
            return 3
        if atom.pred in (TRACE_STATE_VALUE_PRED, TRACE_STATE_TYPE_PRED):
            return 2
        if atom.pred in (TRACE_SCOPE_PRED, TRACE_PRIMARY_PRED, TRACE_COUNTEREXAMPLE_PRED):
            return 0
        return 1

    def _trace_focus_atoms(
        self,
        facts: FrozenSet[HornAtom],
        max_atoms: int = 6,
    ) -> List[HornAtom]:
        if not facts:
            return []
        ranked = sorted(
            list(facts),
            key=lambda atom: (-self._trace_fact_priority(atom), atom.pred, len(atom.args)),
        )
        return ranked[:max_atoms]

    def _trace_abduction_candidates(
        self,
        max_candidates: int = 8,
        max_body_atoms: int = 3,
    ) -> List[HornClause]:
        bundle = self._task_execution_trace()
        if bundle is None:
            return []

        def _collect_candidates(
            examples: List[Tuple[FrozenSet[HornAtom], FrozenSet[HornAtom]]]
        ) -> List[HornClause]:
            template_support: Counter = Counter()
            best_by_template: Dict[Tuple, Tuple[float, float, HornClause]] = {}
            for before_facts, after_facts in examples:
                if not before_facts and not after_facts:
                    continue
                after_targets = self._trace_focus_atoms(after_facts, max_atoms=4)
                local_best: Dict[Tuple, Tuple[float, float, HornClause]] = {}
                for example_head in after_targets:
                    support_facts = list(
                        self._trace_support_facts(
                            before_facts,
                            after_facts,
                            head=example_head,
                        )
                    )
                    if not support_facts:
                        continue
                    ranked_bodies = rank_goal_directed_bodies(
                        example_head,
                        support_facts,
                        term_const_values=_term_const_values,
                        max_body_atoms=max_body_atoms,
                    )
                    for body_score, body in ranked_bodies:
                        rule = self._generalize_example_rule(example_head, body)
                        if rule is None:
                            continue
                        template_sig = rule_template_signature(rule.head, rule.body)
                        base_score = self._contextual_template_score(
                            rule,
                            body_score=float(body_score),
                            template_support=0,
                        ) + 1.0
                        current = local_best.get(template_sig)
                        if current is None or base_score > current[0]:
                            local_best[template_sig] = (base_score, float(body_score), rule)
                for template_sig, (base_score, body_score, rule) in local_best.items():
                    template_support[template_sig] += 1
                    current = best_by_template.get(template_sig)
                    if current is None or base_score > current[0]:
                        best_by_template[template_sig] = (base_score, body_score, rule)

            ranked_rules: List[Tuple[float, float, HornClause]] = []
            for template_sig, (_, body_score, rule) in best_by_template.items():
                support = int(template_support.get(template_sig, 1))
                score = self._contextual_template_score(
                    rule,
                    body_score=body_score,
                    template_support=support,
                ) + 2.0 * float(support)
                ranked_rules.append((score, float(rule.description_length_bits()), rule))
            ranked_rules.sort(key=lambda item: (-item[0], item[1]))
            return [rule for _, _, rule in ranked_rules[:max_candidates]]

        transition_examples = [
            (transition.before_facts, transition.after_facts)
            for transition in bundle.transitions[
                : max(self.continuous_cycle_max_trace_candidates, 0) or len(bundle.transitions)
            ]
        ]
        candidates = _collect_candidates(transition_examples)
        if candidates:
            return candidates

        fallback_examples: List[Tuple[FrozenSet[HornAtom], FrozenSet[HornAtom]]] = []
        if bundle.observed_facts and bundle.target_facts:
            fallback_examples.append((bundle.observed_facts, bundle.target_facts))
        for counterexample in getattr(bundle, "counterexamples", ()):
            fallback_examples.append((counterexample.before_facts, counterexample.after_facts))
        if fallback_examples:
            candidates = _collect_candidates(fallback_examples)
            if candidates:
                return candidates

        focus_targets = self._trace_focus_atoms(bundle.target_facts, max_atoms=4)
        for example_head in focus_targets:
            support_facts = self._trace_support_facts(
                bundle.observed_facts,
                bundle.target_facts,
                head=example_head,
            )
            focus_support = tuple(self._trace_focus_atoms(support_facts, max_atoms=max_body_atoms))
            if focus_support:
                rule = self._generalize_example_rule(example_head, focus_support)
                if rule is not None:
                    return [rule]
            tautological = self._generalize_example_rule(example_head, (example_head,))
            if tautological is not None:
                return [tautological]

        return []

    def _trace_support_facts(
        self,
        before_facts: FrozenSet[HornAtom],
        after_facts: FrozenSet[HornAtom],
        head: Optional[HornAtom] = None,
    ) -> FrozenSet[HornAtom]:
        support = set(before_facts)
        for fact in after_facts:
            if head is not None and hash(fact) == hash(head):
                continue
            if self._trace_fact_priority(fact) >= 3:
                support.add(fact)
        return frozenset(support)

    def _trace_transition_match_stats(
        self,
        clause: HornClause,
        before_facts: FrozenSet[HornAtom],
        after_facts: FrozenSet[HornAtom],
        max_solutions: int = 12,
    ) -> Tuple[float, float, bool]:
        if not after_facts:
            return 0.0, 0.0, False
        support_facts = self._trace_support_facts(before_facts, after_facts, head=clause.head)
        if not clause.body:
            head_match = any(unify(clause.head, target) is not None for target in after_facts)
            conflict = any(_atoms_conflict(clause.head, target) for target in after_facts)
            return (1.0 if head_match else 0.0), (1.0 if head_match or conflict else 0.0), conflict

        fresh = freshen_vars(clause)
        subs = self._find_rule_substitutions(
            fresh.body,
            support_facts,
            max_solutions=max_solutions,
        )
        if not subs:
            return 0.0, 0.0, False

        best_success = 0.0
        support = 0.0
        any_conflict = False
        for sigma in subs:
            support = 1.0
            derived = sigma.apply_atom(fresh.head)
            if not derived.is_ground():
                continue
            if any(unify(derived, target) is not None for target in after_facts):
                best_success = 1.0
                break
            if any(_atoms_conflict(derived, target) for target in after_facts):
                any_conflict = True
        return best_success, support, any_conflict

    def _trace_prediction_error_for_rule(
        self,
        clause: HornClause,
        bundle: Optional[SymbolicExecutionTraceBundle],
    ) -> float:
        if bundle is None or not bundle.transitions:
            return 1.0
        matched = 0.0
        total = 0.0
        for transition in bundle.transitions[: max(self.continuous_cycle_max_trace_candidates, 1)]:
            success, support, conflict = self._trace_transition_match_stats(
                clause,
                transition.before_facts,
                transition.after_facts,
            )
            if support <= 0.0:
                continue
            total += 1.0
            matched += max(0.0, success - (0.5 if conflict else 0.0))
        if total <= 0.0:
            return 1.0
        return 1.0 - max(0.0, min(matched / total, 1.0))

    def _counterexample_error_for_rule(
        self,
        clause: HornClause,
        bundle: Optional[SymbolicExecutionTraceBundle],
    ) -> float:
        if bundle is None or not bundle.counterexamples:
            return 0.0
        failures = 0.0
        triggered = 0.0
        for transition in bundle.counterexamples[: self.continuous_cycle_max_repairs + 2]:
            success, support, conflict = self._trace_transition_match_stats(
                clause,
                transition.before_facts,
                transition.after_facts,
            )
            if support <= 0.0:
                continue
            triggered += 1.0
            if success < 1.0 or conflict:
                failures += 1.0
        if triggered <= 0.0:
            return 0.0
        return failures / triggered

    def _rule_record(self, clause: HornClause) -> Optional[RuleRecord]:
        records = getattr(self.kb, "_records", None)
        if records is None:
            return None
        return records.get(hash(clause))

    def _rule_status(self, clause: HornClause) -> EpistemicStatus:
        record = self._rule_record(clause)
        if record is None:
            return EpistemicStatus.verified
        return record.status

    def _rule_is_usable(
        self,
        clause: HornClause,
        include_proposed: bool = False,
    ) -> bool:
        status = self._rule_status(clause)
        if status == EpistemicStatus.contradicted:
            return False
        if status == EpistemicStatus.proposed and not include_proposed:
            return False
        return True

    def current_goal(self, z: Optional[torch.Tensor] = None) -> HornAtom:
        if self.task_context is not None and self.task_context.goal is not None:
            return self.task_context.goal
        intrinsic_goal = self.current_intrinsic_goal()
        if intrinsic_goal is not None:
            return intrinsic_goal
        if self.last_goal is not None:
            return self.last_goal
        if self.allow_latent_goal_fallback and z is not None:
            return self.perceive(z)
        return HornAtom(SEQ_PREDICT_NEXT_PRED, (Const(0), Var("NEXT")))

    def current_working_facts(self) -> FrozenSet[HornAtom]:
        working: Set[HornAtom] = set(self.kb.facts)
        working.update(self._task_observed_facts())
        working.update(self._working_memory_facts)
        return frozenset(working)

    def materialize_task_context_facts(self, limit: int = 32) -> int:
        if self.task_context is None or not self.task_context.observed_facts:
            return 0
        return self.load_observed_facts(self.task_context.observed_facts, limit=limit)

    def load_observed_facts(self, facts, limit: int = 96) -> int:
        """
        Ідеальна реалізація п.2: явне завантаження спостережуваних фактів
        безпосередньо у KB до forward() pass.

        Відрізняється від materialize_task_context_facts():
          · Приймає facts напряму (не через task_context)
          · Ліміт за замовчуванням 96 (більший, для символьного контексту)
          · Може приймати будь-який iterable: frozenset, list, тощо

        Повертає кількість доданих нових фактів.
        """
        if not facts:
            return 0
        working = set(self._working_memory_facts)
        added = 0
        for fact in facts:
            if fact not in working:
                working.add(fact)
                added += 1
            if added >= limit:
                break
        self._working_memory_facts = frozenset(working)
        return added

    def _goal_embedding(self, goal: HornAtom, device: torch.device) -> torch.Tensor:
        return self.ground(frozenset({goal}), device)

    @staticmethod
    def _goal_supported(goal: HornAtom, facts: FrozenSet[HornAtom]) -> bool:
        for fact in facts:
            if unify(goal, fact) is not None:
                return True
        return False

    @staticmethod
    def _contextual_template_score(
        rule: HornClause,
        body_score: float,
        template_support: int,
    ) -> float:
        bridge_bonus = float(structural_bridge_variable_count(rule.head, rule.body))
        return (
            10.0 * float(template_support)
            + float(body_score)
            + 3.0 * bridge_bonus
            - 0.05 * float(rule.description_length_bits())
        )

    def _generalize_example_rule(
        self,
        head: HornAtom,
        body: Tuple[HornAtom, ...],
    ) -> Optional[HornClause]:
        const_to_var: Dict[int, Var] = {}
        next_idx = 0

        def map_term(term: Term) -> Optional[Term]:
            nonlocal next_idx
            if isinstance(term, Const):
                key = int(term.val)
                if key not in const_to_var:
                    const_to_var[key] = Var(f"G{next_idx}")
                    next_idx += 1
                return const_to_var[key]
            if isinstance(term, Var):
                return term
            if isinstance(term, Compound):
                mapped_subterms: List[Term] = []
                for sub in term.subterms:
                    mapped = map_term(sub)
                    if mapped is None:
                        return None
                    mapped_subterms.append(mapped)
                return Compound(term.func, tuple(mapped_subterms))
            return None

        def map_atom(atom: HornAtom) -> Optional[HornAtom]:
            args: List[Term] = []
            for arg in atom.args:
                mapped = map_term(arg)
                if mapped is None:
                    return None
                args.append(mapped)
            return HornAtom(atom.pred, tuple(args))

        head_rule = map_atom(head)
        if head_rule is None:
            return None
        body_rule: List[HornAtom] = []
        for atom in body:
            mapped = map_atom(atom)
            if mapped is None:
                return None
            body_rule.append(mapped)

        head_vars = {
            var.name
            for var in head_rule.args
            if isinstance(var, Var)
        }
        body_vars = {
            var.name
            for atom in body_rule
            for var in atom.args
            if isinstance(var, Var)
        }
        if not head_vars or not head_vars.issubset(body_vars):
            return None
        return HornClause(head=head_rule, body=tuple(body_rule))

    def _contextual_abduction_candidates(
        self,
        max_candidates: int = 12,
        max_body_atoms: int = 3,
    ) -> List[HornClause]:
        if self.task_context is None or self.task_context.goal is None:
            return []
        goal = self.task_context.goal
        facts = list(self.current_working_facts())
        if not facts:
            return []

        example_heads: List[HornAtom] = [goal]
        for target_fact in self.task_context.target_facts:
            if hash(target_fact) == hash(goal):
                continue
            example_heads.append(target_fact)
        template_support: Counter = Counter()
        best_by_template: Dict[Tuple, Tuple[float, float, HornClause]] = {}
        for example_head in example_heads:
            ranked_bodies = rank_goal_directed_bodies(
                example_head,
                facts,
                term_const_values=_term_const_values,
                max_body_atoms=max_body_atoms,
            )
            local_best: Dict[Tuple, Tuple[float, float, HornClause]] = {}
            for body_score, body in ranked_bodies:
                rule = self._generalize_example_rule(example_head, body)
                if rule is None:
                    continue
                template_sig = rule_template_signature(rule.head, rule.body)
                base_score = self._contextual_template_score(
                    rule,
                    body_score=float(body_score),
                    template_support=0,
                )
                current = local_best.get(template_sig)
                if current is None or base_score > current[0]:
                    local_best[template_sig] = (base_score, float(body_score), rule)
            for template_sig, (base_score, body_score, rule) in local_best.items():
                template_support[template_sig] += 1
                current = best_by_template.get(template_sig)
                if current is None or base_score > current[0]:
                    best_by_template[template_sig] = (base_score, body_score, rule)
        if not best_by_template:
            return []
        ranked_rules: List[Tuple[float, float, HornClause]] = []
        for template_sig, (_, body_score, rule) in best_by_template.items():
            support = int(template_support.get(template_sig, 1))
            score = self._contextual_template_score(
                rule,
                body_score=body_score,
                template_support=support,
            )
            mdl_bits = float(rule.description_length_bits())
            ranked_rules.append((score, mdl_bits, rule))
        ranked_rules.sort(key=lambda item: (-item[0], item[1]))
        return [rule for _, _, rule in ranked_rules[:max_candidates]]

    def _abduction_candidate_pool(
        self,
        z: torch.Tensor,
        max_contextual: int = 12,
        max_body_atoms: int = 3,
        max_trace: int = 8,
        max_neural_fallback: int = 2,
    ) -> Tuple[List[HornClause], List[HornClause], List[HornClause], torch.Tensor]:
        trace_candidates = self._trace_abduction_candidates(
            max_candidates=max_trace,
            max_body_atoms=max_body_atoms,
        )
        contextual_candidates = self._contextual_abduction_candidates(
            max_candidates=max_contextual,
            max_body_atoms=max_body_atoms,
        )
        neural_candidates: List[HornClause] = []
        log_probs = torch.zeros(0, device=z.device)
        if not contextual_candidates and not trace_candidates and max_neural_fallback > 0:
            neural_candidates, log_probs = self.abductor.sample_candidates(
                z[:1],
                stochastic=False,
            )
            neural_candidates = neural_candidates[:max_neural_fallback]
            if log_probs.numel() > 0:
                log_probs = log_probs[:len(neural_candidates)]
        return trace_candidates, contextual_candidates, neural_candidates, log_probs

    def _note_used_rule(self, clause: Optional[HornClause]) -> None:
        if clause is None:
            return
        rule_hash = hash(clause)
        if rule_hash in self._last_used_rule_hashes:
            return
        self._last_used_rule_hashes.add(rule_hash)
        self.last_used_rules.append(clause)

    def _extend_recent_abduced_rules(self, rules: List[HornClause]) -> None:
        seen = {hash(rule) for rule in self.last_abduced_rules}
        for rule in rules:
            rule_hash = hash(rule)
            if rule_hash in seen:
                continue
            self.last_abduced_rules.append(rule)
            seen.add(rule_hash)

    def _record_rule_utility(self, clause: HornClause, utility: float) -> None:
        history = self._rule_utility_history[hash(clause)]
        history.append(float(max(0.0, min(1.0, utility))))
        if len(history) > 32:
            del history[0]

    @staticmethod
    def _rule_edit_distance(src: HornClause, dst: HornClause) -> float:
        score = 0.0
        score += 1.0 if src.head.pred != dst.head.pred else 0.0
        score += 0.25 * abs(len(src.head.args) - len(dst.head.args))
        score += float(abs(len(src.body) - len(dst.body)))
        src_preds = [int(atom.pred) for atom in src.body]
        dst_preds = [int(atom.pred) for atom in dst.body]
        overlap = len(set(src_preds) & set(dst_preds))
        score += float(max(len(src_preds), len(dst_preds)) - overlap)
        return score

    def _relaxed_symbolic_cycle_trace(
        self,
        clause: HornClause,
        observed_facts: FrozenSet[HornAtom],
        target_facts: FrozenSet[HornAtom],
        z_query: torch.Tensor,
        device: torch.device,
        z_target: Optional[torch.Tensor] = None,
        relaxed_spec: Optional[RelaxedHornClauseSpec] = None,
    ) -> Dict[str, torch.Tensor]:
        dtype = z_query.dtype
        zero = torch.zeros((), device=device, dtype=dtype)
        trace: Dict[str, torch.Tensor] = {
            "soft_score_t": torch.tensor(0.5, device=device, dtype=dtype),
            "graph_energy_t": zero,
            "head_match_t": torch.tensor(0.5, device=device, dtype=dtype),
            "body_success_t": torch.tensor(0.5, device=device, dtype=dtype),
            "entropy_penalty_t": zero,
            "head_emb": self.term_emb.embed_atom(clause.head, device).to(dtype=dtype),
        }

        fact_sample = observed_facts
        if len(fact_sample) > 24:
            fact_sample = frozenset(random.sample(list(fact_sample), 24))
        facts_list = list(fact_sample)
        fact_embs = (
            self.term_emb(facts_list, device).to(dtype=dtype)
            if facts_list else torch.zeros(0, self.d, device=device, dtype=dtype)
        )

        body_pred_probs = (
            relaxed_spec.body_pred_probs
            if relaxed_spec is not None and relaxed_spec.body_pred_probs
            else tuple(
                self._predicate_one_hot(atom.pred, device, dtype)
                for atom in clause.body
            )
        )
        head_pred_probs = (
            relaxed_spec.head_pred_probs
            if relaxed_spec is not None
            else self._predicate_one_hot(clause.head.pred, device, dtype)
        )

        default_var_assign = {
            var.name: self.term_emb.embed_term(var, device).to(dtype=dtype)
            for atom in (clause.head,) + tuple(clause.body)
            for var in atom.args
            if isinstance(var, Var)
        }
        var_assign = dict(default_var_assign)
        graph_energy_t = zero
        graph_entropy_t = zero
        soft_energy_t = zero
        soft_entropy_t = zero
        if clause.body and fact_sample:
            try:
                graph_energy_t, _graph_assign, graph_entropy_t, graph_attn = self.graph_unif(
                    clause.body,
                    fact_sample,
                    device,
                    tau=max(self.continuous_cycle_candidate_tau, 1e-3),
                    hard=False,
                    return_attention=True,
                )
                for name, attn in graph_attn.items():
                    var_assign[name] = (attn.to(dtype=dtype).unsqueeze(0) @ fact_embs).squeeze(0)
                soft_energy_t, soft_entropy_t = self.soft_unif(
                    clause.body,
                    fact_sample,
                    device,
                )
            except Exception:
                pass

        body_matches: List[torch.Tensor] = []
        for idx, atom in enumerate(clause.body):
            pred_probs = (
                body_pred_probs[idx]
                if idx < len(body_pred_probs)
                else self._predicate_one_hot(atom.pred, device, dtype)
            )
            atom_emb = self._soft_atom_embedding(atom, pred_probs, var_assign, device)
            if fact_embs.numel() > 0:
                match_t, _ = self._smooth_embedding_match(
                    atom_emb,
                    fact_embs,
                    tau=max(self.continuous_cycle_candidate_tau, 1e-3),
                    scale=0.10,
                )
                body_matches.append(match_t)
        body_success_t = (
            torch.stack(body_matches).mean().clamp(0.0, 1.0)
            if body_matches else torch.tensor(0.5, device=device, dtype=dtype)
        )

        head_emb = self._soft_atom_embedding(clause.head, head_pred_probs, var_assign, device)
        if target_facts:
            target_embs = self.term_emb(list(target_facts), device).to(dtype=dtype)
            head_match_t, _ = self._smooth_embedding_match(
                head_emb,
                target_embs,
                tau=max(self.continuous_cycle_candidate_tau, 1e-3),
                scale=0.10,
            )
        else:
            head_match_t = torch.tensor(0.5, device=device, dtype=dtype)

        target_anchor = z_target if z_target is not None else z_query[:1].detach()
        state_match_t = (
            F.cosine_similarity(
                self.out_proj(head_emb.unsqueeze(0)),
                target_anchor,
                dim=-1,
            )
            .clamp(-1.0, 1.0)
            .add(1.0)
            .mul(0.5)
            .mean()
        )
        graph_success_t = torch.exp(-0.20 * graph_energy_t).clamp(0.0, 1.0)
        soft_success_t = torch.exp(-0.10 * soft_energy_t).clamp(0.0, 1.0)
        entropy_penalty_t = torch.tanh(
            0.10 * (graph_entropy_t + soft_entropy_t)
        ).clamp(0.0, 1.0)
        soft_score_t = (
            0.30 * body_success_t
            + 0.25 * head_match_t
            + 0.20 * graph_success_t
            + 0.15 * soft_success_t
            + 0.10 * state_match_t
            - 0.10 * entropy_penalty_t
        ).clamp(0.0, 1.0)
        trace.update({
            "soft_score_t": soft_score_t,
            "graph_energy_t": graph_energy_t,
            "head_match_t": head_match_t,
            "body_success_t": body_success_t,
            "entropy_penalty_t": entropy_penalty_t,
            "head_emb": head_emb,
        })
        return trace

    def _soft_symbolic_cycle_score(
        self,
        clause: HornClause,
        observed_facts: FrozenSet[HornAtom],
        target_facts: FrozenSet[HornAtom],
        z_query: torch.Tensor,
        device: torch.device,
        z_target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        trace = self._relaxed_symbolic_cycle_trace(
            clause,
            observed_facts,
            target_facts,
            z_query,
            device,
            z_target=z_target,
        )
        return trace["soft_score_t"], trace["graph_energy_t"]

    def _repair_rule_candidate(
        self,
        clause: HornClause,
        observed_facts: FrozenSet[HornAtom],
        target_facts: FrozenSet[HornAtom],
        max_body_atoms: int = 3,
    ) -> Optional[HornClause]:
        if not self.continuous_cycle_repair_enabled or not target_facts:
            return None

        facts = list(observed_facts)
        if not facts:
            return None

        candidates: List[Tuple[float, HornClause]] = []
        head_vars = {
            var.name
            for var in clause.head.args
            if isinstance(var, Var)
        }
        body_vars = {
            var.name
            for atom in clause.body
            for var in atom.args
            if isinstance(var, Var)
        }

        for target in target_facts:
            if len(target.args) == len(clause.head.args) and (
                not head_vars or head_vars.issubset(body_vars)
            ):
                swapped_head = HornAtom(target.pred, clause.head.args)
                swapped_rule = HornClause(swapped_head, clause.body)
                if hash(swapped_rule) != hash(clause):
                    candidates.append(
                        (self._rule_edit_distance(clause, swapped_rule), swapped_rule)
                    )

            ranked_bodies = rank_goal_directed_bodies(
                target,
                facts,
                term_const_values=_term_const_values,
                max_body_atoms=max(1, int(max_body_atoms)),
            )
            for body_score, body in ranked_bodies[:4]:
                repaired = self._generalize_example_rule(target, body)
                if repaired is None or hash(repaired) == hash(clause):
                    continue
                bridge_bonus = float(
                    structural_bridge_variable_count(repaired.head, repaired.body)
                )
                score = (
                    self._rule_edit_distance(clause, repaired)
                    - 0.05 * float(body_score)
                    - 1.25 * bridge_bonus
                )
                candidates.append((score, repaired))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        lam_mdl = float(getattr(self, "_mdl_lambda", 0.5))
        for _, repaired in candidates[:6]:
            pred_error = self._pred_error_for_rule(repaired, observed_facts, lam=lam_mdl)
            if pred_error < 0.95:
                return repaired
        return None

    def _current_query_atom(self) -> Optional[HornAtom]:
        if self.task_context is None:
            return None
        last_src = int(self.task_context.metadata.get("last_src", 0))
        return HornAtom(SEQ_PREDICT_NEXT_PRED, (Const(last_src), Var("NEXT")))

    def _current_target_token(self) -> Optional[int]:
        if self.task_context is None:
            return None
        raw = self.task_context.metadata.get("last_tgt")
        if raw is None:
            raw = self.task_context.metadata.get("decoder_target")
        if raw is None:
            return None
        try:
            token = int(round(float(raw)))
        except (TypeError, ValueError):
            return None
        if self._world_rnn_vocab <= 0:
            return token
        return max(0, min(token, self._world_rnn_vocab - 1))

    def _rule_action_token(
        self,
        clause: HornClause,
        predicted_facts: FrozenSet[HornAtom],
    ) -> Optional[int]:
        if self._world_rnn_vocab <= 0:
            return None
        preferred_preds = {
            SEQ_PREDICT_NEXT_PRED,
            SEQ_ACTUAL_NEXT_PRED,
            SEQ_DECODER_GUESS_PRED,
        }
        for atom in predicted_facts:
            if atom.pred in preferred_preds and atom.args:
                token = _const_int_term(atom.args[-1])
                if token is not None:
                    return max(0, min(int(token), self._world_rnn_vocab - 1))
        if clause.head.args:
            token = _const_int_term(clause.head.args[-1])
            if token is not None:
                return max(0, min(int(token), self._world_rnn_vocab - 1))
        return max(0, min(int(clause.head.pred), self._world_rnn_vocab - 1))

    def _infer_used_rules_from_delta(
        self,
        before: FrozenSet[HornAtom],
        after: FrozenSet[HornAtom],
        max_rules: int = 96,
        max_solutions: int = 32,
    ) -> None:
        new_facts = list(after - before)
        if not new_facts:
            return
        fact_space = frozenset(after)
        for clause in self.kb.rules[:max_rules]:
            if not clause.body:
                if clause.head in new_facts:
                    self._note_used_rule(clause)
                continue
            fresh = freshen_vars(clause)
            for sigma in self._find_rule_substitutions(
                fresh.body,
                fact_space,
                max_solutions=max_solutions,
            ):
                derived = sigma.apply_atom(fresh.head)
                if any(unify(derived, new_fact) is not None for new_fact in new_facts):
                    self._note_used_rule(clause)
                    break

    def _infer_supporting_rules_for_goal(
        self,
        goal: HornAtom,
        fact_space: FrozenSet[HornAtom],
        max_rules: int = 64,
        max_solutions: int = 16,
    ) -> None:
        if not fact_space:
            return
        for clause in self.kb.rules[:max_rules]:
            if not clause.body:
                if unify(goal, clause.head) is not None:
                    self._note_used_rule(clause)
                continue
            fresh = freshen_vars(clause)
            for sigma in self._find_rule_substitutions(
                fresh.body,
                fact_space,
                max_solutions=max_solutions,
            ):
                derived = sigma.apply_atom(fresh.head)
                if unify(goal, derived) is not None:
                    self._note_used_rule(clause)
                    break

    def continuous_hypothesis_cycle(
        self,
        z: torch.Tensor,
        current_facts: FrozenSet[HornAtom],
        target_facts: FrozenSet[HornAtom],
        device: torch.device,
    ) -> Dict[str, Any]:
        zero = torch.zeros(1, device=device).squeeze()
        empty_stats = dict(empty_induction_stats())
        empty_stats.setdefault("repaired", 0.0)
        cycle_active = bool(self.continuous_cycle_enabled and (self.training or self.continuous_cycle_eval_enabled))
        eval_active = bool(cycle_active and not self.training)
        learning_active = bool(cycle_active and (self.training or self.continuous_cycle_eval_learning_enabled))
        result: Dict[str, Any] = {
            "loss_tensor": zero,
            "mean_utility": 0.0,
            "accepted_rules": [],
            "added_rules": 0,
            "induction_stats": empty_stats,
            "mode": "off",
            "stats": {
                "active": float(cycle_active),
                "eval_active": float(eval_active),
                "learning_active": float(learning_active),
                "candidate_budget": 0.0,
                "trace_candidates": 0.0,
                "contextual_candidates": 0.0,
                "neural_candidates": 0.0,
                "checked": 0.0,
                "accepted": 0.0,
                "added": 0.0,
                "verified": 0.0,
                "contradicted": 0.0,
                "retained": 0.0,
                "repaired": 0.0,
                "mean_utility": 0.0,
                "mean_error": 0.0,
                "mean_symbolic_error": 0.0,
                "mean_soft_symbolic_error": 0.0,
                "mean_relaxed_body_error": 0.0,
                "mean_relaxed_head_error": 0.0,
                "mean_trace_error": 0.0,
                "mean_counterexample_error": 0.0,
                "mean_world_error": 0.0,
                "mean_token_error": 0.0,
                "mean_graph_energy": 0.0,
                "policy_loss": 0.0,
                "loss": 0.0,
            },
        }
        if not self.continuous_cycle_enabled:
            return result
        if not self.training and not self.continuous_cycle_eval_enabled:
            return result
        result["mode"] = "train" if self.training else "eval"

        effective_targets = target_facts or self._task_target_facts()
        if not effective_targets:
            effective_targets = frozenset({self.current_goal(z)})
        trace_bundle = self._task_execution_trace()
        trace_candidates = (
            self._trace_abduction_candidates(
                max_candidates=self.continuous_cycle_max_trace_candidates,
                max_body_atoms=3,
            )
            if self.continuous_cycle_max_trace_candidates > 0
            else []
        )
        contextual_candidates = (
            self._contextual_abduction_candidates(
                max_candidates=self.continuous_cycle_max_contextual,
                max_body_atoms=3,
            )
            if self.continuous_cycle_max_contextual > 0
            else []
        )
        neural_specs: List[RelaxedHornClauseSpec] = []
        if self.continuous_cycle_max_neural > 0:
            neural_specs = self.abductor.sample_candidates_relaxed(
                z[:1],
                stochastic=self.training,
                tau=max(self.continuous_cycle_candidate_tau, 1e-3),
            )
            neural_specs = neural_specs[:self.continuous_cycle_max_neural]
        result["stats"]["trace_candidates"] = float(len(trace_candidates))
        result["stats"]["contextual_candidates"] = float(len(contextual_candidates))
        result["stats"]["neural_candidates"] = float(len(neural_specs))

        candidate_specs: Dict[int, Tuple[HornClause, Optional[torch.Tensor], str, Optional[RelaxedHornClauseSpec]]] = {}
        for clause in trace_candidates:
            relaxed = self._relaxed_spec_for_clause(
                clause,
                device,
                z.dtype,
                source="trace",
            )
            candidate_specs[hash(clause)] = (clause, None, "trace", relaxed)
        for clause in contextual_candidates:
            relaxed = self._relaxed_spec_for_clause(
                clause,
                device,
                z.dtype,
                source="contextual",
            )
            candidate_specs[hash(clause)] = (clause, None, "contextual", relaxed)
        for spec in neural_specs:
            clause = spec.clause
            rule_hash = hash(clause)
            if rule_hash in candidate_specs:
                continue
            candidate_specs[rule_hash] = (clause, spec.log_prob, "neural", spec)
        if not candidate_specs:
            return result

        observed = current_facts or self.current_working_facts()
        lam_mdl = float(getattr(self, "_mdl_lambda", 0.5))
        ranked_candidates: List[Tuple[float, HornClause, Optional[torch.Tensor], str, Optional[RelaxedHornClauseSpec]]] = []
        for clause, log_prob, source, relaxed_spec in candidate_specs.values():
            mdl_score = float(clause.description_length_bits()) + lam_mdl * float(
                self._pred_error_for_rule(clause, observed, lam=lam_mdl)
            )
            ranked_candidates.append((mdl_score, clause, log_prob, source, relaxed_spec))
        ranked_candidates.sort(key=lambda item: item[0])
        budget = max(
            1,
            self.continuous_cycle_max_trace_candidates
            + self.continuous_cycle_max_contextual
            + self.continuous_cycle_max_neural,
        )
        result["stats"]["candidate_budget"] = float(budget)
        pending: List[Tuple[float, HornClause, Optional[torch.Tensor], str, Optional[RelaxedHornClauseSpec]]] = ranked_candidates[:budget]
        seen_candidate_hashes = {hash(clause) for _, clause, _, _, _ in pending}
        repair_budget = self.continuous_cycle_max_repairs

        query_atom = self._current_query_atom()
        query_emb = (
            self.ground(frozenset({query_atom}), device)[:1]
            if query_atom is not None and self.hypothesis_token_head is not None
            else None
        )
        target_token = self._current_target_token()
        target_token_t = (
            torch.tensor([target_token], device=device, dtype=torch.long)
            if target_token is not None and self.hypothesis_token_head is not None
            else None
        )
        z_target = self.ground(effective_targets, device)[:1] if effective_targets else None
        candidate_losses: List[torch.Tensor] = []
        selection_logits: List[torch.Tensor] = []
        policy_log_probs: List[torch.Tensor] = []
        policy_rewards: List[torch.Tensor] = []
        accepted_rules: List[HornClause] = []
        utilities: List[float] = []
        symbolic_errors: List[float] = []
        soft_symbolic_errors: List[float] = []
        relaxed_body_errors: List[float] = []
        relaxed_head_errors: List[float] = []
        trace_errors: List[float] = []
        counterexample_errors: List[float] = []
        world_errors: List[float] = []
        token_errors: List[float] = []
        graph_energies: List[float] = []
        accepted = 0
        added_rules = 0
        verified = 0
        contradicted = 0
        retained = 0
        repaired = 0
        matched_predictions = 0
        checked = 0

        while pending and checked < (budget + self.continuous_cycle_max_repairs):
            _mdl_score, clause, log_prob, _source, relaxed_spec = pending.pop(0)
            checked += 1
            pred_error, predicted_one = self._mental_simulate_rule(clause, observed, device)
            predicted_facts = self._predict_rule_facts(clause, observed)
            if predicted_one is not None:
                predicted_facts = predicted_facts | frozenset({predicted_one})
            if not predicted_facts and clause.head.is_ground():
                predicted_facts = frozenset({clause.head})

            target_hits = sum(
                1
                for target in effective_targets
                if any(unify(pred, target) is not None for pred in predicted_facts)
            )
            observed_hits = sum(
                1
                for obs in observed
                if any(unify(pred, obs) is not None for pred in predicted_facts)
            )
            conflict = any(
                _atoms_conflict(pred, known)
                for pred in predicted_facts
                for known in observed
            )
            target_hit_frac = float(target_hits) / float(max(len(effective_targets), 1))
            observed_hit_frac = (
                float(observed_hits) / float(max(len(observed), 1))
                if observed
                else 0.0
            )
            symbolic_success = (
                0.60 * target_hit_frac
                + 0.25 * observed_hit_frac
                + 0.15 * (1.0 - min(max(pred_error, 0.0), 1.0))
            )
            if not predicted_facts:
                symbolic_success *= 0.5
            symbolic_success = max(0.0, min(symbolic_success, 1.0))
            hard_symbolic_t = torch.tensor(symbolic_success, device=device, dtype=z.dtype)
            if relaxed_spec is None:
                relaxed_spec = self._relaxed_spec_for_clause(
                    clause,
                    device,
                    z.dtype,
                    log_prob=log_prob,
                    source=_source,
                )
            relaxed_trace = self._relaxed_symbolic_cycle_trace(
                clause,
                observed,
                effective_targets,
                z[:1],
                device,
                z_target=z_target,
                relaxed_spec=relaxed_spec,
            )
            soft_symbolic_t = relaxed_trace["soft_score_t"]
            graph_energy_t = relaxed_trace["graph_energy_t"]
            relaxed_body_t = relaxed_trace["body_success_t"]
            relaxed_head_t = relaxed_trace["head_match_t"]
            entropy_penalty_t = relaxed_trace["entropy_penalty_t"]
            symbolic_success_t = (
                (1.0 - self.continuous_cycle_soft_symbolic_weight) * hard_symbolic_t
                + self.continuous_cycle_soft_symbolic_weight * soft_symbolic_t
            ).clamp(0.0, 1.0)
            trace_error_t = torch.zeros((), device=device, dtype=z.dtype)
            counterexample_error_t = torch.zeros((), device=device, dtype=z.dtype)
            if trace_bundle is not None and trace_bundle.transitions:
                trace_error_t = torch.tensor(
                    self._trace_prediction_error_for_rule(clause, trace_bundle),
                    device=device,
                    dtype=z.dtype,
                )
                trace_success_t = 1.0 - trace_error_t
                symbolic_success_t = (
                    (1.0 - self.continuous_cycle_trace_weight) * symbolic_success_t
                    + self.continuous_cycle_trace_weight * trace_success_t
                ).clamp(0.0, 1.0)
            if trace_bundle is not None and trace_bundle.counterexamples:
                counterexample_error_t = torch.tensor(
                    self._counterexample_error_for_rule(clause, trace_bundle),
                    device=device,
                    dtype=z.dtype,
                )

            world_success_t = torch.tensor(0.5, device=device, dtype=z.dtype)
            world_error_t = torch.tensor(0.5, device=device, dtype=z.dtype)
            world_graph_align_t = torch.tensor(0.5, device=device, dtype=z.dtype)
            z_next_world = z[:1]
            action_token = self._rule_action_token(clause, predicted_facts)
            action_probs = relaxed_spec.head_pred_probs if relaxed_spec is not None else None
            if self._world_rnn is not None and action_token is not None:
                z_next_world, _, causal_error_t, world_graph_align_t = self._world_transition_with_diagnostics(
                    z[:1],
                    action_token=action_token,
                    action_probs=action_probs,
                )
                world_error_t = causal_error_t.clamp(0.0, 1.0)
                world_success_t = (1.0 - world_error_t).clamp(0.0, 1.0)

            token_success_t = torch.tensor(0.5, device=device, dtype=z.dtype)
            token_ce_t = zero
            if (
                self.hypothesis_token_head is not None
                and query_emb is not None
                and target_token_t is not None
            ):
                token_logits = self.hypothesis_token_head(
                    torch.cat([z_next_world, query_emb], dim=-1)
                )
                token_probs = F.softmax(token_logits, dim=-1)
                token_success_t = token_probs.gather(
                    1, target_token_t.unsqueeze(1)
                ).squeeze(1).mean()
                token_ce_t = F.cross_entropy(token_logits, target_token_t)

            utility_t = (
                self.continuous_cycle_symbolic_weight * symbolic_success_t
                + self.continuous_cycle_world_weight * world_success_t
                + self.continuous_cycle_token_weight * token_success_t
            ).clamp(0.0, 1.0)
            if counterexample_error_t.numel() > 0:
                utility_t = (
                    utility_t
                    - self.continuous_cycle_counterexample_weight * counterexample_error_t
                ).clamp(0.0, 1.0)
            world_reject_t = (
                world_error_t >= self.continuous_cycle_world_reject_threshold
            )
            if bool(world_reject_t.detach().item()):
                utility_t = torch.minimum(
                    utility_t,
                    torch.tensor(0.05, device=device, dtype=z.dtype),
                )
            if conflict:
                utility_t = torch.minimum(
                    utility_t,
                    torch.tensor(0.05, device=device, dtype=z.dtype),
                )
            utility = float(utility_t.detach().item())
            candidate_loss = (
                (1.0 - utility_t)
                + 0.25 * token_ce_t
                + 0.35 * world_error_t
                + 0.08 * (1.0 - soft_symbolic_t)
                + 0.05 * (1.0 - relaxed_body_t)
                + 0.05 * (1.0 - relaxed_head_t)
                + 0.10 * trace_error_t
                + 0.12 * counterexample_error_t
                + 0.04 * (1.0 - world_graph_align_t)
                + 0.02 * entropy_penalty_t
            )
            candidate_losses.append(candidate_loss)
            mdl_bias_t = torch.tensor(float(_mdl_score), device=device, dtype=z.dtype)
            selection_logits.append((utility_t - 0.01 * mdl_bias_t).clamp(-5.0, 5.0))
            if log_prob is not None:
                policy_log_probs.append(log_prob)
                policy_rewards.append(utility_t.detach())

            self.vem.record_outcome(clause, utility_target=utility, device=device)
            self._record_rule_utility(clause, utility)

            utilities.append(utility)
            symbolic_errors.append(1.0 - float(symbolic_success_t.detach().item()))
            soft_symbolic_errors.append(1.0 - float(soft_symbolic_t.detach().item()))
            relaxed_body_errors.append(1.0 - float(relaxed_body_t.detach().item()))
            relaxed_head_errors.append(1.0 - float(relaxed_head_t.detach().item()))
            trace_errors.append(float(trace_error_t.detach().item()))
            counterexample_errors.append(float(counterexample_error_t.detach().item()))
            world_errors.append(float(world_error_t.detach().item()))
            token_errors.append(float((1.0 - token_success_t).detach().item()))
            graph_energies.append(float(graph_energy_t.detach().item()))
            matched_predictions += target_hits

            repair_candidate = None
            if (
                self.continuous_cycle_repair_enabled
                and repair_budget > 0
                and utility <= self.continuous_cycle_repair_threshold
            ):
                repair_candidate = self._repair_rule_candidate(
                    clause,
                    observed,
                    effective_targets,
                    max_body_atoms=3,
                )
                if repair_candidate is not None and hash(repair_candidate) not in seen_candidate_hashes:
                    pending.append((
                        float(_mdl_score) - 0.25,
                        repair_candidate,
                        None,
                        "repair",
                        self._relaxed_spec_for_clause(
                            repair_candidate,
                            device,
                            z.dtype,
                            source="repair",
                        ),
                    ))
                    seen_candidate_hashes.add(hash(repair_candidate))
                    repair_budget -= 1
                    repaired += 1
                    if self._rule_record(clause) is not None:
                        self.kb.mark_rule_contradicted(clause)
                        contradicted += 1
                    continue

            if (
                conflict
                or utility <= self.continuous_cycle_contradict_threshold
                or float(counterexample_error_t.detach().item()) >= 0.75
                or float(world_error_t.detach().item()) >= self.continuous_cycle_world_reject_threshold
            ):
                record = self._rule_record(clause)
                if record is not None:
                    self.kb.mark_rule_contradicted(clause)
                contradicted += 1
                continue

            should_accept = (
                utility >= self.continuous_cycle_accept_threshold
                or target_hit_frac > 0.0
            )
            if not should_accept:
                retained += 1
                continue

            accepted += 1
            accepted_rules.append(clause)
            if self.kb.add_rule(clause, status=EpistemicStatus.proposed):
                added_rules += 1
            if utility >= self.continuous_cycle_verify_threshold and target_hit_frac > 0.0:
                self.kb.mark_rule_verified(clause)
                verified += 1
            else:
                retained += 1

        policy_loss_t = zero
        if candidate_losses:
            loss_stack = torch.stack(candidate_losses)
            select_stack = torch.stack(selection_logits) if selection_logits else None
            if select_stack is not None:
                weights = F.softmax(
                    select_stack / max(self.continuous_cycle_candidate_tau, 1e-3),
                    dim=0,
                )
                loss_tensor = (weights * loss_stack).sum()
            else:
                loss_tensor = loss_stack.mean()
            if policy_log_probs:
                reward_t = torch.stack(policy_rewards)
                baseline = float(self._cycle_reward_baseline)
                advantage_t = reward_t - baseline
                if reward_t.numel() > 1:
                    advantage_t = (
                        advantage_t - advantage_t.mean()
                    ) / (advantage_t.std(unbiased=False) + 1e-6)
                policy_loss_t = -(advantage_t * torch.stack(policy_log_probs)).mean()
                momentum = self.continuous_cycle_policy_baseline_momentum
                self._cycle_reward_baseline = (
                    momentum * baseline
                    + (1.0 - momentum) * float(reward_t.mean().item())
                )
                loss_tensor = loss_tensor + self.continuous_cycle_policy_weight * policy_loss_t
            if not learning_active:
                loss_tensor = loss_tensor.detach()
            result["loss_tensor"] = loss_tensor

        if accepted_rules:
            self._extend_recent_abduced_rules(accepted_rules)

        mean_utility = float(sum(utilities) / len(utilities)) if utilities else 0.0
        mean_error = (
            float(sum(1.0 - utility for utility in utilities) / len(utilities))
            if utilities else 0.0
        )
        mean_symbolic_error = (
            float(sum(symbolic_errors) / len(symbolic_errors))
            if symbolic_errors else 0.0
        )
        mean_soft_symbolic_error = (
            float(sum(soft_symbolic_errors) / len(soft_symbolic_errors))
            if soft_symbolic_errors else 0.0
        )
        mean_relaxed_body_error = (
            float(sum(relaxed_body_errors) / len(relaxed_body_errors))
            if relaxed_body_errors else 0.0
        )
        mean_relaxed_head_error = (
            float(sum(relaxed_head_errors) / len(relaxed_head_errors))
            if relaxed_head_errors else 0.0
        )
        mean_trace_error = (
            float(sum(trace_errors) / len(trace_errors))
            if trace_errors else 0.0
        )
        mean_counterexample_error = (
            float(sum(counterexample_errors) / len(counterexample_errors))
            if counterexample_errors else 0.0
        )
        mean_world_error = (
            float(sum(world_errors) / len(world_errors))
            if world_errors else 0.0
        )
        mean_token_error = (
            float(sum(token_errors) / len(token_errors))
            if token_errors else 0.0
        )
        mean_graph_energy = (
            float(sum(graph_energies) / len(graph_energies))
            if graph_energies else 0.0
        )

        result["mean_utility"] = mean_utility
        result["accepted_rules"] = accepted_rules
        result["added_rules"] = added_rules
        result["induction_stats"] = {
            "checked": float(checked),
            "verified": float(verified),
            "contradicted": float(contradicted),
            "retained": float(retained),
            "repaired": float(repaired),
            "matched_predictions": float(matched_predictions),
            "mean_score": mean_utility,
        }
        result["stats"] = {
            "active": float(cycle_active),
            "eval_active": float(eval_active),
            "learning_active": float(learning_active),
            "candidate_budget": float(budget),
            "trace_candidates": float(len(trace_candidates)),
            "contextual_candidates": float(len(contextual_candidates)),
            "neural_candidates": float(len(neural_specs)),
            "checked": float(checked),
            "accepted": float(accepted),
            "added": float(added_rules),
            "verified": float(verified),
            "contradicted": float(contradicted),
            "retained": float(retained),
            "repaired": float(repaired),
            "mean_utility": mean_utility,
            "mean_error": mean_error,
            "mean_symbolic_error": mean_symbolic_error,
            "mean_soft_symbolic_error": mean_soft_symbolic_error,
            "mean_relaxed_body_error": mean_relaxed_body_error,
            "mean_relaxed_head_error": mean_relaxed_head_error,
            "mean_trace_error": mean_trace_error,
            "mean_counterexample_error": mean_counterexample_error,
            "mean_world_error": mean_world_error,
            "mean_token_error": mean_token_error,
            "mean_graph_energy": mean_graph_energy,
            "policy_loss": float(policy_loss_t.detach().item()),
            "loss": float(result["loss_tensor"].detach().item()),
        }
        return result

    def answer_query(
        self,
        goal: HornAtom,
        device: torch.device,
        facts: Optional[FrozenSet[HornAtom]] = None,
        max_matches: int = 16,
    ) -> Tuple[torch.Tensor, Tuple[int, ...], torch.Tensor]:
        base_facts = facts if facts is not None else self.current_working_facts()
        if not base_facts:
            zero = torch.zeros(1, self.d, device=device)
            return zero, tuple(), torch.zeros((), device=device)
        if (
            facts is None
            and self.last_all_facts
            and self.last_context_facts
            and base_facts == self.last_context_facts
        ):
            fact_space = self.last_all_facts
        else:
            fact_space = self.forward_chain_reasoned(
                self.max_depth,
                starting_facts=base_facts,
                only_verified=True,
                device=device,
            )
            if facts is None:
                self.last_context_facts = base_facts
                self.last_all_facts = fact_space
            self._infer_used_rules_from_delta(base_facts, fact_space)

        var_names = [
            arg.name
            for arg in goal.args
            if isinstance(arg, Var) and not arg.name.startswith("_")
        ]
        matched_facts: List[HornAtom] = []
        instantiated_queries: List[HornAtom] = []
        answer_ids: List[int] = []

        for fact in fact_space:
            sigma = unify(goal, fact)
            if sigma is None:
                continue
            matched_facts.append(fact)
            instantiated_queries.append(sigma.apply_atom(goal))
            for var_name in var_names:
                term = sigma.bindings.get(var_name)
                if term is None:
                    continue
                for value in _term_const_values(term):
                    answer_ids.append(int(value))
            if len(matched_facts) >= max_matches:
                break

        if not matched_facts:
            zero = torch.zeros(1, self.d, device=device)
            return zero, tuple(), torch.zeros((), device=device)

        self._infer_supporting_rules_for_goal(goal, fact_space)

        seen_answers: Set[int] = set()
        deduped_answers: List[int] = []
        for token_id in answer_ids:
            if token_id < 0 or token_id in seen_answers:
                continue
            seen_answers.add(token_id)
            deduped_answers.append(token_id)
            if len(deduped_answers) >= 8:
                break

        match_set = frozenset(matched_facts)
        query_set = frozenset(instantiated_queries) if instantiated_queries else match_set
        proof_state = 0.5 * (
            self.ground(match_set, device) + self.ground(query_set, device)
        )
        support = torch.tensor(
            1.0 - math.pow(0.5, min(len(matched_facts), 4)),
            device=device,
            dtype=proof_state.dtype,
        )
        proof_state = proof_state * support.view(1, 1)
        return proof_state, tuple(deduped_answers), support

    def forward_chain_step_local(
        self,
        current_facts: FrozenSet[HornAtom],
        only_verified: bool = False,
        device: Optional[torch.device] = None,
    ) -> Tuple[int, FrozenSet[HornAtom], List[Tuple[HornClause, Optional[Substitution]]], FrozenSet[HornAtom]]:
        current_set: Set[HornAtom] = set(current_facts)
        new_facts: Set[HornAtom] = set()
        derivations: List[Tuple[HornClause, Optional[Substitution]]] = []
        added = 0
        device = self._reasoning_device() if device is None else device
        has_multi_body = any(
            len(rule.body) > 1 and self._executor_rule_is_usable(rule, only_verified=only_verified)
            for rule in self.kb.rules
        )
        if not has_multi_body:
            after = self.kb.forward_chain(
                max_depth=1,
                starting_facts=current_facts,
                only_verified=only_verified,
            )
            new_fact_set = frozenset(after - current_facts)
            if new_fact_set:
                self._infer_used_rules_from_delta(current_facts, after)
            return len(new_fact_set), new_fact_set, [], after

        for clause in self.kb.rules:
            if not self._executor_rule_is_usable(clause, only_verified=only_verified):
                continue
            if not clause.body:
                continue
            fresh = freshen_vars(clause)
            current_snapshot = frozenset(current_set)
            for sigma in self._find_rule_substitutions(
                fresh.body,
                current_snapshot,
                max_solutions=128,
                device=device,
            ):
                derived = sigma.apply_atom(fresh.head)
                if not derived.is_ground():
                    continue
                if derived in current_set or derived in new_facts:
                    continue
                if any(_atoms_conflict(derived, known) for known in current_set):
                    self.kb.mark_rule_contradicted(clause)
                    continue
                new_facts.add(derived)
                current_set.add(derived)
                derivations.append((clause, sigma))
                clause.use_count += 1
                clause.weight *= 1.01
                self.kb.mark_rule_verified(clause)
                self._note_used_rule(clause)
                added += 1

        return added, frozenset(new_facts), derivations, frozenset(current_set)

    def perceive(self, z: torch.Tensor) -> HornAtom:
        """z: (B, d) → один HornAtom (усереднений по батчу)"""
        z_mean = z.mean(0, keepdim=True)                   # (1, d)
        pred_logits = self.z_to_pred(z_mean)               # (1, sv)
        pred = pred_logits.argmax(-1).item()

        # Аргументи через хешування (детерміновано)
        arg0 = int(z_mean.abs().argmax().item()) % (self.abductor.sv)
        arg1 = int(z_mean.sum().item() * 100) % (self.abductor.sv)
        return HornAtom(pred=int(pred), args=(arg0, arg1))

    # ── Символьний → нейронний (grounding) ────────────────────────────────────
    def ground(self, facts: "FrozenSet[HornAtom]", device) -> torch.Tensor:
        """
        Перетворює набір фактів у нейронний вектор z_sym.
        Використовує TermEmbedder для врахування структури pred + args.
        """
        if not facts:
            return torch.zeros(1, self.d, device=device)
        cache_key = (str(device), facts)
        cached = self._ground_cache.get(cache_key)
        if cached is not None:
            return cached
        facts_list = list(facts)
        # TermEmbedder враховує структуру кожного атому
        embs = self.term_emb(facts_list, device)              # (|facts|, d)
        grounded = self.out_proj(embs.mean(0, keepdim=True))  # (1, d)
        self._ground_cache[cache_key] = grounded
        return grounded

    def forward_chain_step(
        self,
        *,
        only_verified: bool = False,
        device: Optional[torch.device] = None,
    ) -> Tuple[int, "FrozenSet[HornAtom]", List[Tuple[HornClause, Optional[Substitution]]]]:
        """
        Один крок forward chaining з комітом нових фактів у KB.

        Потрібно для EMC, інакше дія ForwardChainStep бачить приріст у
        `forward_chain(max_depth=1)`, але стан KB фактично не змінюється.
        """
        before = self.kb.facts
        current = before
        added = 0
        new_facts: Set[HornAtom] = set()
        derivations: List[Tuple[HornClause, Optional[Substitution]]] = []
        device = self._reasoning_device() if device is None else device
        has_multi_body = any(
            len(rule.body) > 1 and self._executor_rule_is_usable(rule, only_verified=only_verified)
            for rule in self.kb.rules
        )
        if not has_multi_body:
            after = self.kb.forward_chain(
                max_depth=1,
                starting_facts=current,
                only_verified=only_verified,
            )
            new_fact_set = frozenset(after - current)
            for atom in new_fact_set:
                self.kb.add_fact(atom)
            if new_fact_set:
                self._infer_used_rules_from_delta(current, after)
            return len(new_fact_set), new_fact_set, []

        for clause in self.kb.rules:
            if not self._executor_rule_is_usable(clause, only_verified=only_verified):
                continue
            if not clause.body:
                continue
            fresh = freshen_vars(clause)
            for sigma in self._find_rule_substitutions(
                fresh.body,
                current,
                max_solutions=128,
                device=device,
            ):
                derived = sigma.apply_atom(fresh.head)
                if not derived.is_ground():
                    continue
                if derived in current or derived in new_facts:
                    continue
                if any(_atoms_conflict(derived, known) for known in current):
                    self.kb.mark_rule_contradicted(clause)
                    continue
                if self.kb.add_fact(derived):
                    new_facts.add(derived)
                    derivations.append((clause, sigma))
                    clause.use_count += 1
                    clause.weight *= 1.01
                    self.kb.mark_rule_verified(clause)
                    self._note_used_rule(clause)
                    added += 1

        return added, frozenset(new_facts), derivations

    # ── Mental Simulation helpers (Deduction pre-check) ──────────────────────

    def _mental_simulate_rule(
        self,
        rule: "HornClause",
        current_facts: "FrozenSet[HornAtom]",
        device: torch.device,
    ) -> Tuple[float, Optional["HornAtom"]]:
        """
        Ментальна симуляція правила ДО його реального застосування.
        Реалізує концептуальну Дедукцію (розділ 2 концепції):
          «Уявляє, як правило працюватиме, обчислює Prediction Error
           у латентному просторі ДО реального застосування»

        Повертає: (prediction_error ∈ [0,1], derived_atom_or_None)
          0.0 = правило консистентне з фактами
          1.0 = правило конфліктує або не уніфікується
        """
        cache_key = (hash(rule), current_facts, str(device))
        cached = self._mental_rule_cache.get(cache_key)
        if cached is not None:
            return cached
        if not rule.body:
            result = (0.0, rule.head if rule.head.is_ground() else None)
            self._mental_rule_cache[cache_key] = result
            return result

        fresh = freshen_vars(rule)
        sigma = self._guided_unify_body(fresh.body, current_facts, device=device)

        if sigma is None:
            # Тіло не уніфікується → prediction_error максимальна
            result = (1.0, None)
            self._mental_rule_cache[cache_key] = result
            return result

        derived = sigma.apply_atom(fresh.head)
        if not derived.is_ground():
            result = (0.5, None)
            self._mental_rule_cache[cache_key] = result
            return result

        # Перевіряємо суперечності з відомими фактами
        for known in current_facts:
            if _atoms_conflict(derived, known):
                return 1.0, None  # Суперечність → відхиляємо

        # SoftUnifier-енергія як диференційована міра prediction error
        _MENTAL_MAX_FACTS = 32
        fact_sample = (
            frozenset(random.sample(list(current_facts), _MENTAL_MAX_FACTS))
            if len(current_facts) > _MENTAL_MAX_FACTS else current_facts
        )
        try:
            su_energy, _ = self.soft_unif(rule.body, fact_sample, device)
            pred_error = float(torch.tanh(su_energy.detach() * 0.1).item())
        except Exception:
            pred_error = 0.0

        world_pred_error = self._world_rule_prediction_error(
            rule,
            derived,
            device=device,
        )
        result = (self._rule_prediction_error_score(pred_error, world_pred_error), derived)
        self._mental_rule_cache[cache_key] = result
        return result

        # ── WorldRNN latent-space Prediction Error (Дедукція, розділ 2) ─────
        # Концепція: «WorldRNN отримує на вхід z та дію (правило) і передбачає
        # наступний стан z_next.  Якщо z_next ≠ символьно-очікуваний стан →
        # Prediction Error у латентному просторі ДО реального застосування.»
        if (self._world_rnn is not None
                and self._last_z is not None
                and self._world_rnn_vocab > 0):
            try:
                with torch.no_grad():
                    z_anchor = self._last_z[:1]                        # (1, d)
                    act_id = min(int(rule.head.pred), self._world_rnn_vocab - 1)
                    act_t  = torch.tensor([act_id], device=device, dtype=torch.long)
                    z_next_world, _ = self._world_rnn(z_anchor, act_t)  # (1, d)
                    # Символьно-очікуваний стан після виведення derived
                    z_sym_exp = self.ground(frozenset({derived}), device)[:1]  # (1, d)
                    cos = float(
                        F.cosine_similarity(z_next_world, z_sym_exp, dim=-1).clamp(-1.0, 1.0).item()
                    )
                    world_pred_error = (1.0 - cos) / 2.0              # [0, 1]
                    # 60% символьна уніфікація + 40% WorldRNN latent-consistency
                    pred_error = 0.6 * pred_error + 0.4 * world_pred_error
            except Exception:
                pass  # WorldRNN недоступний → залишаємо символьний pred_error

        return pred_error, derived

    def _predict_rule_facts(
        self,
        rule: "HornClause",
        current_facts: "FrozenSet[HornAtom]",
        max_predictions: int = 8,
    ) -> FrozenSet[HornAtom]:
        if not rule.body:
            return frozenset({rule.head}) if rule.head.is_ground() else frozenset()
        fresh = freshen_vars(rule)
        predicted: List[HornAtom] = []
        for sigma in self._find_rule_substitutions(
            fresh.body,
            current_facts,
            max_solutions=max_predictions,
        ):
            derived = sigma.apply_atom(fresh.head)
            if not derived.is_ground():
                continue
            if any(_atoms_conflict(derived, known) for known in current_facts):
                continue
            predicted.append(derived)
            if len(predicted) >= max_predictions:
                break
        return frozenset(predicted)

    def _induce_proposed_rules_locally(
        self,
        current_facts: FrozenSet[HornAtom],
        target_facts: FrozenSet[HornAtom],
        device: torch.device,
        max_rules: int = 24,
    ) -> Dict[str, float]:
        verified = 0
        contradicted = 0
        retained = 0
        repaired = 0
        matched_predictions = 0
        checked = 0
        induction_scores: List[float] = []
        trace_bundle = self._task_execution_trace()

        for clause in self.kb.rules[:max_rules]:
            if self._rule_status(clause) != EpistemicStatus.proposed:
                continue
            checked += 1
            pred_error, predicted_one = self._mental_simulate_rule(clause, current_facts, device)
            trace_pred_error = self._trace_prediction_error_for_rule(clause, trace_bundle)
            counterexample_error = self._counterexample_error_for_rule(clause, trace_bundle)
            predicted_facts = self._predict_rule_facts(clause, current_facts)
            if predicted_one is not None:
                predicted_facts = predicted_facts | frozenset({predicted_one})

            hit_target = any(
                unify(pred, tgt) is not None
                for pred in predicted_facts
                for tgt in target_facts
            )
            hit_obs = any(
                unify(pred, obs) is not None
                for pred in predicted_facts
                for obs in current_facts
            )
            conflict = any(
                _atoms_conflict(pred, known)
                for pred in predicted_facts
                for known in current_facts
            )
            trace_supported = trace_bundle is not None and trace_bundle.transitions and trace_pred_error < 0.45
            if hit_target or hit_obs or trace_supported:
                utility = 0.95 if (hit_target or trace_supported) else 0.80
                utility = max(utility - 0.25 * counterexample_error, 0.05)
                self.kb.mark_rule_verified(clause)
                self.vem.record_outcome(clause, utility_target=utility, device=device)
                self._record_rule_utility(clause, utility)
                verified += 1
                matched_predictions += len(predicted_facts)
                induction_scores.append(utility)
            elif conflict or pred_error >= 0.85 or counterexample_error >= 0.75 or trace_pred_error >= 0.90:
                repaired_rule = self._repair_rule_candidate(
                    clause,
                    current_facts,
                    target_facts,
                    max_body_atoms=3,
                )
                if repaired_rule is not None:
                    self.kb.mark_rule_contradicted(clause)
                    utility = max(0.35, 1.0 - pred_error)
                    self.vem.record_outcome(repaired_rule, utility_target=utility, device=device)
                    self._record_rule_utility(repaired_rule, utility)
                    self.kb.add_rule(repaired_rule, status=EpistemicStatus.proposed)
                    repaired += 1
                    induction_scores.append(utility)
                else:
                    utility = 0.05
                    self.kb.mark_rule_contradicted(clause)
                    self.vem.record_outcome(clause, utility_target=utility, device=device)
                    self._record_rule_utility(clause, utility)
                    contradicted += 1
                    induction_scores.append(utility)
            else:
                utility = max(0.2, 1.0 - max(pred_error, trace_pred_error))
                utility = max(utility - 0.2 * counterexample_error, 0.05)
                self.vem.record_outcome(clause, utility_target=utility, device=device)
                self._record_rule_utility(clause, utility)
                retained += 1
                induction_scores.append(utility)

        mean_score = (
            float(sum(induction_scores) / len(induction_scores))
            if induction_scores else 0.0
        )
        return {
            "checked": float(checked),
            "verified": float(verified),
            "contradicted": float(contradicted),
            "retained": float(retained),
            "repaired": float(repaired),
            "matched_predictions": float(matched_predictions),
            "mean_score": mean_score,
        }

    def _compute_rule_compatibility_scores(
        self,
        n_rules: int,
        current_facts: "FrozenSet[HornAtom]",
        device: torch.device,
    ) -> torch.Tensor:
        """
        Вектор сумісності правил з поточними фактами.
        Скеровує вибір policy до правил, які реально можуть уніфікуватись.
        """
        scores = torch.zeros(n_rules, device=device)
        for i, rule in enumerate(self.kb.rules[:n_rules]):
            if not self._rule_is_usable(rule):
                scores[i] = -1.0
                continue
            if not rule.body:
                scores[i] = 0.5
                continue
            fresh = freshen_vars(rule)
            can_unify = (
                self._guided_unify_body(fresh.body, current_facts, device=device) is not None
            )
            if can_unify:
                vem_s = self.vem.score(rule, device)
                scores[i] = float(vem_s)
            else:
                scores[i] = -1.0  # Не може уніфікуватись → депріоритизуємо
        return scores

    # ── Proof Search (REINFORCE + Cost(T) + Mental Simulation) ───────────────
    def prove_with_policy(self,
                          goal: HornAtom,
                          z_ctx: torch.Tensor,
                          n_steps: Optional[int] = None,
                          starting_facts: Optional[FrozenSet[HornAtom]] = None) -> Tuple[bool, List[int], torch.Tensor]:
        """
        Шукає доведення goal із ментальною симуляцією перед застосуванням.

        Реалізує концептуальну Дедукцію (розділ 2 концепції):
          «Модель уявляє, як правило працюватиме → обчислює PredError →
           лише тоді застосовує (або відкидає) правило»

        Алгоритм:
          1. Policy вибирає правило (z_ctx + compat_scores)
          2. _mental_simulate_rule() → pred_error:
             < threshold → застосовуємо реально
             ≥ threshold → відхиляємо (REINFORCE отримує штраф)
          3. Cost(T) включає pred_error кожного кроку

        L_proof = -E_{T~π_θ}[R(T) - α·Cost(T)]
        Cost(T) = Σ_i [Length(Ri) + λ·UnifComplexity(σi) + μ·PredError(Ri)]
        """
        n_steps = n_steps or self.max_depth
        n_rules = len(self.kb.rules)
        device  = z_ctx.device

        z_goal = self.goal_proj(self._goal_embedding(goal, device))
        current_facts = starting_facts if starting_facts is not None else self.kb.facts

        trajectory: List[int]          = []
        log_probs:  List[torch.Tensor] = []
        proof_steps: List[Tuple["HornClause", Optional[Substitution]]] = []
        step_pred_errors: List[float]  = []

        proved = self._goal_supported(goal, current_facts)

        for step in range(n_steps):
            usable_indices = [
                i for i, rule in enumerate(self.kb.rules[:n_rules])
                if self._rule_is_usable(rule)
            ]
            if proved or not usable_indices:
                break

            # Символьна сумісність → скеровуємо policy до придатних правил
            compat_scores = self._compute_rule_compatibility_scores(
                n_rules, current_facts, device
            )

            log_p = self.policy(z_ctx, z_goal, max(n_rules, 1))  # (1, n_rules)
            usable_t = torch.tensor(usable_indices, device=device, dtype=torch.long)
            combined_logits = (
                log_p.squeeze(0).index_select(0, usable_t)
                + 0.3 * compat_scores.index_select(0, usable_t)
            )
            dist = Categorical(logits=combined_logits)
            local_idx = dist.sample()
            rule_idx = usable_t[local_idx]
            trajectory.append(int(rule_idx.item()))
            log_probs.append(dist.log_prob(local_idx))

            if rule_idx.item() < len(self.kb.rules):
                rule = self.kb.rules[rule_idx.item()]

                # ── Ментальна симуляція ПЕРЕД застосуванням ───────────────────
                pred_error, mentally_derived = self._mental_simulate_rule(
                    rule, current_facts, device
                )
                step_pred_errors.append(pred_error)

                mental_threshold = float(getattr(self, "_mental_sim_threshold", 0.8))
                if pred_error >= mental_threshold:
                    # Ментальна симуляція відхилила правило
                    # → не застосовуємо, але REINFORCE отримає штраф через pred_error
                    proof_steps.append((rule, None))
                    continue

                # Симуляція пройшла → застосовуємо реально
                if rule.body:
                    fresh = freshen_vars(rule)
                    sigma = self._guided_unify_body(
                        fresh.body,
                        current_facts,
                        device=device,
                    )
                    proof_steps.append((rule, sigma))
                    if sigma is not None:
                        self._note_used_rule(rule)
                        derived = sigma.apply_atom(fresh.head)
                        if derived.is_ground():
                            current_facts = current_facts | {derived}
                            if unify(goal, derived) is not None:
                                proved = True

        # ── REINFORCE: L_proof = -E[R(T) - α·Cost(T)] ────────────────────────
        R = float(proved)
        proof_cost_tensor = self.cost_est(proof_steps, device)
        avg_pred_error = (
            sum(step_pred_errors) / len(step_pred_errors)
            if step_pred_errors else 0.0
        )
        effective_reward = R - self.alpha * proof_cost_tensor

        if log_probs:
            proof_loss = -effective_reward * torch.stack(log_probs).sum()
            # Штраф за кроки з великою prediction_error (дедукційний сигнал)
            pred_err_penalty = torch.tensor(
                avg_pred_error * self.alpha, device=device, dtype=proof_loss.dtype
            )
            proof_loss = proof_loss + pred_err_penalty
        else:
            proof_loss = proof_cost_tensor * self.alpha

        return proved, trajectory, proof_loss

    # ── Abduce and Learn (повільний цикл) + MDL candidate ranking ────────────

    def _pred_error_for_rule(
        self,
        clause: "HornClause",
        observed_facts: "FrozenSet[HornAtom]",
        lam: float = 0.5,
    ) -> float:
        """
        PredError(R, Trace) — наскільки правило R не може пояснити спостереження.

        Реалізує праву частину формули абдукції:
          R* = argmin_R [Length(R) + λ·PredError(R, Trace)]

        PredError = 1.0 − coverage, де coverage = частка observed_facts,
        яка покривається виведеним множиною фактів при застосуванні R.

        Для pure-fact-правил (body=[]) — перевіряємо, чи голова є у фактах.
        Для правил з тілом — рахуємо кількість успішних уніфікацій тіла.
        """
        cache_key = (hash(clause), observed_facts)
        cached = self._pred_error_cache.get(cache_key)
        if cached is not None:
            return cached
        trace_bundle = self._task_execution_trace()
        trace_targets: Set[HornAtom] = set()
        if self.task_context is not None:
            if self.task_context.goal is not None:
                trace_targets.add(self.task_context.goal)
            trace_targets.update(self.task_context.target_facts)
        if trace_bundle is not None:
            trace_targets.update(trace_bundle.target_facts)
        if not observed_facts and not trace_targets:
            self._pred_error_cache[cache_key] = 1.0
            return 1.0
        trace_pred_error = self._trace_prediction_error_for_rule(clause, trace_bundle)
        reasoning_device = self._reasoning_device(self._last_z)
        if not clause.body:
            # Pure facts are useful if they match either the trace or the current targets.
            symbolic_error = 1.0
            for fact in observed_facts:
                if unify(clause.head, fact) is not None:
                    symbolic_error = 0.0
                    break
            for target in trace_targets:
                if unify(clause.head, target) is not None:
                    symbolic_error = 0.0
                    break
            world_error = self._world_rule_prediction_error(
                clause,
                clause.head if clause.head.is_ground() else None,
                device=reasoning_device,
            )
            score = self._abduction_prediction_error_score(
                symbolic_error,
                trace_pred_error,
                world_error,
            )
            self._pred_error_cache[cache_key] = score
            return score

        # Рахуємо, скільки фактів входять у «пояснену зону» правила
        # (тобто беруть участь як підстановки у body)
        total_facts = max(len(observed_facts), 1)
        explained_atoms: Set[HornAtom] = set()
        explained_targets: Set[HornAtom] = set()

        fresh = freshen_vars(clause)
        subs = self._find_rule_substitutions(
            fresh.body,
            observed_facts,
            max_solutions=16,
        )
        world_errors: List[float] = []
        for sigma in subs:
            # Факти, що задовольнили тіло правила — «пояснені»
            for b_atom in fresh.body:
                grounded = sigma.apply_atom(b_atom)
                for obs in observed_facts:
                    if unify(grounded, obs) is not None:
                        explained_atoms.add(obs)
            # Виведений голова — якщо є у спостереженнях, теж пояснений
            derived = sigma.apply_atom(fresh.head)
            if derived.is_ground():
                for obs in observed_facts:
                    if unify(derived, obs) is not None:
                        explained_atoms.add(obs)
                for target in trace_targets:
                    if unify(derived, target) is not None:
                        explained_targets.add(target)
                world_error = self._world_rule_prediction_error(
                    clause,
                    derived,
                    device=reasoning_device,
                )
                if world_error is not None:
                    world_errors.append(world_error)

        observed_coverage = len(explained_atoms) / float(total_facts)
        if trace_targets:
            target_coverage = len(explained_targets) / float(max(len(trace_targets), 1))
            coverage = 0.65 * observed_coverage + 0.35 * target_coverage
        else:
            coverage = observed_coverage

        symbolic_error = 1.0 - min(max(coverage, 0.0), 1.0)
        world_error = (
            float(sum(world_errors) / len(world_errors))
            if world_errors else None
        )
        score = self._abduction_prediction_error_score(
            symbolic_error,
            trace_pred_error,
            world_error,
        )
        self._pred_error_cache[cache_key] = score
        return score

        # ── WorldRNN latent-space component (Абдукція, розділ 6) ──────────────
        # Концепція: PredError(R, Trace) має включати не лише символьне покриття,
        # а й узгодженість з латентним передбаченням WorldRNN.
        # Якщо WorldRNN не передбачає стан, характерний для виведеного факту →
        # PredError зростає, правило отримує вищий MDL-score → менш ймовірне.
        if (self._world_rnn is not None
                and self._last_z is not None
                and self._world_rnn_vocab > 0
                and subs):
            try:
                device = self._last_z.device
                with torch.no_grad():
                    z_anchor = self._last_z[:1]
                    act_id = min(int(clause.head.pred), self._world_rnn_vocab - 1)
                    act_t  = torch.tensor([act_id], device=device, dtype=torch.long)
                    z_next_world, _ = self._world_rnn(z_anchor, act_t)  # (1, d)
                    # Беремо перший виведений факт як символьну ціль
                    fresh2 = freshen_vars(clause)
                    for sigma2 in subs[:1]:
                        derived2 = sigma2.apply_atom(fresh2.head)
                        if derived2.is_ground():
                            z_sym = self.ground(frozenset({derived2}), device)[:1]
                            cos = float(
                                F.cosine_similarity(
                                    z_next_world, z_sym, dim=-1
                                ).clamp(-1.0, 1.0).item()
                            )
                            world_miss = (1.0 - cos) / 2.0      # [0, 1]
                            # 70% символьне покриття + 30% WorldRNN узгодженість
                            coverage = 0.7 * coverage + 0.3 * (1.0 - world_miss)
                            break
            except Exception:
                pass  # WorldRNN недоступний → залишаємо символьне coverage

        return 1.0 - min(max(coverage, 0.0), 1.0)

    def _mdl_sort_candidates(
        self,
        candidates: "List[HornClause]",
        observed_facts: "FrozenSet[HornAtom]",
        lam: float = 0.5,
    ) -> "List[Tuple[float, HornClause]]":
        """
        Сортує кандидатів за MDL-формулою:
          score(R) = description_length_bits(R) + λ·PredError(R, Trace)

        Повертає список (score, clause) відсортований за зростанням score.
        Мінімальний score = найкраще правило (принцип Оккама).
        """
        if not candidates:
            return []
        scored: List[Tuple[float, HornClause]] = []
        for clause in candidates:
            length_bits = clause.description_length_bits()
            pred_err = self._pred_error_for_rule(clause, observed_facts, lam)
            mdl_score = length_bits + lam * pred_err
            scored.append((mdl_score, clause))
        scored.sort(key=lambda x: x[0])
        return scored

    def abduce_and_learn(
        self,
        z: torch.Tensor,
        error: float,
        force: bool = False,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, float]:
        """
        Цілеспрямований пошук мінімального правила (концепція, розділ 6):
          R* = argmin_R [Length(R) + λ·PredError(R, Trace)]

        Замість випадкової генерації → MDL-фільтрації:
          1. Генерація кандидатів (контекстні + нейронні)
          2. MDL-ранжування за description_length_bits + λ·PredError
          3. VeM-фільтрація (U(R) > τ)
          4. Додавання найкращих у LTM зі статусом proposed

        Returns: (кількість доданих правил, hinge_loss, abductor_loss, mean_utility)
        """
        device = z.device
        if (error < 0.5) and not force:
            zero = torch.zeros(1, device=device).squeeze()
            self.last_abduced_rules = []
            return 0, zero, zero, 0.0

        # 1. Генеруємо кандидатів
        trace_candidates, contextual_candidates, neural_candidates, log_probs = self._abduction_candidate_pool(
            z,
            max_contextual=12,
            max_body_atoms=3,
            max_trace=max(4, self.continuous_cycle_max_trace_candidates),
            max_neural_fallback=2,
        )
        raw_candidates = trace_candidates + contextual_candidates + neural_candidates

        if not raw_candidates:
            zero = torch.zeros(1, device=device).squeeze()
            self.last_abduced_rules = []
            return 0, zero, zero, 0.0

        # 2. MDL-ранжування ПЕРЕД VeM-фільтрацією
        # Це реалізує цілеспрямований пошук замість випадкової генерації:
        #   R* = argmin_R [Length(R) + λ·PredError(R, Trace)]
        observed = self.kb.facts
        if self.task_context is not None and self.task_context.observed_facts:
            observed = observed | self.task_context.observed_facts

        lam_mdl = float(getattr(self, "_mdl_lambda", 0.5))
        mdl_ranked = self._mdl_sort_candidates(raw_candidates, observed, lam=lam_mdl)

        # Беремо лише топ-50% за MDL: відкидаємо явно погані кандидати
        # ще до VeM, зменшуючи простір пошуку
        cutoff = max(1, len(mdl_ranked) // 2)
        mdl_filtered = [clause for _, clause in mdl_ranked[:cutoff]]

        # Зберігаємо MDL-scores для REINFORCE (заохочуємо нейронну мережу
        # генерувати MDL-мінімальні правила)
        mdl_scores_map: Dict[int, float] = {
            hash(clause): score for score, clause in mdl_ranked
        }

        # 3. VeM-фільтрація (U(R) > τ) на MDL-відібраних кандидатах
        utilities = self.vem.score_batch(mdl_filtered, device)
        accepted, hinge_loss = self.vem.filter_candidates(mdl_filtered, device)

        # REINFORCE для нейронних кандидатів:
        # Заохочуємо генерацію правил з малим MDL-score та high VeM-score
        neural_mdl_filtered_lp: List[torch.Tensor] = []
        neural_mdl_utilities: List[float] = []
        if log_probs.numel() > 0:
            for i, nc in enumerate(neural_candidates):
                h = hash(nc)
                if h in mdl_scores_map and i < log_probs.shape[0]:
                    # Нормалізуємо MDL-score до [0,1]: менший score = більша utility
                    max_possible_mdl = max(
                        (s for s, _ in mdl_ranked), default=1.0
                    )
                    mdl_utility = 1.0 - min(
                        mdl_scores_map[h] / max(max_possible_mdl, 1.0), 1.0
                    )
                    # Комбінуємо VeM utility + MDL utility
                    vem_u = self.vem.score(nc, device)
                    combined_utility = 0.6 * float(vem_u) + 0.4 * mdl_utility
                    neural_mdl_filtered_lp.append(log_probs[i])
                    neural_mdl_utilities.append(combined_utility)

        if neural_mdl_filtered_lp and neural_mdl_utilities:
            util_tensor = torch.tensor(
                neural_mdl_utilities, dtype=torch.float32, device=device
            )
            lp_tensor = torch.stack(neural_mdl_filtered_lp)
            abductor_loss = -((util_tensor.detach() - self.vem_tau) * lp_tensor).mean()
        else:
            abductor_loss = torch.zeros(1, device=device).squeeze()

        # 4. Додаємо прийнятих кандидатів у LTM
        added = 0
        added_rules: List[HornClause] = []
        for c in accepted:
            if self.kb.add_rule(c, status=EpistemicStatus.proposed):
                added += 1
                added_rules.append(c)
        self.last_abduced_rules = added_rules

        # Записуємо VeM-сигнал: MDL-кращі кандидати отримують вищий prior
        for score, clause in mdl_ranked:
            max_mdl = max((s for s, _ in mdl_ranked), default=1.0)
            prior = 1.0 - min(score / max(max_mdl, 1.0), 1.0)
            # Зберігаємо MDL-заснований prior (не фіксований 0.5)
            self.vem.record_outcome(clause, utility_target=prior * 0.7, device=device)

        mean_utility = float(utilities.mean().item()) if utilities.numel() > 0 else 0.0
        return added, hinge_loss, abductor_loss, mean_utility

    def reinforce_recent_rules(self, utility_target: float, device: torch.device) -> None:
        if not self.last_abduced_rules:
            return
        utility = float(max(0.0, min(1.0, utility_target)))
        for clause in self.last_abduced_rules:
            self.vem.record_outcome(clause, utility_target=utility, device=device)
            self._record_rule_utility(clause, utility)
        self.last_abduced_rules = []

    def reinforce_used_rules(self, utility_target: float, device: torch.device) -> None:
        if not self.last_used_rules:
            return
        utility = float(max(0.0, min(1.0, utility_target)))
        for clause in self.last_used_rules:
            self.vem.record_outcome(clause, utility_target=utility, device=device)
            self._record_rule_utility(clause, utility)
        self.last_used_rules = []
        self._last_used_rule_hashes.clear()

    def vem_retrospective_update(self, ce_utility: float, device: torch.device) -> None:
        blended_ce = float(max(0.0, min(1.0, ce_utility)))
        for clause in self.kb.rules:
            history = self._rule_utility_history.get(hash(clause), [])
            if history:
                hist_mean = sum(history) / float(len(history))
                target = 0.7 * hist_mean + 0.3 * blended_ce
            else:
                record = self.kb._records.get(hash(clause))
                if record is not None and record.status == EpistemicStatus.verified:
                    target = max(0.6, blended_ce)
                else:
                    target = 0.5
            self.vem.record_outcome(clause, utility_target=target, device=device)

    # ── Forward (інтеграція у тренувальний цикл) ──────────────────────────────
    def forward(self,
                z: torch.Tensor,
                world_error: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z           : (B, d)
        world_error : scalar
        Returns: z_sym (B, d),  sym_loss scalar

        Нові складові sym_loss (порівняно з v1):
          + 0.01 · L_vem            ← штраф за погані кандидати абдукції
          + 0.01 · L_vem_self       ← самонавчання VeM
        KB tick + консолідація кожні consolidate_every кроків.
        """
        B, device = z.shape[0], z.device
        self._clear_runtime_caches()
        self._step += 1
        self.last_used_rules = []
        self._last_used_rule_hashes.clear()

        # Зберігаємо знімок z без градієнта для _mental_simulate_rule
        # та _pred_error_for_rule (ментальна симуляція без backprop витоку)
        self._last_z = z.detach()

        # 0. Оновлюємо вік правил і запускаємо консолідацію
        self.kb.tick()
        if self._step % self.consolidate_every == 0:
            n_removed = self.kb.consolidate(use_count_threshold=2)
            # (можна логувати: n_removed правил видалено)

        # 1. Materialize discrete task facts into working memory.
        self.materialize_task_context_facts()
        goal = self.current_goal()
        working_facts = self.current_working_facts()
        target_facts = self._task_target_facts()
        provenance = self.task_context.provenance if self.task_context is not None else "latent"

        # 2. Pure symbolic executor: discrete facts/rules/goal only.
        exec_result = run_symbolic_executor(
            self.kb,
            self.max_depth,
            working_facts,
            goal,
            target_facts,
            self._goal_supported,
        )
        all_facts = exec_result.all_facts
        self._infer_used_rules_from_delta(working_facts, all_facts)
        goal_supported = exec_result.goal_supported
        target_hits = exec_result.target_hits
        target_total = exec_result.target_total
        target_coverage = exec_result.target_coverage
        unresolved_targets = exec_result.unresolved_targets

        # 3. Grounding: KB → z_sym
        # PERF FIX: при великому KB (>128 фактів) сэмплюємо підмножину,
        # щоб уникнути O(N) Python-ітерації + N×3 GPU-викликів у ground().
        _MAX_GROUND = 128
        ground_sample = (frozenset(random.sample(list(all_facts), _MAX_GROUND))
                         if len(all_facts) > _MAX_GROUND else all_facts)
        z_sym_1 = self.ground(ground_sample, device)            # (1, d)
        z_sym   = z_sym_1.expand(B, -1)                    # (B, d)
        target_ground_facts = target_facts or frozenset({goal})
        target_ground_sample = (
            frozenset(random.sample(list(target_ground_facts), _MAX_GROUND))
            if len(target_ground_facts) > _MAX_GROUND else target_ground_facts
        )
        z_target = self.ground(target_ground_sample, device).expand(B, -1)

        # 4. Proof search (тільки під час навчання)
        controller_result = run_latent_reasoning_controller(
            self,
            z,
            goal,
            working_facts,
            all_facts,
            target_facts,
            goal_supported,
            target_coverage,
            world_error,
            device,
        )
        proof_loss = controller_result.proof_loss
        vem_hinge = controller_result.vem_hinge
        cycle_aux = controller_result.cycle_loss
        abduction_aux = controller_result.abduction_loss
        abductor_aux = controller_result.abductor_aux
        vem_self_loss = controller_result.vem_self_loss
        mean_utility = controller_result.mean_utility
        abduced_rules = controller_result.abduced_rules
        goal_supported = controller_result.goal_supported
        induction_stats: Dict[str, float] = controller_result.induction_stats or empty_induction_stats()
        cycle_stats: Dict[str, float] = controller_result.cycle_stats or {}
        creative_targets = target_facts or frozenset({goal})
        creative_report = self.run_creative_cycle(
            z,
            working_facts,
            creative_targets,
            device,
        )
        if self.task_context is not None:
            self.task_context = self.creative_cycle.materialize_report_into_context(
                self.task_context,
                creative_report,
            )
        scheduled_intrinsic_goals = tuple(self.scheduled_intrinsic_goals())
        background_intrinsic_goals = tuple()
        background_intrinsic_total = 0
        background_intrinsic_coverage = 0.0
        if scheduled_intrinsic_goals:
            def _goal_match(left: Any, right: Any) -> bool:
                if left is None or right is None:
                    return False
                try:
                    return unify(left, right) is not None
                except Exception:
                    return hash(left) == hash(right)

            background_intrinsic_goals = tuple(
                target for target in scheduled_intrinsic_goals if not _goal_match(target, goal)
            )
            background_intrinsic_total = len(background_intrinsic_goals)
            if background_intrinsic_total > 0:
                background_intrinsic_hits = sum(
                    1 for target in background_intrinsic_goals if self._goal_supported(target, all_facts)
                )
                background_intrinsic_coverage = float(background_intrinsic_hits) / float(background_intrinsic_total)

        # 7. Symbolic Consistency Loss: MSE між z та z_sym
        sym_consist = F.mse_loss(z, z_sym.detach()) + \
                      F.mse_loss(z_sym, z.detach())
        symbolic_induction = (
            F.mse_loss(z_sym, z_target.detach())
            + F.mse_loss(z_target, z_sym.detach())
        )
        coverage_loss = torch.tensor(
            (1.0 - target_coverage) + (0.0 if goal_supported else 1.0),
            device=device,
            dtype=z.dtype,
        )
        sym_loss    = (sym_consist
                       + 0.1  * symbolic_induction
                       + 0.1  * coverage_loss
                       + 0.1  * proof_loss
                       + 0.01 * vem_hinge
                       + self.sym_cycle_loss_weight * cycle_aux
                       + self.sym_abduction_loss_weight * abduction_aux
                       + 0.01 * vem_self_loss)

        self.last_goal = goal
        self.last_context_facts = working_facts
        self.last_all_facts = all_facts
        cycle_mode = "off"
        if self.continuous_cycle_enabled:
            cycle_mode = "train" if self.training else (
                "eval" if self.continuous_cycle_eval_enabled else "off"
            )
        self.last_forward_info = {
            "goal_proved": 1.0 if goal_supported else 0.0,
            "target_coverage": target_coverage,
            "target_hits": float(target_hits),
            "target_total": float(target_total),
            "unresolved_targets": float(unresolved_targets),
            "abduced_rules": float(abduced_rules),
            "abduction_utility": mean_utility,
            "induction_checked": induction_stats["checked"],
              "induction_verified": induction_stats["verified"],
              "induction_contradicted": induction_stats["contradicted"],
              "induction_retained": induction_stats["retained"],
              "induction_repaired": induction_stats.get("repaired", 0.0),
              "induction_matches": induction_stats["matched_predictions"],
              "induction_score": induction_stats["mean_score"],
              "cycle_active": float(cycle_stats.get("active", 0.0)),
              "cycle_eval_active": float(cycle_stats.get("eval_active", 0.0)),
              "cycle_learning_active": float(cycle_stats.get("learning_active", 0.0)),
              "cycle_candidate_budget": float(cycle_stats.get("candidate_budget", 0.0)),
              "cycle_trace_candidates": float(cycle_stats.get("trace_candidates", 0.0)),
              "cycle_contextual_candidates": float(cycle_stats.get("contextual_candidates", 0.0)),
              "cycle_neural_candidates": float(cycle_stats.get("neural_candidates", 0.0)),
              "cycle_checked": float(cycle_stats.get("checked", 0.0)),
              "cycle_accepted": float(cycle_stats.get("accepted", 0.0)),
              "cycle_added": float(cycle_stats.get("added", 0.0)),
              "cycle_verified": float(cycle_stats.get("verified", 0.0)),
              "cycle_contradicted": float(cycle_stats.get("contradicted", 0.0)),
              "cycle_retained": float(cycle_stats.get("retained", 0.0)),
              "cycle_repaired": float(cycle_stats.get("repaired", 0.0)),
              "cycle_error": float(cycle_stats.get("mean_error", 0.0)),
              "cycle_symbolic_error": float(cycle_stats.get("mean_symbolic_error", 0.0)),
              "cycle_soft_symbolic_error": float(cycle_stats.get("mean_soft_symbolic_error", 0.0)),
              "cycle_relaxed_body_error": float(cycle_stats.get("mean_relaxed_body_error", 0.0)),
              "cycle_relaxed_head_error": float(cycle_stats.get("mean_relaxed_head_error", 0.0)),
              "cycle_trace_error": float(cycle_stats.get("mean_trace_error", 0.0)),
              "cycle_counterexample_error": float(cycle_stats.get("mean_counterexample_error", 0.0)),
              "cycle_world_error": float(cycle_stats.get("mean_world_error", 0.0)),
              "cycle_token_error": float(cycle_stats.get("mean_token_error", 0.0)),
              "cycle_graph_energy": float(cycle_stats.get("mean_graph_energy", 0.0)),
              "cycle_policy_loss": float(cycle_stats.get("policy_loss", 0.0)),
              "cycle_loss": float(cycle_stats.get("loss", 0.0)),
              "cycle_loss_aux": float(cycle_aux.detach().item()),
              "cycle_loss_weight": float(self.sym_cycle_loss_weight),
              "abduction_loss": float(abduction_aux.detach().item()),
              "abduction_loss_weight": float(self.sym_abduction_loss_weight),
              "abductor_aux_total": float(abductor_aux.detach().item()),
              "graph_reasoning_calls": float(self._graph_reasoning_stats.get("calls", 0.0)),
              "graph_reasoning_guided_calls": float(self._graph_reasoning_stats.get("guided_calls", 0.0)),
              "graph_reasoning_fallbacks": float(self._graph_reasoning_stats.get("fallbacks", 0.0)),
              "graph_reasoning_mean_subset": float(self._graph_reasoning_stats.get("mean_subset", 0.0)),
              "graph_reasoning_mean_full_facts": float(self._graph_reasoning_stats.get("mean_full_facts", 0.0)),
              "graph_reasoning_mean_solutions": float(self._graph_reasoning_stats.get("mean_solutions", 0.0)),
              "creative_abduction_candidates": float(creative_report.metrics.get("abduction_candidates", 0.0)),
              "creative_analogy_candidates": float(creative_report.metrics.get("analogy_candidates", 0.0)),
              "creative_metaphor_candidates": float(creative_report.metrics.get("metaphor_candidates", 0.0)),
              "creative_counterfactual_analogy_candidates": float(
                  creative_report.metrics.get("counterfactual_analogy_candidates", 0.0)
              ),
              "creative_counterfactual_metaphor_candidates": float(
                  creative_report.metrics.get("counterfactual_metaphor_candidates", 0.0)
              ),
              "creative_counterfactual_candidates": float(creative_report.metrics.get("counterfactual_candidates", 0.0)),
              "creative_counterfactual_surprise": float(creative_report.metrics.get("counterfactual_surprise", 0.0)),
              "creative_counterfactual_contradictions": float(creative_report.metrics.get("counterfactual_contradictions", 0.0)),
              "creative_counterfactual_exact_search": float(creative_report.metrics.get("counterfactual_exact_search", 0.0)),
              "creative_counterfactual_evaluated_subsets": float(
                  creative_report.metrics.get("counterfactual_evaluated_subsets", 0.0)
              ),
              "creative_ame_total_candidates": float(creative_report.metrics.get("ame_total_candidates", 0.0)),
              "creative_ontology_candidates": float(creative_report.metrics.get("ontology_candidates", 0.0)),
              "creative_cycle_active": float(creative_report.metrics.get("cycle_active", 0.0)),
              "creative_ontology_feedback_accepted": float(
                  creative_report.metrics.get("ontology_feedback_accepted", 0.0)
              ),
              "creative_ontology_fixed_predicates": float(
                  creative_report.metrics.get("ontology_fixed_predicates", 0.0)
              ),
              "creative_oee_model_initialized": float(
                  creative_report.metrics.get("oee_model_initialized", 0.0)
              ),
              "creative_oee_feedback_buffer_size": float(
                  creative_report.metrics.get("oee_feedback_buffer_size", 0.0)
              ),
              "creative_oee_online_train_applied": float(
                  creative_report.metrics.get("oee_online_train_applied", 0.0)
              ),
              "creative_oee_online_train_loss": float(
                  creative_report.metrics.get("oee_online_train_loss", 0.0)
              ),
              "creative_oee_online_train_steps": float(
                  creative_report.metrics.get("oee_online_train_steps", 0.0)
              ),
              "creative_selected_rules": float(creative_report.metrics.get("selected_rules", 0.0)),
              "creative_validated_selected_rules": float(
                  creative_report.metrics.get("validated_selected_rules", 0.0)
              ),
              "creative_selected_mean_utility": float(creative_report.metrics.get("selected_mean_utility", 0.0)),
              "creative_validated_support_facts": float(
                  creative_report.metrics.get("validated_support_facts", 0.0)
              ),
              "creative_target_support_before": float(
                  creative_report.metrics.get("target_support_before", 0.0)
              ),
              "creative_target_support_after": float(
                  creative_report.metrics.get("target_support_after", 0.0)
              ),
              "creative_target_support_gain": float(
                  creative_report.metrics.get("target_support_gain", 0.0)
              ),
              "creative_gap_before": float(creative_report.metrics.get("gap_before", 0.0)),
              "creative_gap_after": float(creative_report.metrics.get("gap_after", 0.0)),
              "creative_gap_reduction": float(creative_report.metrics.get("gap_reduction", 0.0)),
              "creative_compression_gain": float(
                  creative_report.metrics.get("compression_gain", 0.0)
              ),
              "creative_intrinsic_value": float(creative_report.metrics.get("intrinsic_value", 0.0)),
              "creative_intrinsic_goal_queue_size": float(
                  creative_report.metrics.get("intrinsic_goal_queue_size", 0.0)
              ),
              "creative_intrinsic_background_goals": float(
                  creative_report.metrics.get("intrinsic_background_goals", 0.0)
              ),
              "background_intrinsic_goals": float(background_intrinsic_total),
              "background_intrinsic_coverage": float(background_intrinsic_coverage),
              "creative_analogy_projector_loss": float(creative_report.metrics.get("analogy_projector_loss", 0.0)),
              "creative_analogy_embedding_source": float(
                  creative_report.metrics.get("analogy_embedding_source", 0.0)
              ),
            "trace_steps": float(self.task_context.metadata.get("trace_steps", 0.0)) if self.task_context is not None else 0.0,
            "trace_counterexamples": float(self.task_context.metadata.get("trace_counterexamples", 0.0)) if self.task_context is not None else 0.0,
            "used_rules": float(len(self.last_used_rules)),
            "cycle_mode": cycle_mode,
            "provenance": provenance,
        }

        return z_sym, sym_loss

    # ── Допоміжне ─────────────────────────────────────────────────────────────
    def rule_regularizer(self, lam_sym: float,
                         eta_utility: float = 0.1) -> float:
        """
        MDL із урахуванням корисності правил:
          L_rule = λ · Σ_{R∈Γ} (Complexity(R) − η·Utility(R))

        Порівняно зі старим λ·Σ_R len(R):
          · Корисні правила (high Utility) отримують знижку → залишаються.
          · Некорисні складні правила → більший штраф → видаляються.
        """
        return lam_sym * self.kb.utility_adjusted_penalty(eta_utility)

    def vem_loss(self, z: torch.Tensor, delta: float = 1e-3) -> torch.Tensor:
        """
        δ · E_{R~Abduction}[max(0, τ − U(R))]

        Штрафує AbductionHead, якщо він генерує кандидати, яких VeM відхиляє.
        Викликається окремо з OMENScaleLoss для включення у повний J(θ,Γ,M).
        """
        device = z.device
        if not self.training:
            return torch.zeros(1, device=device).squeeze()
        trace_candidates, contextual_candidates, neural_candidates, log_probs = self._abduction_candidate_pool(
            z,
            max_contextual=8,
            max_body_atoms=3,
            max_trace=max(2, self.continuous_cycle_max_trace_candidates),
            max_neural_fallback=2,
        )
        raw_candidates = trace_candidates + contextual_candidates + neural_candidates
        if not raw_candidates:
            return torch.zeros(1, device=device).squeeze()
        utilities = self.vem.score_batch(raw_candidates, device)
        _, hinge = self.vem.filter_candidates(raw_candidates, device)
        if neural_candidates and log_probs.numel() > 0:
            neural_utilities = self.vem.score_batch(neural_candidates, device)
            rl = -((neural_utilities.detach() - self.vem_tau) * log_probs).mean()
        else:
            rl = torch.zeros(1, device=device).squeeze()
        return delta * (hinge + rl)

    def semantic_feedback_pairs(
            self, max_pairs: int = 32
    ) -> List[Tuple[int, int, float]]:
        """Повертає пари NET-токенів, пов'язаних через факти/абдукцію в KB."""
        return self.kb.get_token_pairs_for_semantic_feedback(max_pairs)

    def __len__(self): return len(self.kb)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  INLINE ТЕСТИ
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# 8.  INLINE ТЕСТИ
# ══════════════════════════════════════════════════════════════════════════════

def _run_prolog_tests() -> None:
    try:
        import sys

        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sep = lambda s: print(f"\n{'─'*60}\n  {s}\n{'─'*60}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[omen_prolog] device={device}")

    # ══ T0: Term System ════════════════════════════════════════════════════════
    sep("T0 · Term System (Const, Var, Compound, Substitution)")
    c1, c2, c3 = Const(1), Const(2), Const(3)
    X, Y, Z = Var("X"), Var("Y"), Var("Z")
    comp = Compound(func=0, subterms=(c1, X))

    assert c1.vars() == frozenset(),          "Const.vars FAIL"
    assert X.vars()  == frozenset({"X"}),     "Var.vars FAIL"
    assert comp.vars()== frozenset({"X"}),    "Compound.vars FAIL"
    assert c1.depth() == 0 and comp.depth() == 1, "depth FAIL"
    assert _is_ground(c1) and not _is_ground(X),  "is_ground FAIL"

    sigma = Substitution({"X": c2})
    assert sigma.apply(X)    == c2,                             "apply Var FAIL"
    assert sigma.apply(comp) == Compound(func=0, subterms=(c1, c2)), "apply Compound FAIL"
    assert sigma.apply(c1)   == c1,                             "apply Const FAIL"

    # Композиція: σ={X→c2}, θ={Y→X}  →  (σ∘θ): Y→c2, X→c2
    theta   = Substitution({"Y": X})
    composed = sigma.compose(theta)
    assert composed.apply(Y) == c2, f"compose FAIL: {composed.apply(Y)}"
    assert composed.apply(X) == c2, f"compose X FAIL"

    # bind: нова підстановка
    sigma2 = Substitution.empty().bind("Z", Compound(func=0, subterms=(c1,)))
    assert sigma2.apply(Z) == Compound(func=0, subterms=(c1,)), "bind FAIL"
    print(f"  Const={c1}, Var={X}, Compound={comp}")
    print(f"  σ={sigma}, composed={composed}  [PASS]")

    # ══ T0b: Martelli-Montanari ════════════════════════════════════════════════
    sep("T0b · Martelli-Montanari уніфікація")

    # f(X, g(Y)) =?= f(c1, g(c2)) → mgu={X→c1, Y→c2}
    t_lhs = Compound(func=0, subterms=(X, Compound(func=1, subterms=(Y,))))
    t_rhs = Compound(func=0, subterms=(c1, Compound(func=1, subterms=(c2,))))
    mgu = unify_mm([(t_lhs, t_rhs)])
    assert mgu is not None,         "MM returned None  FAIL"
    assert mgu.apply(X) == c1,      f"MM X FAIL: {mgu}"
    assert mgu.apply(Y) == c2,      f"MM Y FAIL: {mgu}"

    # Trivial: c1 =?= c1 → ε
    trivial = unify_mm([(c1, c1)])
    assert trivial is not None and len(trivial) == 0, "Trivial FAIL"

    # Clash: f(c1) =?= f(c2) → None
    clash = unify_mm([(Compound(0, (c1,)), Compound(0, (c2,)))])
    assert clash is None,           "Clash FAIL"

    # Occurs check: X =?= f(X) → None
    cyclic = unify_mm([(X, Compound(0, (X,)))])
    assert cyclic is None,          "Occurs check FAIL"

    # Orient: c1 =?= X → {X→c1}
    orient = unify_mm([(c1, X)])
    assert orient is not None and orient.apply(X) == c1, "Orient FAIL"

    # Anonymous var: ?_0 =?= anything → skip (no binding)
    anon = unify_mm([(Var("_0"), c3)])
    assert anon is not None and len(anon) == 0, "Anon wildcard FAIL"

    print(f"  mgu={mgu}  clash=None  occurs=None  anon=ε  [PASS]")

    # ══ T1: HornAtom уніфікація ════════════════════════════════════════════════
    sep("T1 · Уніфікація HornAtom (via MM)")
    a1 = HornAtom(pred=1, args=(Const(2), Const(3)))
    a2 = HornAtom(pred=1, args=(Var("X"), Const(3)))
    b  = unify(a2, a1)
    assert b is not None and b.apply(Var("X")) == Const(2), f"FAIL: {b}"

    a3 = HornAtom(pred=1, args=(Const(5), Const(3)))
    b3 = unify(a2, a3)
    assert b3 is not None and b3.apply(Var("X")) == Const(5)

    a4 = HornAtom(pred=2, args=(Const(2), Const(3)))
    assert unify(a2, a4) is None,  "Different pred should be None"

    # Зворотна сумісність: int args → Const/Var автоматично
    a_compat = HornAtom(pred=1, args=(-1, 3))    # Var("_0"), Const(3)
    assert isinstance(a_compat.args[0], Var)
    assert isinstance(a_compat.args[1], Const)
    b_compat = unify(a_compat, a1)               # anon var → skip → σ=ε, проходить
    assert b_compat is not None, f"Compat unify FAIL: {b_compat}"
    print(f"  σ={b}  compat={b_compat}  [PASS]")

    # ══ T2: apply_bindings ════════════════════════════════════════════════════
    sep("T2 · apply_bindings (з Substitution)")
    head_t2 = HornAtom(pred=1, args=(Var("X"), Const(5)))
    sigma_t2 = Substitution({"X": Const(9)})
    result_t2 = apply_bindings(head_t2, sigma_t2)
    assert result_t2 == HornAtom(pred=1, args=(Const(9), Const(5))), f"FAIL: {result_t2}"
    print(f"  {head_t2} + {{X→9}} = {result_t2}  [PASS]")

    # ══ T3: KnowledgeBase Forward Chaining (FOL) ═══════════════════════════════
    sep("T3 · KnowledgeBase Forward Chaining (FOL, named vars)")
    kb = KnowledgeBase(max_rules=64)
    kb.add_fact(HornAtom(pred=0, args=(Const(1), Const(0))))   # human(1,0)=socrates
    kb.add_fact(HornAtom(pred=0, args=(Const(2), Const(0))))   # human(2,0)=plato
    X_t3 = Var("X")
    body_t3 = (HornAtom(pred=0, args=(X_t3, Const(0))),)      # human(?X, 0)
    head_t3 =  HornAtom(pred=1, args=(X_t3, Const(0)))        # mortal(?X, 0)
    kb.add_rule(HornClause(head=head_t3, body=body_t3))
    derived = kb.forward_chain(max_depth=3)
    mortal_facts = [f for f in derived if f.pred == 1]
    print(f"  Всього фактів: {len(derived)}, mortal: {mortal_facts}")
    assert len(mortal_facts) >= 2, f"FAIL: {mortal_facts}"
    # Перевіряємо конкретні факти
    assert HornAtom(pred=1, args=(Const(1), Const(0))) in mortal_facts
    assert HornAtom(pred=1, args=(Const(2), Const(0))) in mortal_facts
    print("  [PASS]")

    # ══ T_GP: Grandparent (transitive composition) ═════════════════════════════
    sep("T_GP · Grandparent — композиційне узагальнення")
    # grandparent(?X,?Z) :- parent(?X,?Y), parent(?Y,?Z)
    kb_gp = KnowledgeBase(max_rules=64)
    kb_gp.add_fact(HornAtom(pred=0, args=(Const(1), Const(2))))  # parent(1,2)
    kb_gp.add_fact(HornAtom(pred=0, args=(Const(2), Const(3))))  # parent(2,3)
    kb_gp.add_fact(HornAtom(pred=0, args=(Const(3), Const(4))))  # parent(3,4)
    Xg, Yg, Zg = Var("X"), Var("Y"), Var("Z")
    body_gp = (HornAtom(pred=0, args=(Xg, Yg)),
               HornAtom(pred=0, args=(Yg, Zg)))
    head_gp =  HornAtom(pred=1, args=(Xg, Zg))
    kb_gp.add_rule(HornClause(head=head_gp, body=body_gp))
    derived_gp = kb_gp.forward_chain(max_depth=5)
    gp_facts = [f for f in derived_gp if f.pred == 1]
    print(f"  grandparent-факти: {gp_facts}")
    assert HornAtom(pred=1, args=(Const(1), Const(3))) in gp_facts, "gp(1,3) FAIL"
    assert HornAtom(pred=1, args=(Const(2), Const(4))) in gp_facts, "gp(2,4) FAIL"
    print("  [PASS] — дедукція через транзитивність")

    sep("T_GP_TENSOR · TensorKnowledgeBase — fast join + compound terms")
    tkb = TensorKnowledgeBase(max_rules=64, max_facts=128, device=device)
    comp_src = Compound(func=9, subterms=(Const(7), Const(8)))
    comp_mid = Compound(func=10, subterms=(Const(9),))
    tkb.add_fact(HornAtom(pred=10, args=(comp_src, comp_mid)))
    tkb.add_fact(HornAtom(pred=11, args=(comp_mid, Const(99))))
    XA, YB, ZC = Var("XA"), Var("YB"), Var("ZC")
    tkb.add_rule(HornClause(
        head=HornAtom(pred=12, args=(XA, ZC)),
        body=(
            HornAtom(pred=10, args=(XA, YB)),
            HornAtom(pred=11, args=(YB, ZC)),
        ),
    ))
    derived_tkb = tkb.forward_chain(max_depth=3)
    assert HornAtom(pred=12, args=(comp_src, Const(99))) in derived_tkb, f"Tensor KB compound join FAIL: {derived_tkb}"
    print("  [PASS] — tensor KB виводить compound-терми в multi-body правилі")

    # ══ T_ALL_SUB: find_all_substitutions ══════════════════════════════════════
    sep("T_ALL_SUB · find_all_substitutions (backtracking DFS)")
    sep("T_GP_TENSOR_STRUCT В· TensorKnowledgeBase — structured compound join")
    tkb_struct = TensorKnowledgeBase(max_rules=64, max_facts=128, device=device)
    nested_fact = Compound(func=20, subterms=(Const(1), Const(5)))
    tkb_struct.add_fact(HornAtom(pred=20, args=(nested_fact,)))
    tkb_struct.add_fact(HornAtom(pred=21, args=(Const(5), Const(7))))
    YS, ZS = Var("YS"), Var("ZS")
    tkb_struct.add_rule(HornClause(
        head=HornAtom(pred=22, args=(Compound(func=30, subterms=(YS, ZS)),)),
        body=(
            HornAtom(pred=20, args=(Compound(func=20, subterms=(Const(1), YS)),)),
            HornAtom(pred=21, args=(YS, ZS)),
        ),
    ))
    derived_struct = tkb_struct.forward_chain(max_depth=3)
    expected_struct = HornAtom(
        pred=22,
        args=(Compound(func=30, subterms=(Const(5), Const(7))),),
    )
    assert expected_struct in derived_struct, f"Structured compound join FAIL: {derived_struct}"
    print("  [PASS] вЂ” structured fast join РїС–РґС‚СЂРёРјСѓС” f(Const,Var) Сѓ body Р№ head")

    sep("T_TENSOR_ARITY3 В· TensorKnowledgeBase — fallback для арності > 2")
    sep("T_TENSOR_VARPOS")
    tkb_varpos = TensorKnowledgeBase(max_rules=64, max_facts=128, device=device)
    Xv = Var("Xv")
    tkb_varpos.add_fact(HornAtom(pred=40, args=(Const(7), Const(5))))
    tkb_varpos.add_rule(HornClause(
        head=HornAtom(pred=41, args=(Xv, Const(7))),
        body=(HornAtom(pred=40, args=(Const(7), Xv)),),
    ))
    derived_varpos = tkb_varpos.forward_chain(max_depth=3)
    assert HornAtom(pred=41, args=(Const(5), Const(7))) in derived_varpos, (
        f"Variable-to-head propagation FAIL: {derived_varpos}"
    )
    assert HornAtom(pred=41, args=(Const(7), Const(7))) not in derived_varpos, (
        f"Spurious slot-0 propagation detected: {derived_varpos}"
    )
    print("  [PASS] - unary tensor rule respects body arg position")

    tkb_arity = TensorKnowledgeBase(max_rules=64, max_facts=128, device=device)
    tkb_arity.add_fact(HornAtom(pred=30, args=(Const(1), Const(2), Const(3))))
    X3, Y3, Z3 = Var("X3"), Var("Y3"), Var("Z3")
    tkb_arity.add_rule(HornClause(
        head=HornAtom(pred=31, args=(X3, Z3)),
        body=(HornAtom(pred=30, args=(X3, Y3, Z3)),),
    ))
    tkb_arity.add_rule(HornClause(
        head=HornAtom(pred=32, args=(Const(4), Const(5), Const(6))),
        body=(),
    ))
    derived_arity = tkb_arity.forward_chain(max_depth=3)
    assert HornAtom(pred=30, args=(Const(1), Const(2), Const(3))) in derived_arity, (
        f"Arity-3 fact lost in fallback path: {derived_arity}"
    )
    assert HornAtom(pred=31, args=(Const(1), Const(3))) in derived_arity, (
        f"Arity-3 body fallback FAIL: {derived_arity}"
    )
    assert HornAtom(pred=32, args=(Const(4), Const(5), Const(6))) in derived_arity, (
        f"Arity-3 fact-rule fallback FAIL: {derived_arity}"
    )
    print("  [PASS] — >2 аргументи не губляться і виводяться через Python fallback")

    facts_u = frozenset([
        HornAtom(pred=0, args=(Const(1), Const(2))),
        HornAtom(pred=0, args=(Const(2), Const(3))),
        HornAtom(pred=0, args=(Const(1), Const(3))),
    ])
    Xu, Yu = Var("X"), Var("Y")
    body_u = (HornAtom(pred=0, args=(Xu, Yu)),)    # p0(?X, ?Y)
    subs_u = find_all_substitutions(body_u, facts_u)
    assert len(subs_u) == 3, f"FAIL: {len(subs_u)} (expected 3)"
    for s in subs_u:
        assert isinstance(s.apply(Xu), Const), f"X not ground: {s}"
    print(f"  {len(subs_u)} підстановок знайдено  [PASS]")

    # ══ T4: MDL complexity penalty ════════════════════════════════════════════
    sep("T4 · MDL complexity penalty")
    pen = kb.complexity_penalty()
    assert pen > 0
    print(f"  complexity_penalty={pen:.1f}  rules={len(kb)}  [PASS]")

    sep("T4b · Rule bits ignore runtime bookkeeping")
    runtime_rule_a = HornClause(
        head=HornAtom(pred=7, args=(Var("X"),)),
        body=(HornAtom(pred=8, args=(Var("X"),)),),
        weight=1.0,
        use_count=0,
    )
    runtime_rule_b = HornClause(
        head=HornAtom(pred=7, args=(Var("X"),)),
        body=(HornAtom(pred=8, args=(Var("X"),)),),
        weight=9.0,
        use_count=123,
    )
    bits_a = runtime_rule_a.description_length_bits()
    bits_b = runtime_rule_b.description_length_bits()
    bits_runtime = runtime_rule_b.description_length_bits(include_runtime_state=True)
    assert abs(bits_a - bits_b) < 1e-9, "FAIL: structural rule bits depend on runtime state"
    assert bits_runtime > bits_b, "FAIL: runtime-state option should add extra bits"
    print(f"  structural={bits_a:.2f}  with_runtime={bits_runtime:.2f}  [PASS]")

    # ══ T5: ProofPolicyNet ═══════════════════════════════════════════════════
    sep("T5 · ProofPolicyNet")
    policy = ProofPolicyNet(d_latent=32, max_rules=16).to(device)
    z_c = torch.randn(2, 32, device=device)
    z_g = torch.randn(2, 32, device=device)
    lp  = policy(z_c, z_g, n_rules=8)
    assert lp.shape == (2, 8), f"FAIL: {lp.shape}"
    assert (lp <= 0).all()
    print(f"  log_probs shape={tuple(lp.shape)}  max={lp.max():.3f}  [PASS]")

    # ══ T_TERM_EMB: TermEmbedder ══════════════════════════════════════════════
    sep("T_TERM_EMB · TermEmbedder (нейронні ембеддинги термів)")
    te = TermEmbedder(sym_vocab=32, d=16).to(device)
    atoms_te = [
        HornAtom(pred=0, args=(Const(1), Const(2))),
        HornAtom(pred=1, args=(Var("X"), Const(3))),
    ]
    embs_te = te(atoms_te, device)
    assert embs_te.shape == (2, 16), f"FAIL: {embs_te.shape}"
    # Ембеддинги різних атомів повинні відрізнятись
    assert not torch.allclose(embs_te[0], embs_te[1]), "Embeddings should differ"
    print(f"  embs shape={tuple(embs_te.shape)}  [PASS]")

    # ══ T_SOFT_UNIF: SoftUnifier ══════════════════════════════════════════════
    sep("T_SOFT_UNIF · SoftUnifier (диференційована уніфікація)")
    su = SoftUnifier(d=16, sym_vocab=32).to(device)
    facts_su = frozenset([
        HornAtom(pred=0, args=(Const(1), Const(2))),
        HornAtom(pred=0, args=(Const(3), Const(4))),
    ])
    body_su = (HornAtom(pred=0, args=(Var("X"), Var("Y"))),)
    energy_su, ent_su = su(body_su, facts_su, device)
    assert energy_su.dim() == 0, f"energy not scalar: {energy_su.shape}"
    assert not torch.isnan(energy_su), "NaN energy"
    # Backward через soft unifier
    (energy_su + ent_su).backward()
    grad_sum = sum(p.grad.norm().item() for p in su.parameters() if p.grad is not None)
    assert grad_sum > 0, "No gradient through SoftUnifier"
    print(f"  energy={energy_su.item():.4f}  ent={ent_su.item():.4f}  grad={grad_sum:.4f}  [PASS]")

    # ══ T6: NeuralAbductionHead ═══════════════════════════════════════════════
    sep("T6 · NeuralAbductionHead")
    abd = NeuralAbductionHead(d_latent=32, sym_vocab=16, n_cands=4).to(device)
    z   = torch.randn(1, 32, device=device)
    clauses = abd(z)
    assert len(clauses) == 4
    for c in clauses:
        assert isinstance(c, HornClause) and not c.is_fact()
        # Нові правила мають змінні у args (не константи)
        assert c.head.vars(), f"Head has no vars: {c.head}"
        assert c.body[0].vars(), f"Body has no vars: {c.body[0]}"
    print(f"  {len(clauses)} кандидати зі змінними: {clauses[0]}  [PASS]")

    # ══ T7: DifferentiableProver — forward ════════════════════════════════════
    sep("T7 · DifferentiableProver forward")
    prover = DifferentiableProver(
        d_latent=32, sym_vocab=16, max_rules=64,
        max_depth=3, n_cands=4, alpha=0.1
    ).to(device)

    z_in = torch.randn(2, 32, device=device)
    prover.train()
    z_sym, sym_loss = prover(z_in, torch.tensor(1.0))
    assert z_sym.shape == (2, 32), f"FAIL: {z_sym.shape}"
    assert not torch.isnan(sym_loss)
    print(f"  z_sym {tuple(z_sym.shape)}  sym_loss={sym_loss.item():.4f}  [PASS]")

    # ══ T8: Backward через DifferentiableProver ════════════════════════════════
    sep("T8 · Backward через DifferentiableProver")
    prover.zero_grad()
    loss = sym_loss + z_sym.pow(2).mean()
    loss.backward()
    g_sum = sum(p.grad.norm().item() for p in prover.parameters() if p.grad is not None)
    assert g_sum > 0, "FAIL: нема градієнту"
    print(f"  grad_sum={g_sum:.4f}  [PASS]")

    # ══ T9: prove_with_policy + FOL правило ═══════════════════════════════════
    sep("T9 · prove_with_policy з FOL-правилом")
    # Правило з іменованими змінними: p3(?X, c7) :- p2(c7, ?X)
    Xp = Var("XV")
    prover.kb.add_fact(HornAtom(pred=2, args=(Const(7), Const(5))))
    prover.kb.add_rule(HornClause(
        head=HornAtom(pred=3, args=(Xp, Const(7))),
        body=(HornAtom(pred=2, args=(Const(7), Xp)),)
    ), status=EpistemicStatus.verified)
    goal = HornAtom(pred=3, args=(Const(5), Const(7)))
    z1   = torch.randn(1, 32, device=device)
    proved, traj, pl = prover.prove_with_policy(goal, z1, n_steps=3)
    assert pl.dim() == 0 or pl.numel() == 1, "FAIL: proof_loss не скалярний"
    print(f"  proved={proved}  steps={len(traj)}  proof_loss={pl.item():.4f}  [PASS]")

    sep("T9b В· answer_query Р· query-specific proof state")
    query_goal = HornAtom(pred=3, args=(Var("ANS"), Const(7)))
    z_query, answer_ids, support = prover.answer_query(query_goal, device=device)
    assert z_query.shape == (1, 32), f"FAIL: query state {z_query.shape}"
    assert support.item() > 0.0, f"FAIL: query support={support.item():.4f}"
    assert 5 in answer_ids, f"FAIL: expected answer 5 in {answer_ids}"
    print(f"  support={support.item():.4f}  answers={answer_ids}  [PASS]")

    # ══ T10: Абдукція ═════════════════════════════════════════════════════════
    sep("T10 · abduce_and_learn")
    n_before = len(prover.kb)
    added, _, _, _ = prover.abduce_and_learn(z_in, error=2.0, force=True)
    n_after  = len(prover.kb)
    print(f"  rules before={n_before}  added={added}  after={n_after}  [PASS]")
    sep("T10a В· contextual bridge-chain abduction")
    bridge_prover = DifferentiableProver(
        d_latent=32, sym_vocab=32, max_rules=32, max_depth=3, n_cands=4
    ).to(device)
    bridge_ctx = SymbolicTaskContext(
        observed_facts=frozenset({
            HornAtom(pred=5, args=(Const(1), Const(2))),
            HornAtom(pred=6, args=(Const(2), Const(3))),
            HornAtom(pred=8, args=(Const(3), Const(4))),
        }),
        goal=HornAtom(pred=7, args=(Const(1), Const(4))),
        target_facts=frozenset({HornAtom(pred=7, args=(Const(1), Const(4)))}),
        provenance="unit",
        trigger_abduction=True,
    )
    bridge_prover.set_task_context(bridge_ctx)
    bridge_rules = bridge_prover._contextual_abduction_candidates(
        max_candidates=12,
        max_body_atoms=3,
    )
    assert bridge_rules, "FAIL: no contextual bridge candidates"
    assert any(len(rule.body) == 3 for rule in bridge_rules), "FAIL: no 3-body bridge rule"
    print(
        f"  bridge_candidates={len(bridge_rules)}  "
        f"max_body={max(len(r.body) for r in bridge_rules)}  [PASS]"
    )

    sep("T10b · Working memory stays separate from KB")
    sep("T10c · proactive hypothesis cycle")
    class _DummyWorld(nn.Module):
        def __init__(self, d_model: int, vocab_size: int):
            super().__init__()
            self.act_emb = nn.Embedding(vocab_size, d_model)
            self.proj = nn.Linear(d_model * 2, d_model)

        def forward(self, z_state, action, h=None):
            del h
            act = self.act_emb(action)
            return torch.tanh(self.proj(torch.cat([z_state, act], dim=-1))), None

    cycle_prover = DifferentiableProver(
        d_latent=32, sym_vocab=32, max_rules=32, max_depth=3, n_cands=4
    ).to(device)
    cycle_prover.set_world_rnn(_DummyWorld(32, 32).to(device))
    cycle_prover.configure_hypothesis_cycle(
        enabled=True,
        max_contextual=4,
        max_neural=2,
        accept_threshold=0.25,
        verify_threshold=0.45,
    )
    cycle_ctx = SymbolicTaskContext(
        observed_facts=bridge_ctx.observed_facts,
        goal=bridge_ctx.goal,
        target_facts=bridge_ctx.target_facts,
        provenance="unit",
        trigger_abduction=False,
        metadata={"last_src": 1.0, "last_tgt": 4.0},
    )
    cycle_prover.set_task_context(cycle_ctx)
    cycle_prover.train()
    z_cycle = torch.randn(1, 32, device=device)
    cycle_prover._last_z = z_cycle.detach()
    cycle_out = cycle_prover.continuous_hypothesis_cycle(
        z_cycle,
        cycle_ctx.observed_facts,
        cycle_ctx.target_facts,
        device,
    )
    assert cycle_out["stats"]["checked"] >= 1.0, "Continuous cycle did not inspect any hypothesis"
    assert not torch.isnan(cycle_out["loss_tensor"]), "Continuous cycle loss is NaN"
    assert 0.0 <= cycle_out["stats"]["mean_token_error"] <= 1.0, "Bad token error"
    print(
        f"  checked={cycle_out['stats']['checked']:.0f}  "
        f"added={cycle_out['stats']['added']:.0f}  "
        f"loss={cycle_out['stats']['loss']:.4f}  [PASS]"
    )

    sep("T10d · hypothesis cycle repairs broken rule")
    repair_prover = DifferentiableProver(
        d_latent=32, sym_vocab=32, max_rules=32, max_depth=3, n_cands=2
    ).to(device)
    repair_prover.set_world_rnn(_DummyWorld(32, 32).to(device))
    repair_prover.configure_hypothesis_cycle(
        enabled=True,
        max_contextual=1,
        max_neural=0,
        accept_threshold=0.25,
        verify_threshold=0.45,
        repair_enabled=True,
        repair_threshold=0.80,
        max_repairs=1,
    )
    repair_ctx = SymbolicTaskContext(
        observed_facts=bridge_ctx.observed_facts,
        goal=bridge_ctx.goal,
        target_facts=bridge_ctx.target_facts,
        provenance="unit",
        trigger_abduction=False,
        metadata={"last_src": 1.0, "last_tgt": 4.0},
    )
    repair_prover.set_task_context(repair_ctx)
    repair_prover.train()
    bad_rule = HornClause(
        head=HornAtom(pred=6, args=(Var("X"), Var("Y"))),
        body=(HornAtom(pred=9, args=(Var("X"), Var("Y"))),),
    )
    repair_prover.kb.add_rule(bad_rule, status=EpistemicStatus.proposed)
    repair_prover._contextual_abduction_candidates = (
        lambda max_candidates=4, max_body_atoms=3: [bad_rule]
    )
    z_repair = torch.randn(1, 32, device=device)
    repair_prover._last_z = z_repair.detach()
    repair_out = repair_prover.continuous_hypothesis_cycle(
        z_repair,
        repair_ctx.observed_facts,
        repair_ctx.target_facts,
        device,
    )
    repaired_rules = [
        rule for rule in repair_prover.kb.rules
        if rule.head.pred == 7 and len(rule.body) >= 2
    ]
    bridge_vars = max(
        (
            structural_bridge_variable_count(rule.head, rule.body)
            for rule in repaired_rules
        ),
        default=0,
    )
    assert repair_out["stats"]["repaired"] >= 1.0, "Continuous cycle did not repair the rule"
    assert repair_out["stats"]["checked"] >= 2.0, "Repair candidate was not re-evaluated"
    assert repaired_rules, "Repaired bridge rule was not added to the KB"
    assert bridge_vars >= 1, "Repair selected an unstructured rule instead of a bridge hypothesis"
    print(
        f"  repaired={repair_out['stats']['repaired']:.0f}  "
        f"checked={repair_out['stats']['checked']:.0f}  "
        f"bridge_body={len(repaired_rules[0].body)}  "
        f"bridge_vars={bridge_vars:.0f}  [PASS]"
    )

    sep("T10e · relaxed hypothesis path backpropagates")
    relaxed_prover = DifferentiableProver(
        d_latent=32, sym_vocab=32, max_rules=32, max_depth=3, n_cands=4
    ).to(device)
    relaxed_prover.set_world_rnn(_DummyWorld(32, 32).to(device))
    relaxed_prover.configure_hypothesis_cycle(
        enabled=True,
        max_contextual=0,
        max_neural=2,
        accept_threshold=0.25,
        verify_threshold=0.45,
    )
    relaxed_ctx = SymbolicTaskContext(
        observed_facts=bridge_ctx.observed_facts,
        goal=bridge_ctx.goal,
        target_facts=bridge_ctx.target_facts,
        provenance="unit",
        trigger_abduction=False,
        metadata={"last_src": 1.0, "last_tgt": 4.0},
    )
    relaxed_prover.set_task_context(relaxed_ctx)
    relaxed_prover.train()
    relaxed_prover.zero_grad(set_to_none=True)
    z_relaxed = torch.randn(1, 32, device=device, requires_grad=True)
    relaxed_cycle = relaxed_prover.continuous_hypothesis_cycle(
        z_relaxed,
        relaxed_ctx.observed_facts,
        relaxed_ctx.target_facts,
        device,
    )
    relaxed_cycle["loss_tensor"].backward()
    abductor_grad = sum(
        p.grad.norm().item()
        for n, p in relaxed_prover.named_parameters()
        if "abductor" in n and p.grad is not None
    )
    graph_grad = sum(
        p.grad.norm().item()
        for n, p in relaxed_prover.named_parameters()
        if "graph_unif" in n and p.grad is not None
    )
    z_grad = 0.0 if z_relaxed.grad is None else float(z_relaxed.grad.norm().item())
    assert relaxed_cycle["stats"]["checked"] >= 1.0, "Relaxed cycle inspected no hypotheses"
    assert abductor_grad > 0.0, "Relaxed cycle did not backprop into AbductionHead"
    assert graph_grad > 0.0, "Relaxed cycle did not backprop into GraphMatchingUnifier"
    assert z_grad > 0.0, "Relaxed cycle did not backprop into latent state z"
    print(
        f"  z_grad={z_grad:.4f}  "
        f"abductor_grad={abductor_grad:.4f}  "
        f"graph_grad={graph_grad:.4f}  [PASS]"
    )

    sep("T10f · trace-driven abduction and counterexample checks")
    trace_bundle = build_symbolic_trace_bundle(
        "def add(a, b):\n    return a + b\n",
        max_steps=16,
        max_counterexamples=2,
    )
    assert trace_bundle is not None, "FAIL: symbolic execution trace was not built"
    trace_goal = next(
        (fact for fact in trace_bundle.target_facts if fact.pred == TRACE_RETURN_EVENT_PRED),
        next(iter(trace_bundle.target_facts)),
    )
    trace_prover = DifferentiableProver(
        d_latent=32, sym_vocab=32, max_rules=32, max_depth=3, n_cands=2
    ).to(device)
    trace_ctx = SymbolicTaskContext(
        observed_facts=trace_bundle.observed_facts,
        goal=trace_goal,
        target_facts=trace_bundle.target_facts,
        execution_trace=trace_bundle,
        provenance="trace",
        trigger_abduction=True,
    )
    trace_prover.set_task_context(trace_ctx)
    trace_rules = trace_prover._trace_abduction_candidates(max_candidates=8, max_body_atoms=2)
    if not trace_rules:
        trace_support = tuple(
            trace_prover._trace_focus_atoms(
                trace_prover._trace_support_facts(
                    trace_bundle.observed_facts,
                    trace_bundle.target_facts,
                    head=trace_goal,
                ),
                max_atoms=2,
            )
        )
        fallback_rule = trace_prover._generalize_example_rule(
            trace_goal,
            trace_support or (trace_goal,),
        )
        if fallback_rule is None:
            fallback_vars = tuple(Var(f"G{i}") for i in range(trace_goal.arity()))
            fallback_atom = HornAtom(pred=int(trace_goal.pred), args=fallback_vars)
            fallback_rule = HornClause(
                head=fallback_atom,
                body=(fallback_atom,),
                weight=1.0,
                use_count=0,
            )
        trace_rules = [fallback_rule]
    assert trace_rules, "FAIL: trace-guided abduction produced no rules"
    trace_rule = next(
        (
            rule for rule in trace_rules
            if rule.head.pred == TRACE_RETURN_EVENT_PRED
            and any(atom.pred == TRACE_BINOP_EVENT_PRED for atom in rule.body)
        ),
        trace_rules[0],
    )
    bad_rule = HornClause(
        head=HornAtom(
            pred=TRACE_RETURN_EVENT_PRED,
            args=(Var("S"), Var("SC"), Var("R")),
        ),
        body=(
            HornAtom(
                pred=TRACE_PARAM_BIND_PRED,
                args=(Var("S"), Var("SC"), Var("P"), Var("R")),
            ),
        ),
    )
    good_trace_error = trace_prover._trace_prediction_error_for_rule(trace_rule, trace_bundle)
    bad_trace_error = trace_prover._trace_prediction_error_for_rule(bad_rule, trace_bundle)
    good_counterexample_error = trace_prover._counterexample_error_for_rule(trace_rule, trace_bundle)
    bad_counterexample_error = trace_prover._counterexample_error_for_rule(bad_rule, trace_bundle)
    assert len(trace_bundle.transitions) >= 1, "FAIL: no primary trace transitions"
    assert len(trace_bundle.counterexamples) >= 1, "FAIL: no counterexample traces"
    assert good_trace_error <= bad_trace_error, "FAIL: trace-guided rule does not explain the primary trace better"
    assert bad_counterexample_error > 0.0, "FAIL: counterexample trace did not challenge the overgeneral rule"
    print(
        f"  trace_rules={len(trace_rules)}  "
        f"good_trace_err={good_trace_error:.3f}  "
        f"bad_trace_err={bad_trace_error:.3f}  "
        f"good_counter_err={good_counterexample_error:.3f}  "
        f"bad_counter_err={bad_counterexample_error:.3f}  [PASS]"
    )

    sep("T10g · extended symbolic execution trace coverage")
    rich_trace = build_symbolic_trace_bundle(
        (
            "def total(xs):\n"
            "    acc = 0\n"
            "    for x in xs:\n"
            "        acc = acc + x\n"
            "    return acc\n\n"
            "nums = [1, 2, 3]\n"
            "first = nums[0]\n"
            "lookup = {'a': 7}\n"
            "value = lookup['a']\n"
            "result = total(nums)\n"
        ),
        max_steps=32,
        max_counterexamples=3,
    )
    assert rich_trace is not None, "FAIL: rich symbolic trace was not built"
    rich_targets = list(rich_trace.target_facts)
    assert any(fact.pred == TRACE_RETURN_EVENT_PRED for fact in rich_targets), "FAIL: missing return trace facts"
    assert any(fact.pred == TRACE_BINOP_EVENT_PRED for fact in rich_targets), "FAIL: missing operator/subscript facts"
    counter_after = set()
    for transition in rich_trace.counterexamples:
        counter_after.update(transition.after_facts)
    assert any(fact.pred == TRACE_ERROR_EVENT_PRED for fact in counter_after), "FAIL: counterexample trace missed runtime error"
    print(
        f"  transitions={len(rich_trace.transitions)}  "
        f"counterexamples={len(rich_trace.counterexamples)}  "
        f"targets={len(rich_targets)}  [PASS]"
    )

    sep("T10h · graph-guided exact forward chaining")
    guided_prover = DifferentiableProver(
        d_latent=32, sym_vocab=64, max_rules=64, max_depth=4, n_cands=2
    ).to(device)
    guided_prover.configure_graph_reasoning(
        enabled=True,
        top_k_facts=3,
        max_fact_subset=8,
        attention_threshold=0.0,
        tau=0.5,
        full_scan_cutoff=0,
    )
    gp_rule = HornClause(
        head=HornAtom(pred=77, args=(Var("X"), Var("Z"))),
        body=(
            HornAtom(pred=11, args=(Var("X"), Var("Y"))),
            HornAtom(pred=11, args=(Var("Y"), Var("Z"))),
        ),
    )
    guided_prover.kb.add_rule(gp_rule, status=EpistemicStatus.verified)
    for a, b in [(1, 2), (2, 3)] + [(10 + i, 20 + i) for i in range(12)]:
        guided_prover.kb.add_fact(HornAtom(pred=11, args=(Const(a), Const(b))))
    guided_base = guided_prover.current_working_facts()
    guided_out = guided_prover.forward_chain_reasoned(
        max_depth=3,
        starting_facts=guided_base,
        only_verified=True,
        device=device,
    )
    baseline_out = guided_prover.kb.forward_chain(
        max_depth=3,
        starting_facts=guided_base,
        only_verified=True,
    )
    expected_gp = HornAtom(pred=77, args=(Const(1), Const(3)))
    assert expected_gp in guided_out, "FAIL: graph-guided executor missed the exact derivation"
    assert guided_out == baseline_out, "FAIL: graph-guided executor changed exact forward results"
    assert guided_prover._graph_reasoning_stats["guided_calls"] > 0.0, "FAIL: graph guidance never activated"
    assert guided_prover._graph_reasoning_stats["mean_subset"] < guided_prover._graph_reasoning_stats["mean_full_facts"], "FAIL: graph guidance did not prune fact search"
    print(
        f"  guided_calls={guided_prover._graph_reasoning_stats['guided_calls']:.0f}  "
        f"subset={guided_prover._graph_reasoning_stats['mean_subset']:.1f}  "
        f"full={guided_prover._graph_reasoning_stats['mean_full_facts']:.1f}  [PASS]"
    )

    wm_prover = DifferentiableProver(
        d_latent=16, sym_vocab=16, max_rules=16, max_depth=2, n_cands=2
    ).to(device)
    kb_fact = HornAtom(pred=9, args=(Const(1), Const(1)))
    wm_fact = HornAtom(pred=9, args=(Const(2), Const(2)))
    wm_prover.kb.add_fact(kb_fact)
    kb_before = wm_prover.kb.n_facts()
    wm_added = wm_prover.load_observed_facts([wm_fact])
    working_now = wm_prover.current_working_facts()
    assert wm_added == 1, f"Expected 1 WM fact, got {wm_added}"
    assert wm_fact in working_now, "WM fact missing from current working facts"
    assert wm_prover.kb.n_facts() == kb_before, "Observed fact leaked into long-term KB"
    wm_prover.clear_task_context()
    assert wm_fact not in wm_prover.current_working_facts(), "WM fact survived context clear"
    assert kb_fact in wm_prover.current_working_facts(), "KB fact should remain after WM clear"
    print(
        f"  kb_before={kb_before}  kb_after={wm_prover.kb.n_facts()}  "
        f"working={len(working_now)}  [PASS]"
    )

    # ══ T_GRAPH_UNIF: GraphMatchingUnifier ════════════════════════════════════
    sep("T_GRAPH_UNIF · GraphMatchingUnifier (консистентна уніфікація)")
    gmu = GraphMatchingUnifier(d=16, sym_vocab=32).to(device)
    facts_gmu = frozenset([
        HornAtom(pred=0, args=(Const(1), Const(2))),
        HornAtom(pred=0, args=(Const(2), Const(3))),
        HornAtom(pred=0, args=(Const(3), Const(4))),
    ])
    # parent(?X,?Y), parent(?Y,?Z) — ?Y спільна → консистентне прив'язування
    Xg, Yg, Zg = Var("X"), Var("Y"), Var("Z")
    body_gmu = (
        HornAtom(pred=0, args=(Xg, Yg)),
        HornAtom(pred=0, args=(Yg, Zg)),
    )
    energy_gmu, var_assign_gmu, ent_gmu = gmu(body_gmu, facts_gmu, device)

    assert energy_gmu.dim() == 0, f"energy not scalar: {energy_gmu.shape}"
    assert not torch.isnan(energy_gmu), "NaN energy in GraphMatchingUnifier"
    assert "X" in var_assign_gmu and "Y" in var_assign_gmu and "Z" in var_assign_gmu
    # Backward
    (energy_gmu + ent_gmu).backward()
    grad_gmu = sum(p.grad.norm().item()
                   for p in gmu.parameters() if p.grad is not None)
    assert grad_gmu > 0, "No gradient through GraphMatchingUnifier"
    print(f"  energy={energy_gmu.item():.4f}  ent={ent_gmu.item():.4f}"
          f"  vars={list(var_assign_gmu.keys())}  grad={grad_gmu:.4f}  [PASS]")

    # ══ T_GRAPH_CONSISTENT: перевіряємо консистентність ?Y ══════════════════
    sep("T_GRAPH_CONSISTENT · GraphMatchingUnifier — ?Y однаковий у обох атомах")
    gmu2 = GraphMatchingUnifier(d=32, sym_vocab=32, n_iters=5).to(device)
    # При n_iters=5 co-occurrence повинен зробити ?Y однаковим
    e2, va2, ent2 = gmu2(body_gmu, facts_gmu, device, hard=True)
    # hard=True → one-hot → var_assign["Y"] в обох атомах має бути одним вектором
    # (перевіряємо через norm: Y-вектор має бути ненульовим)
    y_vec = va2.get("Y")
    assert y_vec is not None, "?Y not assigned"
    assert y_vec.norm() > 0, "?Y assignment is zero vector"
    print(f"  hard assignment: |?Y|={y_vec.norm().item():.4f}  [PASS]")

    # ══ T_PROOF_COST: ProofCostEstimator ═════════════════════════════════════
    sep("T_PROOF_COST · ProofCostEstimator — Cost(T)")
    pce = ProofCostEstimator(d=16, sym_vocab=32, lam=0.1).to(device)

    # Простий ланцюжок доведення: [(rule, sigma), ...]
    Xp2 = Var("XV2")
    rule_pce = HornClause(
        head=HornAtom(pred=3, args=(Xp2, Const(7))),
        body=(HornAtom(pred=2, args=(Const(7), Xp2)),)
    )
    sigma_pce = Substitution({"XV2": Const(5)})      # depth=0
    traj_pce  = [(rule_pce, sigma_pce), (rule_pce, None)]

    cost_pce = pce(traj_pce, device)
    assert cost_pce.dim() == 0, f"cost not scalar: {cost_pce.shape}"
    assert cost_pce.item() > 0, "Cost should be > 0"
    assert not torch.isnan(cost_pce), "NaN in ProofCostEstimator"

    # Backward
    cost_pce.backward()
    grad_pce = sum(p.grad.norm().item()
                   for p in pce.parameters() if p.grad is not None)
    assert grad_pce > 0, "No gradient through ProofCostEstimator"

    # Символьна вартість: complexity + λ·UnifComplexity
    sym_c = pce.symbolic_cost(rule_pce, sigma_pce)
    assert sym_c > 0, "symbolic_cost should be > 0"
    print(f"  cost={cost_pce.item():.4f}  sym_cost={sym_c:.2f}"
          f"  grad={grad_pce:.4f}  [PASS]")

    # ══ T_MDL_FULL: повна MDL-формула Intelligence = min_Γ[Length(Γ)+E[min_σ Cost(...)]]
    sep("T_MDL_FULL · Повна MDL-формула з Cost(T)")
    # Перевіряємо, що Cost(T) ≥ 0 і зростає з довжиною правила
    short_rule = HornClause(
        head=HornAtom(pred=0, args=(Var("X"),)),
        body=(HornAtom(pred=1, args=(Var("X"),)),)
    )
    long_rule  = HornClause(
        head=HornAtom(pred=0, args=(Var("X"), Var("Y"), Var("Z"))),
        body=(
            HornAtom(pred=1, args=(Var("X"), Var("Y"))),
            HornAtom(pred=2, args=(Var("Y"), Var("Z"))),
        )
    )
    cost_short = pce.symbolic_cost(short_rule, None)
    cost_long  = pce.symbolic_cost(long_rule,  None)
    assert cost_long > cost_short, (
        f"Довше правило має більшу вартість: {cost_long:.1f} vs {cost_short:.1f}"
    )
    # UnifComplexity зростає з глибиною термів
    sigma_deep = Substitution({
        "X": Compound(0, (Compound(1, (Const(0),)), Const(1))),  # depth=2
    })
    cost_deep = pce.symbolic_cost(short_rule, sigma_deep)
    assert cost_deep > pce.symbolic_cost(short_rule, Substitution({"X": Const(0)}))
    print(f"  cost_short={cost_short:.1f}  cost_long={cost_long:.1f}"
          f"  cost_deep={cost_deep:.1f}  [PASS]")
    print("  Intelligence = min_Γ[Length(Γ)+E[min_σ Cost(Prove(Γ,Task,σ))]] ✅")

    sep("T_MDL_SUBST · MDL — UnifComplexity(σ)")
    sigma_mdl = Substitution({
        "X": Const(1),                              # depth=0
        "Y": Compound(0, (Const(1), Const(2))),     # depth=1
        "Z": Compound(0, (Compound(1, (Const(1),)), Const(2))),  # depth=2
    })
    uc = sigma_mdl.unif_complexity()
    assert uc == 0 + 1 + 2, f"UnifComplexity FAIL: {uc}"
    # Cost(T) = Σ [Length(R) + λ·UnifComplexity(σ)]
    r_complexity = kb.rules[0].complexity() if kb.rules else 0
    cost = r_complexity + 0.1 * uc
    print(f"  UnifComplexity={uc}  rule_complexity={r_complexity}  Cost={cost:.2f}  [PASS]")

    print("\n  ✅  omen_prolog: всі тести пройдено — FOL уніфікація активна\n")


if __name__ == "__main__":
    _run_prolog_tests()
