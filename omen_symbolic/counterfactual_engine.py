"""
counterfactual_engine.py — Counterfactual World Engine (CWE)

Повна реалізація концепції:
  · Повний набір операторів Invert(Γ_mod):
      - swap_args          : A(X,Y)      → A(Y,X)          (симетрія)
      - complement_head    : A ← B       → ¬A ← B          (через complement-предикат)
      - drop_body_literal  : A ← B,C     → A ← B           (ослаблення умови)
      - add_body_literal   : A ← B       → A ← B,C         (посилення умови)
      - swap_head_pred     : A(X) ← B    → A'(X) ← B       (заміна висновку)
      - permute_body       : A ← B,C     → A ← C,B         (переставлення тіла)

  · Greedy argmin оптимізація підмножини Γ_mod ⊆ Γ:
      score(Γ_mod) = Complexity(Γ_mod) − λ · Surprise(ForwardChain(Γ~))
      де Complexity(Γ_mod) = Σ MDL_bits(R), Surprise = |novel| + λ·|contradictions|.
      Жадібно нарощуємо Γ_mod, кожен крок обираючи трансформацію з найкращим
      покращенням score, аж до max_rule_mods кроків.

  · Complement-предикат (¬A ← B):
      COMPLEMENT_OFFSET = 600_000.  ¬p(X) реалізується як предикат
      (600_000 + p)(X).  Суперечність виявляється коли одночасно виводяться
      p(X) і (600_000+p)(X) з однаковими аргументами.

Математична модель:
  Γ~ = (Γ \ Γ_mod) ∪ Invert(Γ_mod)
  Γ_mod* = argmin_{Γ_mod ⊆ Γ} [Complexity(Γ_mod) − λ · Surprise(ForwardChain(Γ~))]
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable, Dict, FrozenSet, Iterator, List, Optional, Sequence, Tuple

import torch

from omen_symbolic.creative_types import CounterfactualResult, RuleCandidate

# Предикат-complement для реалізації ¬A ← B
COMPLEMENT_OFFSET: int = 600_000


# ─── Дескриптор однієї атомарної трансформації ───────────────────────────────

@dataclass
class _Transform:
    """Одна атомарна трансформація правила clause → result."""
    original: Any          # оригінальне правило з Γ
    result: Any            # трансформоване правило, що входить в Invert(Γ_mod)
    op: str                # ім'я оператора
    mdl_cost: float        # MDL bits оригінального правила (= Complexity вкладу в Γ_mod)


# ─── Допоміжні функції для побудови правил ────────────────────────────────────

def _make_clause(template: Any, head: Any, body: Tuple[Any, ...]) -> Any:
    return type(template)(
        head=head,
        body=body,
        weight=float(getattr(template, "weight", 1.0)),
        use_count=int(getattr(template, "use_count", 0)),
    )


def _make_atom(template: Any, pred: int, args: Tuple[Any, ...]) -> Any:
    return type(template)(pred=int(pred), args=args)


def _is_complement_conflict(left: Any, right: Any) -> bool:
    left_pred = int(getattr(left, "pred", -1))
    right_pred = int(getattr(right, "pred", -1))
    left_args = tuple(getattr(left, "args", ()))
    right_args = tuple(getattr(right, "args", ()))
    if left_pred >= COMPLEMENT_OFFSET and right_pred == left_pred - COMPLEMENT_OFFSET:
        return left_args == right_args
    if right_pred >= COMPLEMENT_OFFSET and left_pred == right_pred - COMPLEMENT_OFFSET:
        return left_args == right_args
    return False


def _fact_signature(fact: Any) -> Tuple[int, Tuple[Any, ...]]:
    return int(getattr(fact, "pred", -1)), tuple(getattr(fact, "args", ()))


def _complement_signature(
    pred: int,
    args: Tuple[Any, ...],
) -> Optional[Tuple[int, Tuple[Any, ...]]]:
    if pred < 0:
        return None
    if pred >= COMPLEMENT_OFFSET:
        return pred - COMPLEMENT_OFFSET, args
    return pred + COMPLEMENT_OFFSET, args


def _build_fact_index(facts: Sequence[Any]) -> Dict[Tuple[int, Tuple[Any, ...]], Any]:
    return {_fact_signature(fact): fact for fact in facts}


# ─── Повний набір операторів Invert ──────────────────────────────────────────

def _op_swap_args(clause: Any) -> Iterator[Any]:
    """Переставляє аргументи голови у зворотному порядку: A(X,Y) → A(Y,X)."""
    args = tuple(getattr(clause.head, "args", ()))
    if len(args) < 2:
        return
    swapped = tuple(reversed(args))
    if swapped == args:
        return
    new_head = _make_atom(clause.head, int(clause.head.pred), swapped)
    yield _make_clause(clause, new_head, tuple(clause.body))


def _op_complement_head(clause: Any) -> Iterator[Any]:
    """
    ¬A ← B: замінює голову на complement-предикат (COMPLEMENT_OFFSET + pred).
    Реалізує трансформацію A ← B  →  ¬A ← B із концепції.
    """
    pred = int(clause.head.pred)
    if pred >= COMPLEMENT_OFFSET:
        return  # вже complement — не вкладаємо двічі
    comp_pred = COMPLEMENT_OFFSET + pred
    new_head = _make_atom(clause.head, comp_pred, tuple(clause.head.args))
    yield _make_clause(clause, new_head, tuple(clause.body))


def _op_drop_body_literal(clause: Any) -> Iterator[Any]:
    """A ← B,C  →  A ← B: прибирає кожен літерал тіла по черзі."""
    body = tuple(getattr(clause, "body", ()))
    if len(body) < 2:
        return
    for i in range(len(body)):
        new_body = body[:i] + body[i + 1:]
        yield _make_clause(clause, clause.head, new_body)


def _op_add_body_literal(clause: Any, pool_atoms: Sequence[Any]) -> Iterator[Any]:
    """A ← B  →  A ← B,C: додає атом із пулу до тіла (посилення умови)."""
    body = tuple(getattr(clause, "body", ()))
    if len(body) >= 3:
        return
    head_pred = int(clause.head.pred)
    seen: set = set()
    for atom in pool_atoms:
        atom_pred = int(getattr(atom, "pred", -1))
        if atom_pred == head_pred or atom_pred in seen:
            continue
        seen.add(atom_pred)
        new_body = body + (atom,)
        yield _make_clause(clause, clause.head, new_body)


def _op_swap_head_pred(clause: Any, alt_preds: Sequence[int]) -> Iterator[Any]:
    """A(X) ← B  →  A'(X) ← B: замінює предикат голови на alt_pred з тієї ж арності."""
    head_pred = int(clause.head.pred)
    arity = len(tuple(getattr(clause.head, "args", ())))
    for alt in alt_preds:
        alt = int(alt)
        if alt == head_pred:
            continue
        new_head = _make_atom(clause.head, alt, tuple(clause.head.args))
        yield _make_clause(clause, new_head, tuple(clause.body))


def _op_permute_body(clause: Any) -> Iterator[Any]:
    """A ← B,C  →  A ← C,B: циклічний зсув тіла."""
    body = tuple(getattr(clause, "body", ()))
    if len(body) < 2:
        return
    rotated = body[1:] + body[:1]
    if rotated != body:
        yield _make_clause(clause, clause.head, rotated)


# ─── Головний клас ────────────────────────────────────────────────────────────

class CounterfactualWorldEngine:
    """
    Генератор 'неможливих' гіпотез через argmin-оптимізацію підмножини Γ_mod ⊆ Γ.

    Параметри
    ----------
    max_rule_mods : int
        Максимальна кількість правил у Γ_mod (розмір підмножини).
        Жадібний пошук зупиняється після max_rule_mods кроків.
    max_candidates : int
        Скільки правил з Γ розглядати як кандидати для включення в Γ_mod.
    surprise_lambda : float
        λ у формулі: score = Complexity(Γ_mod) − λ · Surprise(Γ~).
        Більший λ — агресивніший пошук суперечностей.
    max_transforms_per_rule : int
        Максимальна кількість трансформацій на одне правило.
    """

    def __init__(
        self,
        max_rule_mods: int = 2,
        max_candidates: int = 8,
        surprise_lambda: float = 0.5,
        max_transforms_per_rule: int = 4,
    ):
        self.max_rule_mods = max(int(max_rule_mods), 1)
        self.max_candidates = max(int(max_candidates), 1)
        self.surprise_lambda = float(max(surprise_lambda, 1e-6))
        self.max_transforms_per_rule = max(int(max_transforms_per_rule), 1)

    # ─── Клонування KB у sandbox ─────────────────────────────────────────────

    @staticmethod
    def _temporary_sandbox_device(kb: Any) -> Optional[torch.device]:
        # Temporary counterfactual evaluators are tiny; CPU avoids repeated GPU micro-kernel overhead.
        if hasattr(kb, "_max_facts") and hasattr(kb, "_dev"):
            return torch.device("cpu")
        return getattr(kb, "_dev", getattr(kb, "device", None))

    @classmethod
    def _make_empty_sandbox(cls, kb: Any) -> Any:
        max_rules = max(int(getattr(kb, "max_rules", 1024)), 32)
        max_facts = max(int(getattr(kb, "_max_facts", 32)), 32)
        device = cls._temporary_sandbox_device(kb)
        constructor_variants = (
            {"max_rules": max_rules, "max_facts": max_facts, "device": device},
            {"max_rules": max_rules, "max_facts": max_facts},
            {"max_rules": max_rules, "device": device},
            {"max_rules": max_rules},
        )
        last_error: Optional[Exception] = None
        for kwargs in constructor_variants:
            try:
                return type(kb)(**kwargs)
            except TypeError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise TypeError(f"Unable to instantiate sandbox for {type(kb).__name__}")

    @staticmethod
    def _copy_kb_into_sandbox(kb: Any, sandbox: Any, *, preserve_status: bool) -> None:
        from omen_prolog import EpistemicStatus, RuleRecord

        if (
            hasattr(kb, "_fact_buf")
            and hasattr(kb, "_rule_buf")
            and hasattr(sandbox, "_fact_buf")
            and hasattr(sandbox, "_rule_buf")
        ):
            n_facts = int(getattr(kb, "_n_facts", 0))
            if n_facts > 0:
                sandbox._fact_buf[:n_facts] = kb._fact_buf[:n_facts].to(device=sandbox._dev)
            sandbox._n_facts = n_facts
            sandbox._fact_set = set(getattr(kb, "_fact_set", set()))
            sandbox._extra_facts = set(getattr(kb, "_extra_facts", set()))
            sandbox._term_to_id = dict(getattr(kb, "_term_to_id", {}))
            sandbox._id_to_term = dict(getattr(kb, "_id_to_term", {}))
            sandbox._next_term_id = int(getattr(kb, "_next_term_id", 1))
            sandbox._facts_cache = None
            sandbox._facts_cache_n = -1

            rules = list(getattr(kb, "rules", ()))
            n_rules = int(getattr(kb, "_n_rules", len(rules)))
            if n_rules > 0:
                sandbox._rule_buf[:n_rules] = kb._rule_buf[:n_rules].to(device=sandbox._dev)
                sandbox._rule_is_tensor[:n_rules] = kb._rule_is_tensor[:n_rules].to(device=sandbox._dev)
            sandbox._n_rules = n_rules
            sandbox.rules = list(rules)
            sandbox._rule_hash_set = set(getattr(kb, "_rule_hash_set", set()))
            sandbox._global_step = int(getattr(kb, "_global_step", 0))
            status_by_hash = {
                key: getattr(record, "status", EpistemicStatus.proposed)
                for key, record in getattr(kb, "_records", {}).items()
            }
            if preserve_status:
                sandbox._records = {
                    hash(rule): RuleRecord(rule=rule, status=status_by_hash.get(hash(rule), EpistemicStatus.proposed))
                    for rule in sandbox.rules
                }
            else:
                sandbox._records = {
                    hash(rule): RuleRecord(rule=rule, status=EpistemicStatus.proposed)
                    for rule in sandbox.rules
                }
            sandbox._n_proposed = sum(
                1 for record in sandbox._records.values() if record.status == EpistemicStatus.proposed
            )
            sandbox._n_verified = sum(
                1 for record in sandbox._records.values() if record.status == EpistemicStatus.verified
            )
            return

        facts = frozenset(getattr(kb, "facts", frozenset()))
        rules = list(getattr(kb, "rules", ()))
        sandbox.facts = facts
        sandbox.rules = list(rules)
        sandbox._rule_set = set(getattr(kb, "_rule_set", {hash(rule) for rule in rules}))
        sandbox._global_step = int(getattr(kb, "_global_step", 0))
        status_by_hash = {
            key: getattr(record, "status", EpistemicStatus.proposed)
            for key, record in getattr(kb, "_records", {}).items()
        }
        if preserve_status:
            sandbox._records = {
                hash(rule): RuleRecord(rule=rule, status=status_by_hash.get(hash(rule), EpistemicStatus.proposed))
                for rule in rules
            }
        else:
            sandbox._records = {
                hash(rule): RuleRecord(rule=rule, status=EpistemicStatus.proposed)
                for rule in rules
            }

    @staticmethod
    def _clone_kb(kb: Any) -> Any:
        """Повністю клонує KB: факти + правила зі статусами."""
        clone = CounterfactualWorldEngine._make_empty_sandbox(kb)
        CounterfactualWorldEngine._copy_kb_into_sandbox(kb, clone, preserve_status=True)
        return clone

    # ─── Побудова sandbox KB з Γ_mod ─────────────────────────────────────────

    @staticmethod
    def _build_sandbox(kb: Any, gamma_mod: Sequence[_Transform]) -> Any:
        """
        Γ~ = (Γ \ Γ_mod) ∪ Invert(Γ_mod).
        Видаляє оригінали і додає трансформовані правила.
        """
        # Клонуємо базову KB
        sandbox = CounterfactualWorldEngine._make_empty_sandbox(kb)
        CounterfactualWorldEngine._copy_kb_into_sandbox(kb, sandbox, preserve_status=True)

        # Множина оригіналів, що виключаються (Γ \ Γ_mod)
        excluded: FrozenSet[int] = frozenset(hash(t.original) for t in gamma_mod)

        if excluded:
            kept_rules = [rule for rule in sandbox.rules if hash(rule) not in excluded]
            sandbox.rules = kept_rules
            if hasattr(sandbox, "_rule_hash_set"):
                sandbox._rule_hash_set = {hash(rule) for rule in kept_rules}
            if hasattr(sandbox, "_rule_set"):
                sandbox._rule_set = {hash(rule) for rule in kept_rules}
            sandbox._records = {
                hash(rule): sandbox._records[hash(rule)]
                for rule in kept_rules
                if hash(rule) in sandbox._records
            }
            if hasattr(sandbox, "_rule_buf") and hasattr(sandbox, "_rule_is_tensor"):
                kept_indices = [
                    idx for idx, rule in enumerate(getattr(kb, "rules", ()))
                    if hash(rule) not in excluded
                ]
                n_kept = len(kept_indices)
                if n_kept > 0:
                    idx_t = torch.tensor(kept_indices, dtype=torch.long, device=sandbox._dev)
                    sandbox._rule_buf[:n_kept] = sandbox._rule_buf[idx_t].clone()
                    sandbox._rule_is_tensor[:n_kept] = sandbox._rule_is_tensor[idx_t].clone()
                sandbox._rule_buf[n_kept:sandbox._n_rules] = sandbox.NONE
                sandbox._rule_is_tensor[n_kept:sandbox._n_rules] = False
                sandbox._n_rules = n_kept
            if hasattr(sandbox, "_n_proposed"):
                sandbox._n_proposed = sum(
                    1 for record in sandbox._records.values() if record.status.value == "proposed"
                )
            if hasattr(sandbox, "_n_verified"):
                sandbox._n_verified = sum(
                    1 for record in sandbox._records.values() if record.status.value == "verified"
                )

        # Додаємо Invert(Γ_mod)
        for t in gamma_mod:
            sandbox.add_rule(t.result)

        return sandbox

    # ─── Вимірювання Surprise і Complexity ───────────────────────────────────

    @staticmethod
    def _measure_surprise(
        sandbox: Any,
        baseline: FrozenSet[Any],
        starting_facts: FrozenSet[Any],
        max_depth: int,
        conflict_fn: Callable[[Any, Any], bool],
        *,
        baseline_list: Optional[Tuple[Any, ...]] = None,
        baseline_index: Optional[Dict[Tuple[int, Tuple[Any, ...]], Any]] = None,
    ) -> Tuple[float, Tuple[Any, ...], List[Tuple[Any, Any]]]:
        """
        Повертає (surprise_score, novel_facts, contradictions).
        Surprise = |novel| + λ·|contradictions|  (λ застосується в _score).
        """
        derived = sandbox.forward_chain(
            max(int(max_depth), 1),
            starting_facts=starting_facts,
            only_verified=False,
            track_epistemic=False,
        )
        novel = tuple(sorted(derived - baseline, key=hash))
        derived_list = list(derived)
        derived_index = _build_fact_index(derived_list)
        baseline_list = tuple(baseline) if baseline_list is None else baseline_list
        baseline_index = _build_fact_index(baseline_list) if baseline_index is None else baseline_index

        # Complement-суперечності: p(X) та ¬p(X) як усередині sandbox,
        # так і відносно базового виводу Γ.
        complement_pairs: Dict[Tuple[int, int], Tuple[Any, Any]] = {}
        for (pred, args), fact in derived_index.items():
            complement_key = _complement_signature(pred, args)
            if complement_key is None:
                continue
            known = derived_index.get(complement_key)
            if known is not None:
                key = tuple(sorted((hash(fact), hash(known))))
                complement_pairs[key] = (fact, known)
            known = baseline_index.get(complement_key)
            if known is not None:
                key = tuple(sorted((hash(fact), hash(known))))
                complement_pairs[key] = (fact, known)
        complement_contradictions = list(complement_pairs.values())

        # Класичні суперечності через conflict_fn
        classic_contradictions: List[Tuple[Any, Any]] = []
        for fact in novel:
            for known in baseline_list:
                if conflict_fn(fact, known):
                    classic_contradictions.append((fact, known))

        all_contradictions = complement_contradictions + classic_contradictions
        return float(len(novel)), tuple(novel), all_contradictions

    @staticmethod
    def _complexity(gamma_mod: Sequence[_Transform]) -> float:
        """Complexity(Γ_mod) = Σ MDL_bits(оригінального правила)."""
        return sum(t.mdl_cost for t in gamma_mod)

    def _score(
        self,
        gamma_mod: Sequence[_Transform],
        novel_count: float,
        contradiction_count: int,
        world_surprise: float = 0.0,
    ) -> float:
        """
        score = Complexity(Γ_mod) − λ · Surprise(Γ~)
        Мінімізуємо: мала складність + велика surprise = хороший результат.
        """
        surprise = novel_count + float(contradiction_count) + max(float(world_surprise), 0.0)
        complexity = self._complexity(gamma_mod)
        return complexity - self.surprise_lambda * surprise

    # ─── Генерація всіх атомарних трансформацій ──────────────────────────────

    def _all_transforms(
        self,
        rules: Sequence[Any],
        pool_atoms: Sequence[Any],
        alt_pred_map: Dict[int, List[int]],
    ) -> List[_Transform]:
        """
        Генерує всі атомарні трансформації для кожного правила.
        Повертає відсортований список _Transform за зростанням MDL (дешеві — перші).
        """
        transforms: List[_Transform] = []
        seen_results: set = set()

        for clause in rules:
            mdl = float(getattr(clause, "description_length_bits", lambda: 32.0)())
            candidates_for_rule: List[Any] = []

            # 1. Swap args
            candidates_for_rule.extend(_op_swap_args(clause))
            # 2. Complement head (¬A ← B)
            candidates_for_rule.extend(_op_complement_head(clause))
            # 3. Drop body literal
            candidates_for_rule.extend(_op_drop_body_literal(clause))
            # 4. Add body literal
            candidates_for_rule.extend(_op_add_body_literal(clause, pool_atoms))
            # 5. Swap head predicate
            arity = len(tuple(getattr(clause.head, "args", ())))
            alts = alt_pred_map.get(arity, [])
            candidates_for_rule.extend(_op_swap_head_pred(clause, alts))
            # 6. Permute body
            candidates_for_rule.extend(_op_permute_body(clause))

            for result in candidates_for_rule[: self.max_transforms_per_rule]:
                key = hash(result)
                if key in seen_results or hash(result) == hash(clause):
                    continue
                seen_results.add(key)
                transforms.append(_Transform(
                    original=clause,
                    result=result,
                    op=_detect_op(clause, result),
                    mdl_cost=mdl,
                ))

        # Сортуємо за MDL (дешеві модифікації перші — мінімальна складність)
        transforms.sort(key=lambda t: t.mdl_cost)
        return transforms

    # ─── Жадібний пошук оптимального Γ_mod ──────────────────────────────────

    def _evaluate_subset(
        self,
        kb: Any,
        gamma_mod: Sequence[_Transform],
        baseline: FrozenSet[Any],
        baseline_list: Tuple[Any, ...],
        baseline_index: Dict[Tuple[int, Tuple[Any, ...]], Any],
        starting_facts: FrozenSet[Any],
        max_depth: int,
        conflict_fn: Callable[[Any, Any], bool],
        world_surprise_fn: Optional[Callable[[Sequence[Any], Tuple[Any, ...], Any], float]] = None,
    ) -> Tuple[float, Tuple[Any, ...], List[Tuple[Any, Any]], float]:
        sandbox = self._build_sandbox(kb, gamma_mod)
        novel_count, novel, contras = self._measure_surprise(
            sandbox,
            baseline,
            starting_facts,
            max_depth,
            conflict_fn,
            baseline_list=baseline_list,
            baseline_index=baseline_index,
        )
        world_surprise = 0.0
        if world_surprise_fn is not None:
            try:
                world_surprise = float(
                    world_surprise_fn(tuple(t.result for t in gamma_mod), tuple(novel), sandbox)
                )
            except Exception:
                world_surprise = 0.0
        score = self._score(gamma_mod, novel_count, len(contras), world_surprise=world_surprise)
        return score, tuple(novel), list(contras), world_surprise

    def _bounded_exact_search(
        self,
        kb: Any,
        transforms: Sequence[_Transform],
        baseline: FrozenSet[Any],
        starting_facts: FrozenSet[Any],
        max_depth: int,
        conflict_fn: Callable[[Any, Any], bool],
        world_surprise_fn: Optional[Callable[[Sequence[Any], Tuple[Any, ...], Any], float]] = None,
    ) -> Tuple[List[_Transform], float, Tuple[Any, ...], List[Tuple[Any, Any]], float, int]:
        """
        Greedy argmin пошук Γ_mod* ⊆ Γ.

        Алгоритм:
          1. Починаємо з Γ_mod = {} (порожня підмножина).
          2. На кожному кроці (до max_rule_mods ітерацій):
             a. Для кожної трансформації t ∉ Γ_mod, обчислюємо
                score(Γ_mod ∪ {t}) = Complexity(Γ_mod ∪ {t}) − λ·Surprise(Γ~∪{t}).
             b. Обираємо t* = argmin score.
             c. Якщо score(Γ_mod ∪ {t*}) < score(Γ_mod): додаємо t* і продовжуємо.
                Інакше: зупиняємось (немає покращення).
          3. Повертаємо найкращий Γ_mod і відповідний результат.
        """
        candidate_pool = list(transforms[: self.max_candidates])
        gamma_mod: List[_Transform] = []
        best_score = float("inf")
        best_novel: Tuple[Any, ...] = ()
        best_contradictions: List[Tuple[Any, Any]] = []
        best_world_surprise: float = 0.0
        evaluated_subsets = 0
        baseline_list = tuple(baseline)
        baseline_index = _build_fact_index(baseline_list)

        for subset_size in range(1, min(self.max_rule_mods, len(candidate_pool)) + 1):
            for subset in combinations(candidate_pool, subset_size):
                if len({hash(t.original) for t in subset}) != subset_size:
                    continue
                evaluated_subsets += 1
                score, novel, contras, world_surprise = self._evaluate_subset(
                    kb,
                    subset,
                    baseline,
                    baseline_list,
                    baseline_index,
                    starting_facts,
                    max_depth,
                    conflict_fn,
                    world_surprise_fn=world_surprise_fn,
                )
                if score < best_score - 1e-9:
                    gamma_mod = list(subset)
                    best_score = score
                    best_novel = novel
                    best_contradictions = contras
                    best_world_surprise = world_surprise
                elif abs(score - best_score) <= 1e-9 and gamma_mod and subset_size < len(gamma_mod):
                    gamma_mod = list(subset)
                    best_novel = novel
                    best_contradictions = contras
                    best_world_surprise = world_surprise

        return gamma_mod, best_score, best_novel, best_contradictions, best_world_surprise, evaluated_subsets

        # Множина вже використаних оригінальних правил
        used_originals: set = set()

        for _step in range(self.max_rule_mods):
            step_best_score = best_score
            step_best_transform: Optional[_Transform] = None
            step_best_novel: Tuple[Any, ...] = best_novel
            step_best_contras: List[Tuple[Any, Any]] = best_contradictions
            step_best_world_surprise: float = best_world_surprise

            for t in transforms[: self.max_candidates]:
                orig_hash = hash(t.original)
                if orig_hash in used_originals:
                    continue

                trial_mod = gamma_mod + [t]
                sandbox = self._build_sandbox(kb, trial_mod)
                novel_count, novel, contras = self._measure_surprise(
                    sandbox, baseline, starting_facts, max_depth, conflict_fn
                )
                world_surprise = 0.0
                if world_surprise_fn is not None:
                    try:
                        world_surprise = float(world_surprise_fn(tuple(t.result for t in trial_mod), tuple(novel), sandbox))
                    except Exception:
                        world_surprise = 0.0
                score = self._score(trial_mod, novel_count, len(contras), world_surprise=world_surprise)

                if score < step_best_score:
                    step_best_score = score
                    step_best_transform = t
                    step_best_novel = novel
                    step_best_contras = contras
                    step_best_world_surprise = world_surprise

            if step_best_transform is None:
                break  # немає покращення

            gamma_mod.append(step_best_transform)
            used_originals.add(hash(step_best_transform.original))
            best_score = step_best_score
            best_novel = step_best_novel
            best_contradictions = step_best_contras
            best_world_surprise = step_best_world_surprise

        return gamma_mod, best_score, best_novel, best_contradictions, best_world_surprise

    # ─── Публічний API ───────────────────────────────────────────────────────

    def explore(
        self,
        kb: Any,
        base_facts: Sequence[Any],
        max_depth: int,
        conflict_fn: Callable[[Any, Any], bool],
        world_surprise_fn: Optional[Callable[[Sequence[Any], Tuple[Any, ...], Any], float]] = None,
    ) -> CounterfactualResult:
        """
        Головний метод: знаходить оптимальний Γ_mod* через greedy argmin.

        Повертає CounterfactualResult з кандидатами-правилами (мітка 'counterfactual'),
        novel_facts, contradictions (включаючи complement-суперечності).
        """
        rules = list(getattr(kb, "rules", ()))
        if not rules:
            return CounterfactualResult()

        starting_facts = frozenset(base_facts)
        baseline: FrozenSet[Any] = kb.forward_chain(
            max(int(max_depth), 1),
            starting_facts=starting_facts,
            only_verified=False,
            track_epistemic=False,
        )

        # Пул атомів для _op_add_body_literal
        pool_atoms = _collect_pool_atoms(rules)

        # Карта: arity → [pred_id, ...] для _op_swap_head_pred
        alt_pred_map = _build_arity_pred_map(rules)

        # Всі атомарні трансформації
        transforms = self._all_transforms(rules, pool_atoms, alt_pred_map)
        if not transforms:
            return CounterfactualResult()

        # Жадібний пошук оптимального Γ_mod*
        gamma_mod, best_score, novel, contradictions, world_surprise, evaluated_subsets = self._bounded_exact_search(
            kb,
            transforms,
            baseline,
            starting_facts,
            max_depth,
            conflict_fn,
            world_surprise_fn=world_surprise_fn,
        )

        if not gamma_mod:
            return CounterfactualResult()

        surprise = float(len(novel) + len(contradictions) + max(world_surprise, 0.0))

        candidates = [
            RuleCandidate(
                clause=t.result,
                source="counterfactual",
                score=float(surprise / max(len(gamma_mod), 1)),
                metadata={
                    "modified_rule_hash": float(hash(t.original)),
                    "op": t.op,
                    "complement_offset": float(COMPLEMENT_OFFSET),
                    "gamma_mod_size": float(len(gamma_mod)),
                    "greedy_score": float(best_score),
                    "search_score": float(best_score),
                    "exact_search": 1.0,
                    "evaluated_subsets": float(evaluated_subsets),
                    "world_surprise": float(world_surprise),
                },
            )
            for t in gamma_mod
        ]

        return CounterfactualResult(
            candidates=candidates,
            novel_facts=tuple(novel),
            contradictions=tuple(contradictions),
            modified_rules=tuple(t.result for t in gamma_mod),
            surprise=surprise,
            metadata={
                "novel_facts": float(len(novel)),
                "contradictions": float(len(contradictions)),
                "gamma_mod_size": float(len(gamma_mod)),
                "greedy_score": float(best_score),
                "search_score": float(best_score),
                "exact_search": 1.0,
                "evaluated_subsets": float(evaluated_subsets),
                "world_surprise": float(world_surprise),
                "complement_contradictions": float(
                    sum(1 for f, _ in contradictions if int(getattr(f, "pred", 0)) >= COMPLEMENT_OFFSET)
                ),
            },
        )


# ─── Утиліти ──────────────────────────────────────────────────────────────────

def _collect_pool_atoms(rules: Sequence[Any]) -> List[Any]:
    """Збирає тіла усіх правил у єдиний пул атомів для add_body_literal."""
    seen: set = set()
    pool: List[Any] = []
    for rule in rules:
        for atom in getattr(rule, "body", ()):
            h = hash(atom)
            if h not in seen:
                seen.add(h)
                pool.append(atom)
    return pool


def _build_arity_pred_map(rules: Sequence[Any]) -> Dict[int, List[int]]:
    """Будує відображення arity → [список pred_id] для swap_head_pred."""
    result: Dict[int, List[int]] = {}
    for rule in rules:
        arity = len(tuple(getattr(rule.head, "args", ())))
        pred = int(rule.head.pred)
        if arity not in result:
            result[arity] = []
        if pred not in result[arity]:
            result[arity].append(pred)
    return result


def _detect_op(original: Any, result: Any) -> str:
    """Визначає ім'я застосованого оператора за різницею original → result."""
    orig_pred = int(original.head.pred)
    res_pred = int(result.head.pred)
    if res_pred >= COMPLEMENT_OFFSET and res_pred == COMPLEMENT_OFFSET + orig_pred:
        return "complement_head"
    if res_pred != orig_pred:
        return "swap_head_pred"
    orig_body = tuple(getattr(original, "body", ()))
    res_body = tuple(getattr(result, "body", ()))
    if original.head.args != result.head.args:
        return "swap_args"
    if len(res_body) < len(orig_body):
        return "drop_body_literal"
    if len(res_body) > len(orig_body):
        return "add_body_literal"
    if res_body != orig_body:
        return "permute_body"
    return "unknown"
