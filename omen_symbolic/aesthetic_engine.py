"""
aesthetic_engine.py — Aesthetic Evolution Engine (AEE)

Повна реалізація концепції:
  · Популяція ініціалізується з ДВОХ джерел:
      1. Кандидати від AME/CWE/OEE (зовнішній потік).
      2. Найкращі правила поточного LTM (за естетичною оцінкою) — "генний пул".
  · Персистентний _gene_pool зберігається між викликами evolve() і
    поповнюється щоразу найкращими виживачами.
  · Оператори відбору:
      - Crossover: тіло від R1, голова від R2 (якщо змінні уніфікуються).
      - Mutation: випадкова заміна предиката або додавання/видалення літерала.
      - Selection: U_total(R) = U_VeM(R) + γ·A(R), зберігаються найкращі.
  · Естетична функція A(R):
      A_sym     — симетрія відносно перестановки змінних голови
      A_compr   — MDL компресія (1 / (1 + bits/32))
      A_novel   — 1 − max_{R'∈LTM} sim(R, R')  (унікальність структури)
      A_elegance — utility / MDL (щільність інформації)

Математична модель:
  A(R) = 0.25·A_sym + 0.25·A_compr + 0.25·A_novel + 0.25·A_elegance
  U_total(R) = U_VeM(R) + γ · A(R)
"""

from __future__ import annotations

from collections import Counter
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from omen_symbolic.abduction_search import rule_template_signature
from omen_symbolic.creative_types import RuleCandidate


# ─── Структурна подібність правил ────────────────────────────────────────────

def _rule_similarity(left: Any, right: Any) -> float:
    """
    Структурна подібність між двома правилами [0, 1].
    Враховує: однаковий head.pred, різницю в довжині тіла, перетин предикатів тіла.
    """
    score = 0.0
    score += 1.0 if int(left.head.pred) == int(right.head.pred) else 0.0
    score += 0.25 / (1.0 + abs(len(left.body) - len(right.body)))
    left_preds = {int(atom.pred) for atom in getattr(left, "body", ())}
    right_preds = {int(atom.pred) for atom in getattr(right, "body", ())}
    union = len(left_preds | right_preds)
    overlap = len(left_preds & right_preds)
    if union > 0:
        score += float(overlap) / float(union)
    return max(0.0, min(score / 2.25, 1.0))


def _rule_embedding(
    rule: Any,
    predicate_embeddings: Optional[Dict[int, torch.Tensor]],
) -> Optional[torch.Tensor]:
    if not predicate_embeddings:
        return None
    parts: List[torch.Tensor] = []
    head_pred = int(getattr(getattr(rule, "head", None), "pred", -1))
    if head_pred in predicate_embeddings:
        parts.append(predicate_embeddings[head_pred].detach())
    for atom in getattr(rule, "body", ()):
        pred = int(getattr(atom, "pred", -1))
        if pred in predicate_embeddings:
            parts.append(predicate_embeddings[pred].detach())
    if not parts:
        return None
    return F.normalize(torch.stack(parts, dim=0).mean(dim=0), dim=-1, eps=1e-6)


def _term_signature(term: Any) -> str:
    if hasattr(term, "name") and not hasattr(term, "val"):
        return f"var:{getattr(term, 'name')}"
    if hasattr(term, "val"):
        return f"const:{repr(getattr(term, 'val'))}"
    if hasattr(term, "func") and hasattr(term, "args"):
        parts = ",".join(_term_signature(arg) for arg in getattr(term, "args", ()))
        return f"comp:{getattr(term, 'func')}({parts})"
    return repr(term)


def _atom_positional_signature(atom: Any) -> Tuple[int, Tuple[str, ...]]:
    return (
        int(getattr(atom, "pred", -1)),
        tuple(_term_signature(arg) for arg in getattr(atom, "args", ())),
    )


def _rule_atom_counter(rule: Any) -> Counter:
    return Counter(
        [_atom_positional_signature(getattr(rule, "head", None))]
        + [_atom_positional_signature(atom) for atom in getattr(rule, "body", ())]
    )


def _permute_atom_args(atom: Any, left: int, right: int) -> Any:
    args = list(getattr(atom, "args", ()))
    if max(left, right) >= len(args):
        return atom
    args[left], args[right] = args[right], args[left]
    return type(atom)(pred=int(getattr(atom, "pred", -1)), args=tuple(args))


def _permute_rule_args(rule: Any, left: int, right: int) -> Any:
    head = _permute_atom_args(rule.head, left, right)
    body = tuple(_permute_atom_args(atom, left, right) for atom in getattr(rule, "body", ()))
    return type(rule)(
        head=head,
        body=body,
        weight=float(getattr(rule, "weight", 1.0)),
        use_count=int(getattr(rule, "use_count", 0)),
    )


# ─── Уніфікація змінних для crossover ────────────────────────────────────────

def _var_names(rule: Any) -> set:
    """Збирає всі імена змінних у правилі."""
    names: set = set()
    for atom in (rule.head,) + tuple(getattr(rule, "body", ())):
        for arg in getattr(atom, "args", ()):
            if hasattr(arg, "name") and not hasattr(arg, "val"):
                names.add(str(arg.name))
    return names


def _can_crossover(left: Any, right: Any) -> bool:
    """
    Перевіряє, чи можна схрестити two правила:
    вимагає перетину змінних між тілом left і головою right (змінні уніфікуються).
    """
    left_body_vars = set()
    for atom in getattr(left, "body", ()):
        for arg in getattr(atom, "args", ()):
            if hasattr(arg, "name") and not hasattr(arg, "val"):
                left_body_vars.add(str(arg.name))
    right_head_vars = set()
    for arg in getattr(right.head, "args", ()):
        if hasattr(arg, "name") and not hasattr(arg, "val"):
            right_head_vars.add(str(arg.name))
    # Схрещуємо якщо є хоч якийсь перетин або обидва мають змінні
    return bool(left_body_vars & right_head_vars) or bool(left_body_vars and right_head_vars)


# ─── Головний клас ────────────────────────────────────────────────────────────

class AestheticEvolutionEngine:
    """
    Естетична еволюція правил.

    Параметри
    ----------
    population_size : int
        Розмір популяції (включає і кандидатів, і насіння з LTM).
    generations : int
        Кількість еволюційних поколінь за один виклик evolve().
    gamma : float
        Вага естетичного доданку в U_total = U_VeM + γ·A(R).
    mutation_rate : float
        Ймовірність мутації після crossover.
    crossover_rate : float
        Ймовірність crossover (інакше — лише мутація).
    max_selected : int
        Кількість правил, що повертаються після відбору.
    ltm_seed_ratio : float
        Частка популяції [0,1], що засівається правилами LTM.
    gene_pool_size : int
        Розмір персистентного gene pool між викликами.
    """

    def __init__(
        self,
        population_size: int = 16,
        generations: int = 2,
        gamma: float = 0.25,
        mutation_rate: float = 0.35,
        crossover_rate: float = 0.5,
        max_selected: int = 2,
        ltm_seed_ratio: float = 0.35,
        gene_pool_size: int = 32,
    ):
        self.population_size = max(int(population_size), 4)
        self.generations = max(int(generations), 1)
        self.gamma = float(max(gamma, 0.0))
        self.mutation_rate = float(max(0.0, min(1.0, mutation_rate)))
        self.crossover_rate = float(max(0.0, min(1.0, crossover_rate)))
        self.max_selected = max(int(max_selected), 1)
        self.ltm_seed_ratio = float(max(0.0, min(1.0, ltm_seed_ratio)))
        self.gene_pool_size = max(int(gene_pool_size), self.population_size)

        # Персистентний gene pool між викликами evolve()
        self._gene_pool: List[RuleCandidate] = []

    # ─── Естетична функція A(R) ───────────────────────────────────────────────

    @staticmethod
    def _symmetry_score(rule: Any) -> float:
        """
        A_sym ≈ Corr(R, permute(R)): кореляція правила з аргументно-переставленими
        варіантами. Це ближче до концепції, ніж проста перевірка head[0]==head[-1].
        """
        max_arity = max(
            [len(tuple(getattr(rule.head, "args", ())))]
            + [len(tuple(getattr(atom, "args", ()))) for atom in getattr(rule, "body", ())],
            default=0,
        )
        if max_arity < 2:
            return 0.0
        base = _rule_atom_counter(rule)
        best = 0.0
        for left in range(max_arity):
            for right in range(left + 1, max_arity):
                permuted = _permute_rule_args(rule, left, right)
                permuted_counter = _rule_atom_counter(permuted)
                overlap = sum((base & permuted_counter).values())
                union = sum((base | permuted_counter).values())
                if union > 0:
                    best = max(best, float(overlap) / float(union))
        return float(best)

    @staticmethod
    def _compression_score(rule: Any, compression_delta: Optional[float] = None) -> float:
        """A_compr: якщо доступна оцінка ΔMDL, використовуємо її; інакше fallback по довжині."""
        if compression_delta is not None:
            bits = max(float(rule.description_length_bits()), 1.0)
            scaled = float(compression_delta) / bits
            return float(torch.sigmoid(torch.tensor(scaled, dtype=torch.float32)).item())
        length_bits = float(rule.description_length_bits())
        return 1.0 / (1.0 + length_bits / 32.0)

    @staticmethod
    def _novelty_score(
        rule: Any,
        existing_rules: Sequence[Any],
        predicate_embeddings: Optional[Dict[int, torch.Tensor]] = None,
    ) -> float:
        """
        A_novel = 1 − max_{R'∈LTM} sim(R, R').
        Правило унікальне, якщо воно максимально відрізняється від усіх наявних.
        """
        if not existing_rules:
            return 1.0
        if predicate_embeddings:
            rule_emb = _rule_embedding(rule, predicate_embeddings)
            if rule_emb is not None:
                sims: List[float] = []
                for existing in existing_rules:
                    existing_emb = _rule_embedding(existing, predicate_embeddings)
                    if existing_emb is None:
                        continue
                    sim = F.cosine_similarity(
                        rule_emb.unsqueeze(0),
                        existing_emb.unsqueeze(0),
                        dim=-1,
                    )
                    sims.append(float(sim.item()))
                if sims:
                    return max(0.0, 1.0 - max(max(sims), 0.0))
        return 1.0 - max(_rule_similarity(rule, existing) for existing in existing_rules)

    @staticmethod
    def _elegance_score(rule: Any, utility: float) -> float:
        """A_elegance = utility / MDL: щільність інформації."""
        return float(utility) / max(float(rule.description_length_bits()), 1.0)

    def _aesthetic_value(
        self,
        rule: Any,
        utility: float,
        existing_rules: Sequence[Any],
        predicate_embeddings: Optional[Dict[int, torch.Tensor]] = None,
        compression_delta: Optional[float] = None,
    ) -> float:
        """
        A(R) = 0.25·A_sym + 0.25·A_compr + 0.25·A_novel + 0.25·A_elegance
        """
        return (
            0.25 * self._symmetry_score(rule)
            + 0.25 * self._compression_score(rule, compression_delta=compression_delta)
            + 0.25 * self._novelty_score(rule, existing_rules, predicate_embeddings=predicate_embeddings)
            + 0.25 * self._elegance_score(rule, utility)
        )

    # ─── Crossover і Mutation ─────────────────────────────────────────────────

    @staticmethod
    def _crossover(left: Any, right: Any) -> Any:
        """
        Схрещування: голова від right, тіло від left (якщо змінні уніфікуються).
        Це реалізує логічну рекомбінацію правил.
        """
        body = tuple(getattr(left, "body", ())) or tuple(getattr(right, "body", ()))
        return type(left)(
            head=right.head,
            body=body,
            weight=float(getattr(left, "weight", 1.0)),
            use_count=int(getattr(left, "use_count", 0)),
        )

    @staticmethod
    def _mutate_rule(
        rule: Any,
        alt_body_atom: Optional[Any] = None,
        alt_preds: Optional[Sequence[int]] = None,
    ) -> Any:
        """
        Мутація: заміна предиката, додавання або видалення літерала.
        """
        body = list(getattr(rule, "body", ()))
        mutation_ops: List[str] = []
        if alt_preds:
            mutation_ops.append("replace_head_pred")
        if alt_body_atom is not None and body:
            mutation_ops.append("replace_body_pred")
        if alt_body_atom is not None and len(body) < 3:
            mutation_ops.append("add_body_literal")
        if body:
            mutation_ops.append("drop_body_literal")
        if not mutation_ops:
            mutation_ops.append("noop")

        head = rule.head
        mutation = random.choice(mutation_ops)
        if mutation == "replace_head_pred" and alt_preds:
            replacement = int(random.choice(list(alt_preds)))
            if replacement != int(head.pred):
                head = type(head)(pred=replacement, args=tuple(head.args))
        elif mutation == "replace_body_pred" and alt_body_atom is not None and body:
            body_idx = random.randrange(len(body))
            old_atom = body[body_idx]
            body[body_idx] = type(old_atom)(
                pred=int(getattr(alt_body_atom, "pred", old_atom.pred)),
                args=tuple(old_atom.args),
            )
        elif mutation == "add_body_literal" and alt_body_atom is not None and len(body) < 3:
            body.append(alt_body_atom)
        elif mutation == "drop_body_literal" and body:
            del body[-1]
        return type(rule)(
            head=head,
            body=tuple(body),
            weight=float(getattr(rule, "weight", 1.0)),
            use_count=int(getattr(rule, "use_count", 0)),
        )

    # ─── Ініціалізація популяції з LTM ───────────────────────────────────────

    def _seed_from_ltm(
        self,
        ltm_rules: Sequence[Any],
        existing_rules: Sequence[Any],
        n_seeds: int,
    ) -> List[RuleCandidate]:
        """
        Відбирає n_seeds правил з LTM за естетичною цінністю (utility=1.0 як плейсхолдер,
        оскільки LTM правила вже пройшли VeM).
        Це реалізує засів популяції з кращих наявних правил.
        """
        if not ltm_rules or n_seeds <= 0:
            return []

        scored: List[Tuple[float, Any]] = []
        for rule in ltm_rules:
            aesthetic = self._aesthetic_value(rule, utility=1.0, existing_rules=existing_rules)
            scored.append((aesthetic, rule))
        scored.sort(key=lambda x: x[0], reverse=True)

        seeds: List[RuleCandidate] = []
        for aesthetic, rule in scored[:n_seeds]:
            seeds.append(
                RuleCandidate(
                    clause=rule,
                    source="ltm_seed",
                    score=float(aesthetic),
                    utility=1.0,
                    aesthetic=float(aesthetic),
                    structural_similarity=0.0,
                    metadata={"ltm_seed": 1.0},
                )
            )
        return seeds

    def _restore_from_gene_pool(self, n: int) -> List[RuleCandidate]:
        """Повертає n найкращих правил з персистентного gene pool."""
        if not self._gene_pool or n <= 0:
            return []
        sorted_pool = sorted(self._gene_pool, key=lambda c: c.score, reverse=True)
        return sorted_pool[:n]

    def _update_gene_pool(self, candidates: List[RuleCandidate]) -> None:
        """Поповнює gene pool виживачами, обмежуючи розмір gene_pool_size."""
        pool_hashes = {hash(c.clause) for c in self._gene_pool}
        for candidate in candidates:
            if hash(candidate.clause) not in pool_hashes:
                self._gene_pool.append(candidate)
                pool_hashes.add(hash(candidate.clause))
        # Відсортувати і обрізати
        self._gene_pool.sort(key=lambda c: c.score, reverse=True)
        self._gene_pool = self._gene_pool[: self.gene_pool_size]

    # ─── Головний метод еволюції ──────────────────────────────────────────────

    def evolve(
        self,
        candidates: Sequence[RuleCandidate],
        existing_rules: Sequence[Any],
        utility_fn: Callable[[Sequence[Any]], Sequence[float]],
        ltm_rules: Optional[Sequence[Any]] = None,
        predicate_embeddings: Optional[Dict[int, torch.Tensor]] = None,
        compression_fn: Optional[Callable[[Any], float]] = None,
    ) -> List[RuleCandidate]:
        """
        Еволюція популяції правил.

        Параметри
        ----------
        candidates : Sequence[RuleCandidate]
            Кандидати від AME / CWE / OEE.
        existing_rules : Sequence[Any]
            Наявні правила LTM (для оцінки новизни та як gene pool).
        utility_fn : Callable
            Функція VeM.score_batch — повертає utility для списку правил.
        ltm_rules : Sequence[Any], optional
            Правила LTM для засіву популяції (якщо None — використовує existing_rules).
        """
        effective_ltm = list(ltm_rules or existing_rules)

        # ── Формуємо початкову популяцію ─────────────────────────────────────
        # 1. Зовнішні кандидати (AME/CWE/OEE)
        external = list(candidates[: self.population_size])

        # 2. Насіння з LTM (ltm_seed_ratio частина популяції)
        n_ltm_seeds = max(1, int(self.population_size * self.ltm_seed_ratio))
        ltm_seeds = self._seed_from_ltm(effective_ltm, existing_rules, n_ltm_seeds)

        # 3. Відновлення з персистентного gene pool
        n_pool = max(1, self.population_size // 4)
        pool_seeds = self._restore_from_gene_pool(n_pool)

        # Об'єднуємо і обрізаємо до population_size
        population: List[RuleCandidate] = (external + ltm_seeds + pool_seeds)[: self.population_size]

        if not population:
            return []

        # Пул атомів і предикатів для мутацій
        body_atoms = [atom for rule in effective_ltm for atom in getattr(rule, "body", ())]
        head_pred_by_arity: Dict[int, List[int]] = {}
        for rule in effective_ltm:
            arity = int(rule.head.arity())
            pred = int(rule.head.pred)
            if arity not in head_pred_by_arity:
                head_pred_by_arity[arity] = []
            if pred not in head_pred_by_arity[arity]:
                head_pred_by_arity[arity].append(pred)

        # ── Еволюційні покоління ──────────────────────────────────────────────
        for _gen in range(self.generations):
            # Оцінюємо поточну популяцію через VeM
            clauses = [c.clause for c in population]
            utilities = list(utility_fn(clauses))

            rescored: List[RuleCandidate] = []
            for candidate, utility in zip(population, utilities):
                compression_delta = (
                    float(compression_fn(candidate.clause))
                    if compression_fn is not None else None
                )
                aesthetic = self._aesthetic_value(
                    candidate.clause,
                    float(utility),
                    existing_rules,
                    predicate_embeddings=predicate_embeddings,
                    compression_delta=compression_delta,
                )
                total = float(utility) + self.gamma * aesthetic
                metadata = dict(candidate.metadata)
                if compression_delta is not None:
                    metadata["compression_delta"] = float(compression_delta)
                rescored.append(
                    RuleCandidate(
                        clause=candidate.clause,
                        source=candidate.source,
                        score=total,
                        utility=float(utility),
                        aesthetic=float(aesthetic),
                        structural_similarity=float(candidate.structural_similarity),
                        metadata=metadata,
                    )
                )

            # Відбір виживачів (еліта = половина популяції)
            rescored.sort(key=lambda item: item.score, reverse=True)
            n_survivors = max(2, self.population_size // 2)
            survivors = rescored[:n_survivors]
            population = list(survivors)

            # Відтворення: crossover + mutation до population_size
            while len(population) < self.population_size:
                seed = random.choice(survivors)
                new_rule = seed.clause

                # Crossover: перевіряємо уніфікованість змінних
                if random.random() < self.crossover_rate and len(survivors) > 1:
                    partner = random.choice(survivors)
                    if _can_crossover(seed.clause, partner.clause):
                        new_rule = self._crossover(seed.clause, partner.clause)
                    elif _can_crossover(partner.clause, seed.clause):
                        new_rule = self._crossover(partner.clause, seed.clause)

                # Mutation
                if random.random() < self.mutation_rate:
                    alt_atom = random.choice(body_atoms) if body_atoms else None
                    alt_preds = head_pred_by_arity.get(int(new_rule.head.arity()), [])
                    new_rule = self._mutate_rule(
                        new_rule,
                        alt_body_atom=alt_atom,
                        alt_preds=alt_preds,
                    )

                population.append(
                    RuleCandidate(
                        clause=new_rule,
                        source=seed.source,
                        score=seed.score,
                        structural_similarity=seed.structural_similarity,
                        metadata=dict(seed.metadata),
                    )
                )

        # ── Фінальний відбір кращих унікальних правил ─────────────────────────
        best_by_template: Dict[Any, RuleCandidate] = {}
        for candidate in population:
            template = rule_template_signature(candidate.clause.head, tuple(candidate.clause.body))
            current = best_by_template.get(template)
            if current is None or candidate.score > current.score:
                best_by_template[template] = candidate
        ranked = sorted(best_by_template.values(), key=lambda item: item.score, reverse=True)
        selected = ranked[: self.max_selected]

        # Оновлюємо персистентний gene pool найкращими виживачами
        all_survivors = sorted(population, key=lambda c: c.score, reverse=True)
        self._update_gene_pool(all_survivors[: max(self.max_selected * 2, 4)])

        return selected

    def gene_pool_stats(self) -> Dict[str, float]:
        """Статистика gene pool для моніторингу."""
        if not self._gene_pool:
            return {"size": 0.0, "mean_score": 0.0, "max_score": 0.0}
        scores = [c.score for c in self._gene_pool]
        return {
            "size": float(len(self._gene_pool)),
            "mean_score": float(sum(scores) / len(scores)),
            "max_score": float(max(scores)),
        }
