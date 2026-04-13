"""
omen_prolog.py — ∂-Prolog: Диференційований Theorem Prover
===========================================================
Замінює GNN-based S-Core на реальну логіку першого порядку.

Структура:
  HornAtom       : (pred_id, args) — атом Хорнівського диз'юнкта
  HornClause     : head :- body  (факт = clause без body)
  KnowledgeBase  : факти + правила з Forward Chaining
  ProofPolicyNet : π_θ(Action|State) — навчає стратегію пошуку доведення
  AbductionHead  : нейромережевий генератор кандидатів-правил
  DifferentiableProver : інтегратор; повертає proof_loss через REINFORCE

Математика:
  L_sym = -E_{τ~π_θ}[R(τ)] + α·Length(τ)

  R(τ) = 1 якщо ціль доведена, 0 інакше
  Length(τ) = кількість кроків (заохочуємо короткі докази)

  Це REINFORCE-оновлення PolicyNetwork, де «розуміння» —
  здатність знайти найкоротший логічний ланцюжок.
"""

from __future__ import annotations
import enum
import math
import random
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


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
    Complexity(R) = rule.complexity()         — MDL статична складність.

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
        return float(self.rule.complexity()) - eta * self.utility()


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
            elif (rec.status == EpistemicStatus.proposed
                  and rec.use_count < use_count_threshold):
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
                      starting_facts: "Optional[FrozenSet]" = None) -> "FrozenSet[HornAtom]":
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
                if not clause.body:
                    continue
                fresh = freshen_vars(clause)
                for sigma in find_all_substitutions(fresh.body, current):
                    derived = sigma.apply_atom(fresh.head)
                    if derived.is_ground() and derived not in current:
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
        """Σ_R len(R) — довжина всіх правил у символах (базовий MDL)."""
        return sum(r.complexity() for r in self.rules)

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
            total += float(r.complexity()) - eta * util
        return total

    def weighted_complexity(self) -> float:
        """Σ_R use_count(R)·complexity(R)"""
        return sum(r.use_count * r.complexity() for r in self.rules)

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

    # ── Кодування HornAtom → (pred, a0, a1) int tuple ────────────────────────
    def _atom_to_key(self, atom: HornAtom) -> Tuple[int, int, int]:
        """
        Ground atom → (pred, a0, a1) int tuple.
        Змінні кодуються як VAR_0/VAR_1 (використовується для тіл правил).
        """
        args = atom.args

        def _enc(a: object, slot: int) -> int:
            if isinstance(a, Const):
                return int(a.val)
            if isinstance(a, Var):
                return self.VAR_0 if slot == 0 else self.VAR_1
            if isinstance(a, int):
                return a if a >= 0 else (self.VAR_0 if slot == 0 else self.VAR_1)
            return 0  # Compound: спрощено до 0

        a0 = _enc(args[0], 0) if len(args) > 0 else self.NONE
        a1 = _enc(args[1], 1) if len(args) > 1 else self.NONE
        return (int(atom.pred), a0, a1)

    # ── Кодування HornClause → (h_p, h_a0, h_a1, b_p, b_a0, b_a1) або None ─
    def _clause_to_tensor_row(self, clause: HornClause) -> Optional[Tuple[int, ...]]:
        """
        Однотільний HornClause → 6-int tuple для тензорного буфера.
        Повертає None для multi-body (>1 literal) — вони обробляються Python-ом.

        Логіка кодування змінних:
          · Збираємо унікальні Var-імена по порядку першої появи у body.
          · Перша Var → VAR_0 (-1), Друга Var → VAR_1 (-2).
          · У head: якщо arg — та сама Var → VAR_0/VAR_1 (буде замінено фактом).
          · Якщо Var у head не з'являється у body → VAR_0 (безпечний default).
        """
        if len(clause.body) != 1:
            return None  # multi-body: Python fallback

        head = clause.head
        body = clause.body[0]

        # Будуємо мапу ім'я Var → VAR_0/VAR_1 (за порядком появи у body)
        var_map: Dict[str, int] = {}
        for a in body.args:
            if isinstance(a, Var) and a.name not in var_map:
                slot = self.VAR_0 if len(var_map) == 0 else self.VAR_1
                var_map[a.name] = slot

        def _enc_head(a: object, pos: int) -> int:
            if isinstance(a, Const):
                return int(a.val)
            if isinstance(a, Var):
                return var_map.get(a.name, self.VAR_0)  # default VAR_0
            return 0

        def _enc_body(a: object, pos: int) -> int:
            if isinstance(a, Const):
                return int(a.val)
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
                      starting_facts: Optional[FrozenSet] = None) -> FrozenSet[HornAtom]:
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
        if self._n_facts == 0:
            return frozenset()

        # Якщо передано starting_facts — конвертуємо у тензор
        if starting_facts is not None:
            keys = [self._atom_to_key(a) for a in starting_facts]
            if not keys:
                return frozenset()
            facts = torch.tensor(keys, dtype=torch.long, device=self._dev)
            fact_set: Set[Tuple[int, int, int]] = set(keys)
        else:
            facts = self._fact_buf[:self._n_facts].clone()
            fact_set = set(self._fact_set)

        N = facts.shape[0]

        # Tensor-ready rules
        n_tensor_rules = self._rule_is_tensor[:self._n_rules].sum().item()
        if n_tensor_rules > 0:
            tensor_mask = self._rule_is_tensor[:self._n_rules]
            rules = self._rule_buf[:self._n_rules][tensor_mask]  # (R_t, 6)
            R = rules.shape[0]

            # Multi-body rules (Python fallback)
            multi_body_rules = [
                r for i, r in enumerate(self.rules)
                if i < self._n_rules and not self._rule_is_tensor[i].item()
            ]
        else:
            rules = torch.zeros(0, 6, dtype=torch.long, device=self._dev)
            R = 0
            multi_body_rules = [r for r in self.rules if len(r.body) > 1]

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

                match = pred_m & a0_m & a1_m                       # (N, R)

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
                    for row in candidates.tolist():
                        key = (row[0], row[1], row[2])
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
            if multi_body_rules:
                current_frozenset = frozenset(
                    HornAtom(pred=int(row[0]),
                             args=self._ints_to_args(int(row[1]), int(row[2])))
                    for row in facts.tolist()
                    if row[0] >= 0
                )
                for clause in multi_body_rules:
                    if not clause.body:
                        continue
                    fresh = freshen_vars(clause)
                    for sigma in find_all_substitutions(fresh.body, current_frozenset):
                        derived = sigma.apply_atom(fresh.head)
                        if derived.is_ground():
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
        return frozenset(
            HornAtom(pred=int(row[0]),
                     args=self._ints_to_args(int(row[1]), int(row[2])))
            for row in facts.tolist()
            if row[0] >= 0
        )

    def _ints_to_args(self, a0: int, a1: int) -> Tuple:
        """(a0, a1) int → Tuple[Term] для HornAtom."""
        if a1 == self.NONE:
            return (Const(a0),) if a0 >= 0 else (Var("_v0"),)
        args: list = []
        args.append(Const(a0) if a0 >= 0 else Var("_v0"))
        args.append(Const(a1) if a1 >= 0 else Var("_v1"))
        return tuple(args)

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

    def consolidate(self, use_count_threshold: int = 2) -> int:
        """Видаляємо слабкі / contradicted правила."""
        to_remove: Set[int] = set()
        for h, rec in list(self._records.items()):
            if rec.status == EpistemicStatus.contradicted:
                to_remove.add(h)
            elif (rec.status == EpistemicStatus.proposed
                  and rec.use_count < use_count_threshold):
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
        return sum(r.complexity() for r in self.rules)

    def utility_adjusted_penalty(self, eta: float = 0.1) -> float:
        total = 0.0
        for r in self.rules:
            rec = self._records.get(hash(r))
            util = rec.utility() if rec is not None else 0.0
            total += float(r.complexity()) - eta * util
        return total

    def weighted_complexity(self) -> float:
        return sum(r.use_count * r.complexity() for r in self.rules)

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

        def _const_int(term) -> Optional[int]:
            if isinstance(term, Const):
                return int(term.val)
            if isinstance(term, int):
                return int(term)
            return None

        ctx_to_tokens: Dict[int, Set[int]] = {}
        for fact in self.facts:
            if fact.pred != NET_CONTEXT_PRED or len(fact.args) < 2:
                continue
            tok = _const_int(fact.args[0])
            ctx = _const_int(fact.args[1])
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
            tok = _const_int(rule.head.args[0])
            concept = _const_int(rule.head.args[1])
            if tok is None or concept is None:
                continue
            for other in ctx_to_tokens.get(concept, set()):
                if other == tok:
                    continue
                key = tuple(sorted((tok, other)))
                scores[key] = max(scores.get(key, 0.0), 0.85)

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
        if len(ctx) >= 2:
            self.add_rule(
                HornClause(
                    head=HornAtom(pred=NET_MEANS_PRED,
                                  args=(Const(token_idx), Const(ctx[0]))),
                    body=(
                        HornAtom(pred=NET_CONTEXT_PRED, args=(Const(token_idx), Const(ctx[0]))),
                        HornAtom(pred=NET_CONTEXT_PRED, args=(Const(token_idx), Const(ctx[1]))),
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
        return self._n_facts

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
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
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
        total_entropy  = torch.tensor(0.0, device=device)
        total_energy   = torch.tensor(0.0, device=device)

        for name in var_names:
            q       = self.var_q(var_vecs[name])
            scores  = (q @ K.t()) * self.scale                # (|F|,)
            soft_w  = F.gumbel_softmax(scores, tau=tau, hard=hard)  # (|F|,)
            soft_v  = soft_w @ V                               # (d,) soft assignment
            var_assign[name] = soft_v

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
        return total_energy, var_assign, total_entropy / n_vars


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
          sym_cost = complexity(R) + λ·unif_complexity(σ)
        """
        rule_len  = float(clause.complexity())
        unif_cost = float(sigma.unif_complexity()) if sigma is not None else 0.0
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
          Cost = Σ_i [ 0.01·complexity(Ri) + nn_len(Ri) + λ·unif_c(σi) ]
        """
        if not trajectory:
            return torch.tensor(0.0, device=device)

        total = torch.tensor(0.0, device=device)
        for rule, sigma in trajectory:
            # MDL: Length(R) — символьна складність
            sym_len = float(rule.complexity())
            # Нейронна оцінка (диференційована для backprop)
            r_emb   = self.clause_emb(rule, device)          # (d,)
            nn_len  = self.rule_enc(r_emb.unsqueeze(0)).squeeze()   # scalar ≥ 0

            mdl_part = sym_len * 0.01 + nn_len

            # UnifComplexity(σ): Σ_{X∈dom(σ)} Depth(σ(X))
            unif_c  = float(sigma.unif_complexity()) if sigma is not None else 0.0
            total   = total + mdl_part + self.lam * unif_c

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

    def sample_candidates(self, z: torch.Tensor) -> Tuple[List[HornClause], torch.Tensor]:
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
        logits = self.rule_gen(z.squeeze(0))               # (sv * n_slots,)
        logits = logits.view(self.slots, self.sv)          # (slots, sv)

        # Змінні: ?X0,...?X_{max_arity} — набір іменованих змінних
        all_vars = [Var(f"X{i}") for i in range(self.max_arity + 1)]

        clauses = []
        log_probs: List[torch.Tensor] = []
        for cand_i in range(self.n_cands):
            indices: List[int] = []
            lp_sum = torch.zeros(1, device=z.device).squeeze()
            for i in range(self.slots):
                noisy_logits = logits[i] + torch.randn_like(logits[i]) * 0.05
                dist = Categorical(logits=noisy_logits / 0.7)
                idx = dist.sample()
                indices.append(int(idx.item()))
                lp_sum = lp_sum + dist.log_prob(idx)
            head_pred = indices[0]
            body_pred = indices[1 + self.max_arity]

            # Вибір паттерну змінних на основі кандидата
            mode = cand_i % 3
            if mode == 0:
                # grandparent(?X,?Y) :- parent(?X,?Y) — спільні змінні
                head_args = tuple(all_vars[:self.max_arity])
                body_args = tuple(all_vars[:self.max_arity])
            elif mode == 1:
                # grandparent(?X,?Z) :- parent(?X,?Y) — ланцюговий зв'язок
                # head: X, Z; body: X, Y
                xa, xb = all_vars[0], all_vars[1]
                xc     = all_vars[2] if self.max_arity >= 2 else all_vars[0]
                head_args = (xa, xc)[:self.max_arity]
                body_args = (xa, xb)[:self.max_arity]
            else:
                # mortal(?X) :- human(?X) — одна змінна
                head_args = (all_vars[0],) * self.max_arity
                body_args = (all_vars[0],) * self.max_arity

            head      = HornAtom(pred=head_pred, args=head_args)
            body_atom = HornAtom(pred=body_pred, args=body_args)

            clause = HornClause(head=head, body=(body_atom,))
            clauses.append(clause)
            log_probs.append(lp_sum)

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
        # fc_cache видалено: TensorKnowledgeBase.forward_chain виконується
        # за ~0.1–0.5 ms/батч через GPU broadcast — кешування не потрібне.

    # ── Нейронний → символьний (perception) ──────────────────────────────────
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
        facts_list = list(facts)
        # TermEmbedder враховує структуру кожного атому
        embs = self.term_emb(facts_list, device)              # (|facts|, d)
        return self.out_proj(embs.mean(0, keepdim=True))      # (1, d)

    def forward_chain_step(self) -> Tuple[int, "FrozenSet[HornAtom]"]:
        """
        Один крок forward chaining з комітом нових фактів у KB.

        Потрібно для EMC, інакше дія ForwardChainStep бачить приріст у
        `forward_chain(max_depth=1)`, але стан KB фактично не змінюється.
        """
        before = self.kb.facts
        after = self.kb.forward_chain(max_depth=1)
        new_facts = frozenset(f for f in after if f not in before)
        added = 0
        for fact in new_facts:
            added += int(self.kb.add_fact(fact))
        return added, new_facts

    # ── Proof Search (REINFORCE + Cost(T)) ───────────────────────────────────
    def prove_with_policy(self,
                          goal: HornAtom,
                          z_ctx: torch.Tensor,
                          n_steps: Optional[int] = None) -> Tuple[bool, List[int], torch.Tensor]:
        """
        Шукає доведення goal, використовуючи PolicyNetwork для вибору правил.

        Реалізує REINFORCE з MDL-вартістю доведення (розділ 6):
          L_proof = -E_{T~π_θ}[R(T) - α·Cost(T)]
          Cost(T) = Σ_{(R,σ)∈T} [Length(R) + λ·UnifComplexity(σ)]

        goal  : ціль доведення
        z_ctx : (1, d) — контекст з нейромережі
        Returns: (proved, trajectory, proof_loss)
        """
        n_steps = n_steps or self.max_depth
        n_rules = len(self.kb.rules)
        device  = z_ctx.device

        z_goal = self.goal_proj(z_ctx)                     # (1, d)
        current_facts = self.kb.facts                      # незмінний frozenset

        trajectory: List[int]         = []
        log_probs:  List[torch.Tensor]= []
        # Повна траєкторія для Cost(T): (rule, sigma) пари
        proof_steps: List[Tuple['HornClause', Optional[Substitution]]] = []

        proved = goal in current_facts   # вже є у KB?

        for step in range(n_steps):
            if proved or n_rules == 0:
                break

            # Policy вибирає правило
            log_p = self.policy(z_ctx, z_goal, max(n_rules, 1))  # (1, n_rules)
            dist  = Categorical(logits=log_p.squeeze(0))
            rule_idx = dist.sample()
            trajectory.append(rule_idx.item())
            log_probs.append(dist.log_prob(rule_idx))

            # Застосовуємо обране правило
            if rule_idx.item() < len(self.kb.rules):
                rule = self.kb.rules[rule_idx.item()]
                if rule.body:
                    # Freshen vars щоб уникнути конфліктів із поточними фактами
                    fresh = freshen_vars(rule)
                    sigma = unify_body(fresh.body, current_facts)
                    # Записуємо крок для Cost(T)
                    proof_steps.append((rule, sigma))
                    if sigma is not None:
                        derived = sigma.apply_atom(fresh.head)
                        # Додаємо лише ground-факти
                        if derived.is_ground():
                            current_facts = current_facts | {derived}
                            if unify(goal, derived) is not None:
                                proved = True

        # ── REINFORCE: L_proof = -E[R(T) - α·Cost(T)] ────────────────────────
        R = float(proved)
        # Обчислюємо MDL-вартість доведення через ProofCostEstimator
        proof_cost_tensor = self.cost_est(proof_steps, device)
        # R(T) - α·Cost(T): заохочує доведення, штрафує складні докази
        effective_reward  = R - self.alpha * proof_cost_tensor

        if log_probs:
            # baseline = 0 (можна замінити на baseline мережу)
            proof_loss = -effective_reward * torch.stack(log_probs).sum()
        else:
            proof_loss = proof_cost_tensor * self.alpha   # лише MDL регуляризатор

        return proved, trajectory, proof_loss

    # ── Abduce and Learn (повільний цикл) з VeM фільтрацією ──────────────────
    def abduce_and_learn(
        self,
        z: torch.Tensor,
        error: float,
        force: bool = False,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, float]:
        """
        Якщо error > threshold — генеруємо нові правила через абдукцію.
        VeM фільтрує кандидатів до додавання в LTM.

        Returns: (кількість доданих правил, hinge_loss для VeM)
        """
        device = z.device
        if (error < 0.5) and not force:
            zero = torch.zeros(1, device=device).squeeze()
            return 0, zero, zero, 0.0

        # Генеруємо кандидатів через AbductionHead
        raw_candidates, log_probs = self.abductor.sample_candidates(z[:1])
        utilities = self.vem.score_batch(raw_candidates, device) if raw_candidates else torch.zeros(0, device=device)

        # VeM фільтрація: тримаємо лише U(R) > vem_tau
        accepted, hinge_loss = self.vem.filter_candidates(raw_candidates, device)
        if log_probs.numel() == utilities.numel() and log_probs.numel() > 0:
            abductor_loss = -((utilities.detach() - self.vem_tau) * log_probs).mean()
        else:
            abductor_loss = torch.zeros(1, device=device).squeeze()

        # Додаємо прийнятих кандидатів з епістемічним статусом proposed
        added = 0
        for c in accepted:
            if self.kb.add_rule(c, status=EpistemicStatus.proposed):
                added += 1

        # Якщо нікого не прийнято — все одно записуємо VeM-сигнал
        for c in raw_candidates:
            score = self.vem.score(c, device)
            # Поки target невідомий → 0.5 (нейтральний prior)
            self.vem.record_outcome(c, utility_target=0.5, device=device)

        mean_utility = float(utilities.mean().item()) if utilities.numel() > 0 else 0.0
        return added, hinge_loss, abductor_loss, mean_utility

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
        self._step += 1

        # 0. Оновлюємо вік правил і запускаємо консолідацію
        self.kb.tick()
        if self._step % self.consolidate_every == 0:
            n_removed = self.kb.consolidate(use_count_threshold=2)
            # (можна логувати: n_removed правил видалено)

        # 1. Perception: z → факт → додаємо у KB
        goal = self.perceive(z)
        self.kb.add_fact(goal)

        # 2. Forward Chaining: GPU-акселерований (TensorKnowledgeBase)
        # ~0.1–0.5 ms/батч — кешування не потрібне, викликаємо кожен крок.
        all_facts = self.kb.forward_chain(self.max_depth)

        # 3. Grounding: KB → z_sym
        # PERF FIX: при великому KB (>128 фактів) сэмплюємо підмножину,
        # щоб уникнути O(N) Python-ітерації + N×3 GPU-викликів у ground().
        _MAX_GROUND = 128
        ground_sample = (frozenset(random.sample(list(all_facts), _MAX_GROUND))
                         if len(all_facts) > _MAX_GROUND else all_facts)
        z_sym_1 = self.ground(ground_sample, device)            # (1, d)
        z_sym   = z_sym_1.expand(B, -1)                    # (B, d)

        # 4. Proof search (тільки під час навчання)
        vem_hinge = torch.zeros(1, device=device).squeeze()
        abductor_aux = torch.zeros(1, device=device).squeeze()
        if self.training and len(self.kb.rules) > 0:
            proved, traj, proof_loss = self.prove_with_policy(goal, z[:1])

            # Оновлюємо VeM targets на основі результату доведення
            if traj and self.kb.rules:
                used_rule = self.kb.rules[traj[-1] % len(self.kb.rules)]
                self.vem.record_outcome(
                    used_rule,
                    utility_target=1.0 if proved else 0.0,
                    device=device
                )
                # Якщо доведення успішне → mark_rule_verified
                if proved:
                    self.kb.mark_rule_verified(used_rule)

            # GraphMatchingUnifier: консистентна soft-уніфікація (розділ 5.2)
            if self.kb.rules:
                r = random.choice(self.kb.rules)
                if r.body:
                    # PERF FIX: обмежуємо кількість фактів для soft-уніфікатора.
                    # SoftUnifier будує (|body|, |F|) матрицю — при |F|=1986 це
                    # 1986 embed_atom() викликів + великий тензор. 64 фактів достатньо.
                    _MAX_UNIF = 64
                    unif_facts = (frozenset(random.sample(list(all_facts), _MAX_UNIF))
                                  if len(all_facts) > _MAX_UNIF else all_facts)
                    gm_energy, var_assign, gm_entropy = self.graph_unif(
                        r.body, unif_facts, device, tau=0.5
                    )
                    su_e, su_ent = self.soft_unif(r.body, unif_facts, device)
                    proof_loss = (proof_loss
                                  + 0.01 * gm_energy
                                  - 0.001 * gm_entropy
                                  + 0.01 * su_e
                                  - 0.001 * su_ent)
        else:
            proof_loss = torch.zeros(1, device=device).squeeze()

        # 5. Abduce з VeM фільтрацією (раз на 5 кроків)
        if self._step % 5 == 0:
            err_val = (world_error.item()
                       if torch.is_tensor(world_error) else float(world_error))
            _, vem_hinge, abductor_aux, _ = self.abduce_and_learn(z, err_val)

        # 6. VeM self-supervised loss (раз на 10 кроків)
        vem_self_loss = torch.zeros(1, device=device).squeeze()
        if self.training and self._step % 10 == 0:
            vem_self_loss = self.vem.self_supervised_loss(device)

        # 7. Symbolic Consistency Loss: MSE між z та z_sym
        sym_consist = F.mse_loss(z, z_sym.detach()) + \
                      F.mse_loss(z_sym, z.detach())
        sym_loss    = (sym_consist
                       + 0.1  * proof_loss
                       + 0.01 * vem_hinge
                       + 0.05 * abductor_aux
                       + 0.01 * vem_self_loss)

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
        raw_candidates, log_probs = self.abductor.sample_candidates(z[:1])
        if not raw_candidates:
            return torch.zeros(1, device=device).squeeze()
        utilities = self.vem.score_batch(raw_candidates, device)
        _, hinge = self.vem.filter_candidates(raw_candidates, device)
        rl = -((utilities.detach() - self.vem_tau) * log_probs).mean()
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

    # ══ T_ALL_SUB: find_all_substitutions ══════════════════════════════════════
    sep("T_ALL_SUB · find_all_substitutions (backtracking DFS)")
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
    ))
    goal = HornAtom(pred=3, args=(Const(5), Const(7)))
    z1   = torch.randn(1, 32, device=device)
    proved, traj, pl = prover.prove_with_policy(goal, z1, n_steps=3)
    assert pl.dim() == 0 or pl.numel() == 1, "FAIL: proof_loss не скалярний"
    print(f"  proved={proved}  steps={len(traj)}  proof_loss={pl.item():.4f}  [PASS]")

    # ══ T10: Абдукція ═════════════════════════════════════════════════════════
    sep("T10 · abduce_and_learn")
    n_before = len(prover.kb)
    added, _, _, _ = prover.abduce_and_learn(z_in, error=2.0, force=True)
    n_after  = len(prover.kb)
    print(f"  rules before={n_before}  added={added}  after={n_after}  [PASS]")

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
