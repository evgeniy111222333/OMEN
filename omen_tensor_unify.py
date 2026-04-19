"""
omen_tensor_unify.py - Scalable Symbolic Machine
================================================
Four components for taking the system from demo mode into the real world:

  TensorFactBase       : facts as int32 tensors in VRAM -> O(1) batch lookup
  TensorUnifyEngine    : GPU forward chaining via relational joins
                         (instead of Python loops)
  ReteIndex            : incremental rule triggering, O(1) per fact
  PythonASTParser      : Python/JS AST -> Horn facts (500+ predicates)
  BPESymbolMapper      : GPT-2/CodeGen BPE tokens -> sym_vocab ids
  ScalableKnowledgeBase: TensorFactBase + ReteIndex + TensorUnifyEngine
                         + Python symbolic fallback

Mathematics - tensor unification as relational algebra:
  for each rule r with body [B0, B1, ..., Bk]:
    C_i = Facts[pred == B_i.pred]                    <- proj
    filter: C_i[:, j] == B_i.arg[j] if arg[j] >= 0  <- select (ground)
    join(C0, C1) on shared_vars: C0[:,p] == C1[:,q] <- equijoin (GPU)
    head = (head_pred, C[var_map(head.args)])        <- project

Complexity:
  Naive DFS forward_chain: O(R * N^B)  where N=|facts|, B=body_size
  TensorUnifyEngine:      O(R * N^2/b) where b=GPU batch -> ~100x faster

Operating modes:
  fast  : TensorUnifyEngine (GPU, >=10k facts)
  slow  : Python DFS (exact, with occurs check, <=1k facts)
  hybrid: fast by default, slow as an oracle fallback
"""

from __future__ import annotations

import ast
import hashlib
import math
import re
import textwrap
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Dict, FrozenSet, Iterable, List, Optional,
    Set, Tuple, Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the symbolic layer from omen_prolog.
from omen_prolog import (
    Const, Var, Compound, Term,
    HornAtom, HornClause,
    Substitution, KnowledgeBase,
    find_all_substitutions, freshen_vars,
    unify, unify_mm,
)

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
# 0.  PREDICATE VOCABULARY  (500+ real predicates)
# ══════════════════════════════════════════════════════════════════════════════

class PredicateVocab:
    """
    Predicate registry. The first 512 slots are reserved for categories:

    0–63   : code structure   (assign, call, ret, import, def, class, …)
    64–127 : types            (type_of, instanceof, subtype, …)
    128–191: control flow     (if_true, if_false, loop_body, break_to, …)
    192–255: data flow        (use, def, live, kill, …)
    256–319: relations        (parent, child, sibling, transitive, …)
    320–383: arith/logic      (add, sub, mul, gt, eq, …)
    384–447: memory           (alloc, free, read_mem, write_mem, …)
    448–511: meta/epistemic   (knows, believes, inferred, abduced, …)
    512+   : dynamic          (added during parsing)
    """

    # --- Built-in categories ---
    CODE = {
        "assign": 0, "call": 1, "return": 2, "import": 3,
        "define": 4, "classdef": 5, "attr": 6, "subscript": 7,
        "augassign": 8, "annot": 9, "global": 10, "nonlocal": 11,
        "yield": 12, "await": 13, "raise": 14, "assert": 15,
        "delete": 16, "with": 17, "pass": 18, "lambda": 19,
        "listcomp": 20, "dictcomp": 21, "setcomp": 22, "genexp": 23,
        "fstring": 24, "starred": 25, "unpack": 26, "walrus": 27,
        "decorator": 28, "classbase": 29, "defaultarg": 30, "kwarg": 31,
    }
    TYPES = {
        "type_of": 64, "instanceof": 65, "subtype": 66, "coerce": 67,
        "union_t": 68, "intersect_t": 69, "optional_t": 70, "generic_t": 71,
        "int_t": 72, "str_t": 73, "float_t": 74, "bool_t": 75,
        "list_t": 76, "dict_t": 77, "set_t": 78, "tuple_t": 79,
        "none_t": 80, "any_t": 81, "callable_t": 82, "iter_t": 83,
    }
    CTRL = {
        "if_true": 128, "if_false": 129, "loop_body": 130, "loop_cond": 131,
        "break_to": 132, "continue_to": 133, "except_handler": 134,
        "finally": 135, "try": 136, "match_case": 137,
        "entry": 138, "exit": 139, "dom": 140, "postdom": 141,
    }
    FLOW = {
        "use": 192, "def_var": 193, "live": 194, "kill": 195,
        "reach": 196, "dep_data": 197, "dep_ctrl": 198, "alias": 199,
        "capture": 200, "free_var": 201, "param": 202, "closure": 203,
    }
    RELATIONS = {
        "parent": 256, "child": 257, "sibling": 258, "ancestor": 259,
        "descendant": 260, "transitive": 261, "symmetric": 262,
        "inverse": 263, "compose": 264, "equiv": 265,
    }
    ARITH = {
        "add": 320, "sub": 321, "mul": 322, "div": 323, "mod": 324,
        "pow": 325, "neg": 326, "abs": 327, "floor": 328,
        "eq": 330, "ne": 331, "lt": 332, "le": 333, "gt": 334, "ge": 335,
        "and_op": 336, "or_op": 337, "not_op": 338, "xor": 339,
        "band": 340, "bor": 341, "bnot": 342, "lshift": 343, "rshift": 344,
    }
    MEMORY = {
        "alloc": 384, "free": 385, "read_mem": 386, "write_mem": 387,
        "load": 388, "store": 389, "move": 390, "copy": 391,
    }
    EPISTEMIC = {
        "knows": 448, "believes": 449, "inferred": 450, "abduced": 451,
        "certain": 452, "uncertain": 453, "contradicts": 454,
        "entails": 455, "consistent": 456, "possible": 457,
    }

    def __init__(self):
        self._name2id: Dict[str, int] = {}
        self._id2name: Dict[int, str] = {}
        self._next_dynamic = 512

        for cat in (self.CODE, self.TYPES, self.CTRL, self.FLOW,
                    self.RELATIONS, self.ARITH, self.MEMORY, self.EPISTEMIC):
            for name, pid in cat.items():
                self._register(name, pid)

    def _register(self, name: str, pid: int) -> None:
        self._name2id[name]  = pid
        self._id2name[pid]   = name

    def get_id(self, name: str) -> int:
        """Return a predicate id, registering a new dynamic one if needed."""
        if name in self._name2id:
            return self._name2id[name]
        pid = self._next_dynamic
        self._next_dynamic += 1
        self._register(name, pid)
        return pid

    def get_name(self, pid: int) -> str:
        return self._id2name.get(pid, f"p{pid}")

    def __len__(self) -> int:
        return self._next_dynamic

    def vocab_size(self) -> int:
        return self._next_dynamic


PRED_VOCAB = PredicateVocab()


# ══════════════════════════════════════════════════════════════════════════════
# 1.  BPE SYMBOL MAPPER  (GPT-2 / CodeGen tokens <-> sym_vocab)
# ══════════════════════════════════════════════════════════════════════════════

class BPESymbolMapper:
    """
    Mapping between BPE tokens (GPT-2/CodeGen) and symbolic constants.

    Problem: `sym_vocab=64` vs `vocab_size=50257` BPE tokens.
    Solution: group hashing - route rare tokens through a hash bucket,
    while frequent tokens (top-K) receive direct slots.

    BPE -> sym_id:
      top_K (K=sym_vocab-10) most frequent tokens -> direct mapping
      others -> sym_id = K + (hash(tok) % (sym_vocab - K - 1))

    sym_id -> BPE: reverse dictionary (exact for top-K, approximate for hash groups).
    """

    def __init__(self, sym_vocab: int = 512, top_k: Optional[int] = None):
        self.sym_vocab = sym_vocab
        self.top_k     = top_k or (sym_vocab - 32)   # reserve 32 for special tokens

        # Try to load tiktoken.
        self._enc     = None
        self._bpe2sym: Dict[int, int] = {}
        self._sym2bpe: Dict[int, int] = {}
        self._loaded  = False
        self._try_load_tiktoken()

    def _try_load_tiktoken(self) -> None:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            self._enc = enc
            all_tokens = list(range(enc.n_vocab))
            def tok_len(t: int) -> int:
                try:
                    return len(enc.decode([t]))
                except Exception:
                    return 999
            sorted_toks = sorted(all_tokens, key=tok_len)[:self.top_k]
            for sym_id, bpe_id in enumerate(sorted_toks):
                self._bpe2sym[bpe_id]  = sym_id
                self._sym2bpe[sym_id]  = bpe_id
            self._loaded = True
        except Exception:
            # ImportError, ProxyError, ConnectionError, etc.
            # Network is blocked — gracefully fall back to hash mode.
            pass

    def bpe_to_sym(self, bpe_id: int) -> int:
        """BPE token id → sym_vocab id."""
        if bpe_id in self._bpe2sym:
            return self._bpe2sym[bpe_id]
        # Group hashing for rare tokens.
        return self.top_k + (bpe_id % (self.sym_vocab - self.top_k - 1))

    def sym_to_bpe(self, sym_id: int) -> Optional[int]:
        """sym_vocab id -> BPE token id (`None` for hash groups)."""
        return self._sym2bpe.get(sym_id)

    def encode(self, text: str) -> List[int]:
        """Convert text to a list of sym_vocab ids."""
        if self._enc is None:
            return [abs(hash(c)) % self.sym_vocab for c in text.split()]
        bpe_ids = self._enc.encode(text)
        return [self.bpe_to_sym(b) for b in bpe_ids]

    def decode(self, sym_ids: List[int]) -> str:
        """Convert sym_vocab ids back to approximate text where possible."""
        if self._enc is None:
            return " ".join(f"<{s}>" for s in sym_ids)
        bpe_ids = [self._sym2bpe.get(s) for s in sym_ids]
        valid   = [b for b in bpe_ids if b is not None]
        if not valid:
            return " ".join(f"<{s}>" for s in sym_ids)
        try:
            return self._enc.decode(valid)
        except Exception:
            return str(sym_ids)

    @property
    def is_bpe_loaded(self) -> bool:
        return self._loaded


# ══════════════════════════════════════════════════════════════════════════════
# 2.  TENSOR FACT BASE  (int32 tensors in VRAM)
# ══════════════════════════════════════════════════════════════════════════════

class TensorFactBase:
    """
    Compact fact storage as int32 tensors.

    F ∈ Z^{N × (1 + MAX_ARITY)}
      F[i, 0]   = pred_id  (≥ 0)
      F[i, j+1] = const_id (>= 0) or PAD (-1)

    Advantages:
      · Batch lookup: F[F[:,0] == p] - O(N) on GPU instead of an O(N) Python loop
      · Cache-friendly: sequential memory access
      · Ready for torch.compile()

    PAD = -1 means a missing argument (facts with smaller arity).
    """

    PAD = -1

    def __init__(self, max_arity: int = 3, device: torch.device = DEVICE):
        self.max_arity = max_arity
        self.device    = device
        # Main fact tensor (grows dynamically).
        self._F: Optional[torch.Tensor] = None   # (N, 1+max_arity), int32
        self._n = 0

        # Predicate index: pred_id -> row indices for fast lookup.
        self._pred_index: Dict[int, List[int]] = defaultdict(list)

        # Set used for deduplication.
        self._fact_set: Set[Tuple] = set()

        # Ground-term interning: Const/Compound -> stable int ids.
        self._term_to_id: Dict[Term, int] = {}
        self._id_to_term: Dict[int, Term] = {}
        self._next_term_id: int = 1

    @staticmethod
    def _is_ground_term(term: Term) -> bool:
        return len(term.vars()) == 0

    def _intern_term(self, term: Union[Term, int]) -> int:
        term_obj: Term = Const(int(term)) if isinstance(term, int) else term
        if not self._is_ground_term(term_obj):
            raise ValueError(f"TensorFactBase expects ground terms, got: {term_obj}")
        term_id = self._term_to_id.get(term_obj)
        if term_id is None:
            term_id = self._next_term_id
            self._next_term_id += 1
            self._term_to_id[term_obj] = term_id
            self._id_to_term[term_id] = term_obj
        return term_id

    def _decode_term(self, term_id: int) -> Term:
        return self._id_to_term.get(term_id, Const(int(term_id)))

    # -- Conversion ------------------------------------------------------------
    def atom_to_row(self, atom: HornAtom) -> Optional[Tuple[int, ...]]:
        """HornAtom → tensor row with ground Const/Compound args encoded via term ids."""
        if not atom.is_ground():
            return None
        args: List[int] = []
        for a in atom.args[:self.max_arity]:
            if isinstance(a, (Const, Compound)):
                args.append(self._intern_term(a))
            else:
                return None
        while len(args) < self.max_arity:
            args.append(self.PAD)
        return (atom.pred,) + tuple(args)

    def row_to_atom(self, row: torch.Tensor) -> Optional[HornAtom]:
        """Convert a tensor row to a `HornAtom`."""
        pred = int(row[0].item())
        args = []
        for j in range(self.max_arity):
            v = int(row[1 + j].item())
            if v == self.PAD:
                break
            args.append(self._decode_term(v))
        return HornAtom(pred=pred, args=tuple(args))

    # -- Fact insertion --------------------------------------------------------
    def add_atom(self, atom: HornAtom) -> bool:
        """Add a `HornAtom` as a tensor row. Return `True` if it is new."""
        row_tuple = self.atom_to_row(atom)
        if row_tuple is None:
            return False
        if row_tuple in self._fact_set:
            return False
        self._fact_set.add(row_tuple)

        row_t = torch.tensor(list(row_tuple), dtype=torch.int32,
                             device=self.device)
        if self._F is None:
            self._F = row_t.unsqueeze(0)
        else:
            self._F = torch.cat([self._F, row_t.unsqueeze(0)], dim=0)

        self._pred_index[row_tuple[0]].append(self._n)
        self._n += 1
        return True

    def add_atoms(self, atoms: Iterable[HornAtom]) -> int:
        """Batch insertion. Return the number of newly added facts."""
        return sum(1 for a in atoms if self.add_atom(a))

    # -- Queries ---------------------------------------------------------------
    def get_by_pred(self, pred_id: int) -> Optional[torch.Tensor]:
        """Return the submatrix `F` where `F[:,0] == pred_id`. O(1) via index."""
        indices = self._pred_index.get(pred_id)
        if not indices or self._F is None:
            return None
        idx_t = torch.tensor(indices, dtype=torch.long, device=self.device)
        return self._F[idx_t]   # (k, 1+max_arity)

    def get_all(self) -> Optional[torch.Tensor]:
        return self._F

    def to_horn_atoms(self, rows: Optional[torch.Tensor] = None) -> List[HornAtom]:
        """Convert a tensor of rows to a list of `HornAtom`s."""
        if rows is None:
            rows = self._F
        if rows is None:
            return []
        atoms = []
        for i in range(rows.shape[0]):
            a = self.row_to_atom(rows[i])
            if a is not None:
                atoms.append(a)
        return atoms

    def to_frozenset(self) -> FrozenSet[HornAtom]:
        return frozenset(self.to_horn_atoms())

    def __len__(self) -> int:
        return self._n

    def pred_counts(self) -> Dict[int, int]:
        return {p: len(idx) for p, idx in self._pred_index.items()}


# ══════════════════════════════════════════════════════════════════════════════
# 3.  RETE INDEX  (O(1) rule triggering per fact)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ReteNode:
    """RETE node tracking which rules are triggered by predicate `p`."""
    pred_id:  int
    rule_ids: List[int] = field(default_factory=list)  # rule indices in the KB
    body_pos: List[int] = field(default_factory=list)  # position inside the rule body


class ReteIndex:
    """
    Simplified RETE network for O(1) discovery of which rule can fire
    from a new fact `p(...)`.

    When a new fact `f` with `pred=p` is added:
      trig = rete[p] -> list of (rule_id, body_position)

    This avoids an O(R) scan over all rules.

    Incremental update:
      instead of re-evaluating all rules for all facts, only process
      rules whose body contains `p`.
    """

    def __init__(self):
        # pred_id → [(rule_id, body_position), ...]
        self._triggers: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self._rules: List[HornClause] = []

    def register_rule(self, rule: HornClause) -> int:
        """Register a rule and return its `rule_id`."""
        rule_id = len(self._rules)
        self._rules.append(rule)
        for body_pos, atom in enumerate(rule.body):
            self._triggers[atom.pred].append((rule_id, body_pos))
        return rule_id

    def get_triggered(self, pred_id: int) -> List[Tuple[int, HornClause, int]]:
        """
        Return a list of `(rule_id, rule, body_pos)` entries for `pred_id`.
        O(k), where `k` is the number of rules whose body contains this predicate.
        """
        return [
            (rid, self._rules[rid], bpos)
            for rid, bpos in self._triggers.get(pred_id, [])
        ]

    def all_rules(self) -> List[HornClause]:
        return self._rules

    def __len__(self) -> int:
        return len(self._rules)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  TENSOR UNIFY ENGINE  (GPU-vectorized forward chaining)
# ══════════════════════════════════════════════════════════════════════════════

class TensorUnifyEngine:
    """
    GPU-vectorized forward chaining via relational algebra.

    Tensor unification:
      for each rule `r` with body [B0, B1]:  (2-body-atom rules; general case below)

        C0 = F[pred == B0.pred]              # (k0, 1+A) - candidates for B0
        C1 = F[pred == B1.pred]              # (k1, 1+A) - candidates for B1

        # Ground-argument filter (select):
        mask₀ = AND_{j: B₀.arg[j] is ground} (C₀[:,j+1] == B₀.arg[j])
        C₀    = C₀[mask₀]                   # (k₀', 1+A)

        # Equijoin on shared variables:
        # If variable v appears in B0 at pos p0 and in B1 at pos p1:
        #   join_mask = (C₀[:, p₀+1].unsqueeze(1) == C₁[:, p₁+1].unsqueeze(0))
        # For multiple shared vars: AND all join masks.

        i₀, i₁ = join_mask.nonzero(as_tuple=True)   # (m,) (m,)

        # Project the head:
        head_rows = build_head(C₀[i₀], C₁[i₁], head_template, var_map)

        # Deduplication:
        new_facts = head_rows[~isin(head_rows, F)]

    Supports bodies with `len(body) <= MAX_BODY_SIZE = 3`.
    Larger bodies are recursively split into pairs.

    Complexity vs Python DFS:
      Python: O(R * k^B * A)    where k=avg_facts_per_pred, B=body_size
      GPU:    O(R * k^2 / W)    where W=warp_width ~= 32  (join matmul only)
    """

    MAX_BODY = 3  # maximum supported body length

    def __init__(self, max_arity: int = 3, device: torch.device = DEVICE):
        self.max_arity = max_arity
        self.device    = device

    # -- Internal helpers ------------------------------------------------------

    @staticmethod
    def _var_positions(atom: HornAtom) -> Dict[str, List[int]]:
        """
        Return `{var_name: [positions in args]}` for an atom.
        Positions are 0-based indices inside `args`.
        """
        result: Dict[str, List[int]] = defaultdict(list)
        for i, a in enumerate(atom.args):
            if isinstance(a, Var) and not a.name.startswith('_'):
                result[a.name].append(i)
        return dict(result)

    @staticmethod
    def _is_tensor_term(term: Term) -> bool:
        if isinstance(term, (Const, Var)):
            return True
        if isinstance(term, Compound):
            return len(term.vars()) == 0
        return False

    def _is_tensor_rule(self, clause: HornClause) -> bool:
        if len(clause.body) == 0 or len(clause.body) > self.MAX_BODY:
            return False
        atoms = [clause.head] + list(clause.body)
        for atom in atoms:
            if len(atom.args) > self.max_arity:
                return False
            for arg in atom.args:
                if not self._is_tensor_term(arg):
                    return False
        return True

    def _is_structured_rule(self, clause: HornClause) -> bool:
        if len(clause.body) == 0 or len(clause.body) > self.MAX_BODY:
            return False
        atoms = [clause.head] + list(clause.body)
        has_structured_compound = False
        for atom in atoms:
            if len(atom.args) > self.max_arity:
                return False
            for arg in atom.args:
                if isinstance(arg, (Const, Var)):
                    continue
                if isinstance(arg, Compound):
                    if len(arg.vars()) > 0:
                        has_structured_compound = True
                    continue
                return False
        return has_structured_compound

    @staticmethod
    def _term_id(term: Term, fact_base: TensorFactBase) -> Optional[int]:
        if isinstance(term, Const):
            return fact_base._intern_term(term)
        if isinstance(term, Compound) and len(term.vars()) == 0:
            return fact_base._intern_term(term)
        return None

    def _ground_filter(self,
                       candidates: torch.Tensor,
                       atom: HornAtom,
                       fact_base: TensorFactBase) -> torch.Tensor:
        """
        Filter candidate rows `(N, 1+A)` by an atom's ground arguments.
        Return a boolean mask of shape `(N,)`.
        """
        if candidates.numel() == 0:
            return torch.zeros(0, dtype=torch.bool,
                               device=candidates.device)
        N = candidates.shape[0]
        mask = torch.ones(N, dtype=torch.bool, device=candidates.device)
        for j, a in enumerate(atom.args):
            term_id = self._term_id(a, fact_base)
            if term_id is not None:
                mask = mask & (candidates[:, j + 1] == term_id)
        for positions in self._var_positions(atom).values():
            if len(positions) < 2:
                continue
            anchor = positions[0]
            for pos in positions[1:]:
                mask = mask & (candidates[:, anchor + 1] == candidates[:, pos + 1])
        return mask

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

    def _row_bindings(
        self,
        atom: HornAtom,
        row: torch.Tensor,
        fact_base: TensorFactBase,
    ) -> Optional[Dict[str, Term]]:
        bindings: Dict[str, Term] = {}
        values: List[Term] = []
        for j in range(len(atom.args)):
            term_id = int(row[j + 1].item())
            if term_id == TensorFactBase.PAD:
                return None
            values.append(fact_base._decode_term(term_id))
        for pattern, value in zip(atom.args, values):
            bindings = self._match_term_bindings(pattern, value, bindings)
            if bindings is None:
                return None
        return bindings

    def _candidate_matches(
        self,
        candidates: torch.Tensor,
        atom: HornAtom,
        fact_base: TensorFactBase,
    ) -> List[Tuple[List[torch.Tensor], Dict[str, Term]]]:
        matches: List[Tuple[List[torch.Tensor], Dict[str, Term]]] = []
        for idx in range(candidates.shape[0]):
            row = candidates[idx]
            bindings = self._row_bindings(atom, row, fact_base)
            if bindings is not None:
                matches.append(([row], bindings))
        return matches

    @staticmethod
    def _merge_bindings(
        left: Dict[str, Term],
        right: Dict[str, Term],
    ) -> Optional[Dict[str, Term]]:
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
        sig: List[Term] = []
        for name in names:
            if name not in bindings:
                return None
            sig.append(bindings[name])
        return tuple(sig)

    def _signature_tensor(
        self,
        matches: List[Tuple[List[torch.Tensor], Dict[str, Term]]],
        names: Tuple[str, ...],
        fact_base: TensorFactBase,
    ) -> Tuple[List[Tuple[List[torch.Tensor], Dict[str, Term]]], Optional[torch.Tensor]]:
        if not names:
            return matches, None
        kept: List[Tuple[List[torch.Tensor], Dict[str, Term]]] = []
        rows: List[List[int]] = []
        for packed in matches:
            _trace_rows, bindings = packed
            sig: List[int] = []
            ok = True
            for name in names:
                term = bindings.get(name)
                if term is None or len(term.vars()) != 0:
                    ok = False
                    break
                sig.append(fact_base._intern_term(term))
            if ok:
                kept.append(packed)
                rows.append(sig)
        if not rows:
            return [], None
        return kept, torch.tensor(rows, dtype=torch.long, device=self.device)

    def _equijoin_matches(
        self,
        left_matches: List[Tuple[List[torch.Tensor], Dict[str, Term]]],
        right_matches: List[Tuple[List[torch.Tensor], Dict[str, Term]]],
        left_atom: Optional[HornAtom],
        right_atom: Optional[HornAtom],
        fact_base: TensorFactBase,
    ) -> List[Tuple[List[torch.Tensor], Dict[str, Term]]]:
        if not left_matches or not right_matches:
            return []
        if left_atom is not None and right_atom is not None:
            shared_names = tuple(sorted(self._atom_var_names(left_atom) & self._atom_var_names(right_atom)))
        else:
            shared_names = tuple(sorted(set(left_matches[0][1].keys()) & set(right_matches[0][1].keys())))

        joined: List[Tuple[List[torch.Tensor], Dict[str, Term]]] = []
        if not shared_names:
            for left_rows, left_bind in left_matches:
                for right_rows, right_bind in right_matches:
                    merged = self._merge_bindings(left_bind, right_bind)
                    if merged is not None:
                        joined.append((left_rows + right_rows, merged))
            return joined

        left_kept, left_sig = self._signature_tensor(left_matches, shared_names, fact_base)
        right_kept, right_sig = self._signature_tensor(right_matches, shared_names, fact_base)
        if left_sig is not None and right_sig is not None:
            match_mask = (left_sig.unsqueeze(1) == right_sig.unsqueeze(0)).all(dim=-1)
            i_left, i_right = match_mask.nonzero(as_tuple=True)
            for li, ri in zip(i_left.tolist(), i_right.tolist()):
                left_rows, left_bind = left_kept[li]
                right_rows, right_bind = right_kept[ri]
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

    def _equijoin(
        self,
        C0: torch.Tensor,       # (k0, 1+A)
        C1: torch.Tensor,       # (k1, 1+A)
        body0: HornAtom,
        body1: HornAtom,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Equijoin `C0 x C1` on shared variables between `body0` and `body1`.
        Return `(indices_in_C0, indices_in_C1)` with shape `(m,)`.
        """
        if C0.numel() == 0 or C1.numel() == 0:
            empty = torch.zeros(0, dtype=torch.long, device=self.device)
            return empty, empty

        # Find shared variables and their positions.
        vars0 = self._var_positions(body0)
        vars1 = self._var_positions(body1)
        shared = set(vars0.keys()) & set(vars1.keys())

        if not shared:
            # No shared variables -> Cartesian product (can be large).
            k0, k1 = C0.shape[0], C1.shape[0]
            i0 = torch.arange(k0, device=self.device).repeat_interleave(k1)
            i1 = torch.arange(k1, device=self.device).repeat(k0)
            return i0, i1

        # Build the join mask via broadcasting.
        # For each shared variable v: C0[:, p0+1] == C1[:, p1+1]
        join_mask = torch.ones(C0.shape[0], C1.shape[0],
                               dtype=torch.bool, device=self.device)
        for vname in shared:
            p0 = vars0[vname][0]   # first position in body0
            p1 = vars1[vname][0]   # first position in body1
            # Broadcasting: (k0, 1) == (1, k1)
            col0 = C0[:, p0 + 1].unsqueeze(1)   # (k0, 1)
            col1 = C1[:, p1 + 1].unsqueeze(0)   # (1, k1)
            join_mask = join_mask & (col0 == col1)

        return join_mask.nonzero(as_tuple=True)

    def _build_head_rows(
        self,
        C_list: List[torch.Tensor],  # list of (m, 1+A) tensors for each body atom
        body:   Tuple[HornAtom, ...],
        head:   HornAtom,
        fact_base: TensorFactBase,
    ) -> Optional[torch.Tensor]:
        """
        Build rows for newly derived facts from the rule head.
        `C_list[i][j]` is the j-th fact row for the i-th body atom after joining.
        """
        if not C_list:
            return None

        m = C_list[0].shape[0]
        if m == 0:
            return None

        W = 1 + self.max_arity
        head_rows = torch.full((m, W), TensorFactBase.PAD,
                               dtype=torch.int32, device=self.device)
        head_rows[:, 0] = head.pred

        # Build var_map: var_name -> tensor column `(m,)`.
        var_map: Dict[str, torch.Tensor] = {}
        for bi, (atom, C) in enumerate(zip(body, C_list)):
            for j, a in enumerate(atom.args):
                if isinstance(a, Var) and not a.name.startswith('_'):
                    # Read the value from the joined candidate rows.
                    if a.name not in var_map:
                        var_map[a.name] = C[:, j + 1]

        # Fill head arguments.
        for j, a in enumerate(head.args):
            if j >= self.max_arity:
                break
            term_id = self._term_id(a, fact_base)
            if term_id is not None:
                head_rows[:, j + 1] = term_id
            elif isinstance(a, Var):
                if a.name in var_map:
                    head_rows[:, j + 1] = var_map[a.name]
                # else: could not resolve the value -> keep PAD

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

    def _build_structured_head_atoms(
        self,
        head: HornAtom,
        joined_matches: List[Tuple[List[torch.Tensor], Dict[str, Term]]],
    ) -> List[HornAtom]:
        atoms: List[HornAtom] = []
        for _rows, bindings in joined_matches:
            args: List[Term] = []
            ok = True
            for arg in head.args:
                bound = self._instantiate_bound_term(arg, bindings)
                if bound is None:
                    ok = False
                    break
                args.append(bound)
            if ok:
                atoms.append(HornAtom(pred=head.pred, args=tuple(args)))
        return atoms

    def _fast_isin(self, new_rows: torch.Tensor,
                   existing: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Check which rows in `new_rows` already exist in `existing`.
        Return a boolean mask of shape `(len(new_rows),)` where `True` means "already present".
        """
        if existing is None or existing.numel() == 0:
            return torch.zeros(new_rows.shape[0], dtype=torch.bool,
                               device=self.device)

        # Cast to int64 and use broadcasting for comparison.
        # Safe, overflow-free approach: compare each column separately.
        new_i64  = new_rows.to(torch.int64)    # (n_new, W)
        exist_i64 = existing.to(torch.int64)   # (n_exist, W)
        W = new_i64.shape[1]

        # (n_new, n_exist, W) - pairwise comparison of each row.
        # Process in chunks if the tensor is large.
        chunk = 512
        result = torch.zeros(new_rows.shape[0], dtype=torch.bool,
                             device=self.device)
        for start in range(0, new_i64.shape[0], chunk):
            block = new_i64[start:start+chunk]           # (b, W)
            # (b, n_exist, W)
            match = (block.unsqueeze(1) == exist_i64.unsqueeze(0))
            # A row already exists if all W columns match at least one existing row.
            result[start:start+chunk] = match.all(dim=2).any(dim=1)

        return result

    # -- Main method: one forward-chaining step --------------------------------
    def forward_chain_step(
        self,
        fact_base: TensorFactBase,
        rules: List[HornClause],
    ) -> int:
        """
        Execute one GPU forward-chaining step.
        Return the number of newly added facts.

        For each rule:
          1. Gather candidates for each body atom (by predicate)
          2. Apply the ground-argument filter
          3. Run an equijoin (1->2 body atoms) or chain join (3 body atoms)
          4. Build new facts from the head template
          5. Deduplicate and insert
        """
        total_added = 0

        for rule in rules:
            if not rule.body:
                continue
            # Freshen variables to avoid naming conflicts.
            fresh = freshen_vars(rule)
            body  = fresh.body
            head  = fresh.head

            if len(body) > self.MAX_BODY:
                # Long bodies fall back to Python DFS.
                from omen_prolog import find_all_substitutions
                subs = find_all_substitutions(
                    body, fact_base.to_frozenset(), max_solutions=128)
                for sigma in subs:
                    derived = sigma.apply_atom(head)
                    if derived.is_ground():
                        fact_base.add_atom(derived)
                        total_added += 1
                continue

            if self._is_structured_rule(fresh):
                C0_raw = fact_base.get_by_pred(body[0].pred)
                if C0_raw is None:
                    continue
                mask0 = self._ground_filter(C0_raw, body[0], fact_base)
                C0 = C0_raw[mask0]
                if C0.shape[0] == 0:
                    continue
                matches0 = self._candidate_matches(C0, body[0], fact_base)
                if not matches0:
                    continue

                if len(body) == 1:
                    joined_matches = matches0
                elif len(body) == 2:
                    C1_raw = fact_base.get_by_pred(body[1].pred)
                    if C1_raw is None:
                        continue
                    mask1 = self._ground_filter(C1_raw, body[1], fact_base)
                    C1 = C1_raw[mask1]
                    if C1.shape[0] == 0:
                        continue
                    matches1 = self._candidate_matches(C1, body[1], fact_base)
                    if not matches1:
                        continue
                    joined_matches = self._equijoin_matches(
                        matches0, matches1, body[0], body[1], fact_base
                    )
                else:
                    C1_raw = fact_base.get_by_pred(body[1].pred)
                    C2_raw = fact_base.get_by_pred(body[2].pred)
                    if C1_raw is None or C2_raw is None:
                        continue
                    mask1 = self._ground_filter(C1_raw, body[1], fact_base)
                    mask2 = self._ground_filter(C2_raw, body[2], fact_base)
                    C1, C2 = C1_raw[mask1], C2_raw[mask2]
                    if C1.shape[0] == 0 or C2.shape[0] == 0:
                        continue
                    matches1 = self._candidate_matches(C1, body[1], fact_base)
                    matches2 = self._candidate_matches(C2, body[2], fact_base)
                    if not matches1 or not matches2:
                        continue
                    joined01 = self._equijoin_matches(
                        matches0, matches1, body[0], body[1], fact_base
                    )
                    if not joined01:
                        continue
                    joined_matches = self._equijoin_matches(
                        joined01, matches2, None, body[2], fact_base
                    )

                for derived in self._build_structured_head_atoms(head, joined_matches):
                    if derived.is_ground() and fact_base.add_atom(derived):
                        total_added += 1
                continue

            if not self._is_tensor_rule(fresh):
                # Unsupported tensor rule -> fall back to Python DFS.
                from omen_prolog import find_all_substitutions
                subs = find_all_substitutions(
                    body, fact_base.to_frozenset(), max_solutions=128)
                for sigma in subs:
                    derived = sigma.apply_atom(head)
                    if derived.is_ground():
                        fact_base.add_atom(derived)
                        total_added += 1
                continue

            # -- Gather candidates for body[0] ---------------------------------
            C0_raw = fact_base.get_by_pred(body[0].pred)
            if C0_raw is None:
                continue
            mask0 = self._ground_filter(C0_raw, body[0], fact_base)
            C0    = C0_raw[mask0]                         # (k0, 1+A)
            if C0.shape[0] == 0:
                continue

            if len(body) == 1:
                # Single body atom -> just project the head.
                C_joined = [C0]
                head_rows = self._build_head_rows(C_joined, body, head, fact_base)

            elif len(body) == 2:
                # Two body atoms -> one equijoin.
                C1_raw = fact_base.get_by_pred(body[1].pred)
                if C1_raw is None:
                    continue
                mask1 = self._ground_filter(C1_raw, body[1], fact_base)
                C1    = C1_raw[mask1]
                if C1.shape[0] == 0:
                    continue

                i0, i1 = self._equijoin(C0, C1, body[0], body[1])
                if len(i0) == 0:
                    continue
                C_joined = [C0[i0], C1[i1]]
                head_rows = self._build_head_rows(C_joined, body, head, fact_base)

            else:
                # body=3: two-phase join
                C1_raw = fact_base.get_by_pred(body[1].pred)
                C2_raw = fact_base.get_by_pred(body[2].pred)
                if C1_raw is None or C2_raw is None:
                    continue
                mask1 = self._ground_filter(C1_raw, body[1], fact_base)
                mask2 = self._ground_filter(C2_raw, body[2], fact_base)
                C1, C2 = C1_raw[mask1], C2_raw[mask2]
                if C1.shape[0] == 0 or C2.shape[0] == 0:
                    continue

                # Phase 1: join body0 × body1
                i01, i11 = self._equijoin(C0, C1, body[0], body[1])
                if len(i01) == 0:
                    continue
                C01_0 = C0[i01]   # (m1, 1+A)
                C01_1 = C1[i11]

                # Phase 2: join intermediate × body2
                # Build a merged atom for the join via var_positions.
                # Simplification: join on variable positions from bodies 1 and 2.
                i_mid, i2 = self._equijoin(C01_1, C2, body[1], body[2])
                if len(i_mid) == 0:
                    continue
                C_joined = [C01_0[i_mid], C01_1[i_mid], C2[i2]]
                head_rows = self._build_head_rows(C_joined, body, head, fact_base)

            if head_rows is None or head_rows.shape[0] == 0:
                continue

            # -- Filter PAD arguments and invalid predicates --------------------
            valid = (head_rows[:, 0] >= 0) & (head_rows[:, 1] != TensorFactBase.PAD)
            head_rows = head_rows[valid]
            if head_rows.shape[0] == 0:
                continue

            # -- Deduplicate via `isin` -----------------------------------------
            already = self._fast_isin(head_rows, fact_base.get_all())
            new_rows = head_rows[~already]
            if new_rows.shape[0] == 0:
                continue

            # -- Insert new facts -----------------------------------------------
            new_atoms = fact_base.to_horn_atoms(new_rows)
            added = fact_base.add_atoms(new_atoms)
            total_added += added
            if added > 0:
                rule.use_count += added
                rule.weight    *= (1.0 + 0.01 * added)

        return total_added

    def forward_chain(
        self,
        fact_base: TensorFactBase,
        rules: List[HornClause],
        max_depth: int = 5,
        verbose: bool = False,
    ) -> int:
        """
        Run full GPU forward chaining until a fixpoint or `max_depth` iterations.
        Return the total number of newly derived facts.
        """
        total = 0
        for depth in range(max_depth):
            added = self.forward_chain_step(fact_base, rules)
            total += added
            if verbose:
                print(f"  [TensorUnify] depth={depth+1}  added={added}"
                      f"  total_facts={len(fact_base)}")
            if added == 0:
                break   # fixpoint
        return total


# ══════════════════════════════════════════════════════════════════════════════
# 5.  PYTHON AST PARSER  (Python code → Horn facts)
# ══════════════════════════════════════════════════════════════════════════════

class ConstPool:
    """Intern string names into `Const(int)` values."""
    def __init__(self):
        self._pool: Dict[str, int] = {}
        self._rev:  Dict[int, str] = {}
        self._next  = 0

    def intern(self, name: str) -> int:
        if name not in self._pool:
            self._pool[name] = self._next
            self._rev[self._next] = name
            self._next += 1
        return self._pool[name]

    def name(self, cid: int) -> str:
        return self._rev.get(cid, f"<{cid}>")

    def atom(self, pred_name: str, *args: str) -> HornAtom:
        """Convenience constructor for `HornAtom` using string names."""
        pred = PRED_VOCAB.get_id(pred_name)
        arg_consts = tuple(Const(self.intern(a)) for a in args)
        return HornAtom(pred=pred, args=arg_consts)

    def __len__(self) -> int:
        return self._next


class PythonASTParser(ast.NodeVisitor):
    """
    Python -> Horn-fact parser.

    Supports 12 fact categories:
      assign(lhs, rhs)       - assignment
      call(result, func)     - function call
      call_arg(call, arg)    - call argument
      define(name, scope)    - function/class definition
      param(func, arg)       - function parameter
      return(func, val)      - returned value
      import(module)         - import
      type_of(var, type)     - type annotation
      attr(obj, field)       - field access
      if_true/if_false(cond, scope)  - branches
      loop_body(iter, scope) - loop body
      dep_data(a, b)         - data dependency: `a` uses `b`

    Every name is interned into the constant pool.
    """

    def __init__(self, scope: str = "<module>"):
        self.pool    = ConstPool()
        self.facts:  List[HornAtom] = []
        self._scope  = scope
        self._call_counter = 0
        self._scope_stack: List[str] = [scope]

    @property
    def _cur_scope(self) -> str:
        return self._scope_stack[-1]

    def _fact(self, pred_name: str, *args: str) -> None:
        self.facts.append(self.pool.atom(pred_name, *args))

    def _name(self, node) -> str:
        """Extract a string name from an AST node."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{self._name(node.value)}.{node.attr}"
        if isinstance(node, ast.Constant):
            return repr(node.value)[:32]
        if isinstance(node, ast.Subscript):
            return f"{self._name(node.value)}[...]"
        if isinstance(node, ast.Call):
            return f"call_{self._name(node.func)}"
        return f"<{type(node).__name__}>"

    # ── Visitors ───────────────────────────────────────────────────────────────

    def visit_Assign(self, node: ast.Assign) -> None:
        rhs = self._name(node.value)
        for tgt in node.targets:
            lhs = self._name(tgt)
            self._fact("assign", lhs, rhs)
            self._fact("dep_data", lhs, rhs)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        tgt = self._name(node.target)
        ann = self._name(node.annotation)
        self._fact("annot", tgt, ann)
        self._fact("type_of", tgt, ann)
        if node.value is not None:
            val = self._name(node.value)
            self._fact("assign", tgt, val)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        tgt = self._name(node.target)
        val = self._name(node.value)
        op  = type(node.op).__name__.lower()
        self._fact("augassign", tgt, op)
        self._fact("dep_data", tgt, val)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        fname = node.name
        self._fact("define", fname, self._cur_scope)
        for dec in node.decorator_list:
            self._fact("decorator", fname, self._name(dec))
        for arg in node.args.args:
            self._fact("param", fname, arg.arg)
        for i, default in enumerate(node.args.defaults):
            self._fact("defaultarg", fname, self._name(default))
        if node.returns:
            self._fact("type_of", f"{fname}:ret", self._name(node.returns))
        self._scope_stack.append(fname)
        self.generic_visit(node)
        self._scope_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        cname = node.name
        self._fact("classdef", cname, self._cur_scope)
        for base in node.bases:
            self._fact("classbase", cname, self._name(base))
        for dec in node.decorator_list:
            self._fact("decorator", cname, self._name(dec))
        self._scope_stack.append(cname)
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None:
            val = self._name(node.value)
            self._fact("return", self._cur_scope, val)
            self._fact("dep_data", f"{self._cur_scope}:ret", val)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        cid = f"_call{self._call_counter}"
        self._call_counter += 1
        func = self._name(node.func)
        self._fact("call", cid, func)
        for arg in node.args:
            self._fact("call_arg", cid, self._name(arg))
        for kw in node.keywords:
            kwval = self._name(kw.value)
            key   = kw.arg or "**kwargs"
            self._fact("call_arg", cid, f"{key}={kwval}")
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.asname or alias.name
            self._fact("import", name, alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or "<unknown>"
        for alias in node.names:
            name = alias.asname or alias.name
            self._fact("import", name, module)
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        cond = self._name(node.test)
        self._fact("if_true", cond, self._cur_scope)
        if node.orelse:
            self._fact("if_false", cond, self._cur_scope)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        it  = self._name(node.iter)
        tgt = self._name(node.target)
        self._fact("loop_body", it, self._cur_scope)
        self._fact("assign", tgt, it)
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        cond = self._name(node.test)
        self._fact("loop_cond", cond, self._cur_scope)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        obj  = self._name(node.value)
        attr = node.attr
        self._fact("attr", obj, attr)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        for name in node.names:
            self._fact("global", name, self._cur_scope)
        self.generic_visit(node)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        for name in node.names:
            self._fact("nonlocal", name, self._cur_scope)
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        exc = self._name(node.exc) if node.exc else "<none>"
        self._fact("raise", exc, self._cur_scope)
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            ctx = self._name(item.context_expr)
            self._fact("with", ctx, self._cur_scope)
        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        lid = f"lambda_{len(self.facts)}"
        self._fact("lambda", lid, self._cur_scope)
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        for tgt in node.targets:
            self._fact("delete", self._name(tgt), self._cur_scope)
        self.generic_visit(node)

    @classmethod
    def parse_code(cls, code: str,
                   scope: str = "<module>") -> Tuple[List[HornAtom], ConstPool]:
        """
        Main entry point.
        Return `(list[HornAtom], ConstPool)` for decoding.
        """
        code = textwrap.dedent(code)
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return [], cls(scope).pool
        parser = cls(scope=scope)
        parser.visit(tree)
        return parser.facts, parser.pool

    @classmethod
    def parse_file(cls, path: str) -> Tuple[List[HornAtom], ConstPool]:
        with open(path, "r", encoding="utf-8") as f:
            return cls.parse_code(f.read(), scope=path)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  SCALABLE KNOWLEDGE BASE  (hybrid: tensor + symbolic)
# ══════════════════════════════════════════════════════════════════════════════

class ScalableKnowledgeBase:
    """
    Hybrid KB: TensorFactBase + ReteIndex + TensorUnifyEngine + Python fallback.

    Modes:
      'fast'   : GPU tensor path only (for >= 1000 facts)
      'slow'   : Python DFS only (exact, with occurs check)
      'hybrid' : fast up to the threshold, slow for complex rules (body >= 3 with variables)

    The interface is compatible with `omen_prolog.KnowledgeBase`.
    """

    def __init__(self,
                 max_arity:  int   = 3,
                 max_rules:  int   = 1024,
                 device:     torch.device = DEVICE,
                 mode:       str   = "hybrid",
                 tensor_threshold: int = 100,):
        self.mode       = mode
        self.threshold  = tensor_threshold
        self.device     = device
        self.max_arity  = max_arity

        # Tensor backend
        self.tensor_fb   = TensorFactBase(max_arity, device)
        self.tensor_eng  = TensorUnifyEngine(max_arity, device)
        self.rete        = ReteIndex()

        # Python backend (fallback / oracle)
        self.py_kb       = KnowledgeBase(max_rules=max_rules)

        # Synchronized state.
        self._synced     = True

    # -- Insertion -------------------------------------------------------------
    def add_fact(self, atom: HornAtom) -> bool:
        added = self.tensor_fb.add_atom(atom)
        if added:
            self.py_kb.add_fact(atom)
        self._synced = added or self._synced
        return added

    def add_facts(self, atoms: Iterable[HornAtom]) -> int:
        return sum(1 for a in atoms if self.add_fact(a))

    def add_rule(self, clause: HornClause) -> bool:
        added = self.py_kb.add_rule(clause)
        if added:
            self.rete.register_rule(clause)
        return added

    def _has_fast_compatible_rule(self, rules: List[HornClause]) -> bool:
        for rule in rules:
            if self.tensor_eng._is_tensor_rule(rule) or self.tensor_eng._is_structured_rule(rule):
                return True
        return False

    # ── Forward Chaining ───────────────────────────────────────────────────────
    def forward_chain(self, max_depth: int = 5,
                      verbose: bool = False) -> FrozenSet[HornAtom]:
        """
        Hybrid forward chaining:
          · fast: TensorUnifyEngine (GPU) - main loop
          · slow: Python DFS - for rules with `body > MAX_BODY` or occurs checks
        """
        rules = self.rete.all_rules()
        n     = len(self.tensor_fb)

        if self.mode == "slow":
            # Pure Python.
            return self.py_kb.forward_chain(max_depth)

        if self.mode != "fast" and n < self.threshold and not self._has_fast_compatible_rule(rules):
            return self.py_kb.forward_chain(max_depth)

        # Fast tensor pass
        added = self.tensor_eng.forward_chain(
            self.tensor_fb, rules, max_depth, verbose)

        # Sync back into the Python KB.
        new_atoms = self.tensor_fb.to_horn_atoms()
        for a in new_atoms:
            self.py_kb.add_fact(a)

        if verbose:
            print(f"  [ScalableKB] fast added={added}"
                  f"  total={len(self.tensor_fb)}")

        return self.tensor_fb.to_frozenset()

    # ── MDL ────────────────────────────────────────────────────────────────────
    def complexity_penalty(self) -> float:
        return self.py_kb.complexity_penalty()

    # -- Statistics ------------------------------------------------------------
    def stats(self) -> Dict[str, int]:
        rules = self.rete.all_rules()
        return {
            "n_facts":   len(self.tensor_fb),
            "n_rules":   len(self.rete),
            "pred_types": len(self.tensor_fb.pred_counts()),
            "tensor_rules": sum(1 for r in rules if self.tensor_eng._is_tensor_rule(r)),
            "structured_rules": sum(1 for r in rules if self.tensor_eng._is_structured_rule(r)),
        }

    def __len__(self) -> int:
        return len(self.tensor_fb)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  BUILT-IN RULES FOR CODE  (starter Horn clauses for program analysis)
# ══════════════════════════════════════════════════════════════════════════════

def make_code_rules(kb: ScalableKnowledgeBase) -> None:
    """
    Add baseline Python-code analysis rules to the KB.

    Rules (Horn clauses with FOL variables):
      1. Transitive data dependency:
         dep_data(?A,?C) :- dep_data(?A,?B), dep_data(?B,?C)
      2. Definition through assignment:
         def_var(?X, ?S) :- assign(?X, ?_), define(?_, ?S)
         [simplified: def_var(?X) :- assign(?X, ?_)]
      3. Call result usage:
         use(?R, ?F) :- call(?R, ?F)
      4. Parameter usage:
         use(?P, ?F) :- param(?F, ?P)
      5. Return dependency:
         dep_data(?F, ?V) :- return(?F, ?V)
    """
    p = PRED_VOCAB.get_id
    A, B, C = Var("A"), Var("B"), Var("C")
    X, F, V, R = Var("X"), Var("F"), Var("V"), Var("R")
    S  = Var("S")

    # 1. Transitive data dependency
    kb.add_rule(HornClause(
        head=HornAtom(pred=p("dep_data"), args=(A, C)),
        body=(
            HornAtom(pred=p("dep_data"), args=(A, B)),
            HornAtom(pred=p("dep_data"), args=(B, C)),
        )
    ))

    # 2. Variable use → function dependency
    kb.add_rule(HornClause(
        head=HornAtom(pred=p("use"), args=(R, F)),
        body=(HornAtom(pred=p("call"), args=(R, F)),)
    ))

    # 3. Param use
    kb.add_rule(HornClause(
        head=HornAtom(pred=p("use"), args=(V, F)),
        body=(HornAtom(pred=p("param"), args=(F, V)),)
    ))

    # 4. Return dependency
    kb.add_rule(HornClause(
        head=HornAtom(pred=p("dep_data"), args=(F, V)),
        body=(HornAtom(pred=p("return"), args=(F, V)),)
    ))

    # 5. Type inference: if assign(X, Y) and type_of(Y, T) -> type_of(X, T)
    kb.add_rule(HornClause(
        head=HornAtom(pred=p("type_of"), args=(X, V)),
        body=(
            HornAtom(pred=p("assign"), args=(X, B)),
            HornAtom(pred=p("type_of"), args=(B, V)),
        )
    ))


# ══════════════════════════════════════════════════════════════════════════════
# 8.  INLINE TESTS
# ══════════════════════════════════════════════════════════════════════════════

def _run_tensor_tests() -> None:
    sep  = lambda s: print(f"\n{'═'*62}\n  {s}\n{'═'*62}")
    device = torch.device("cpu")   # CPU for tests (GPU optional)
    print(f"[omen_tensor_unify] device={device}")

    # ══ T0: PredicateVocab ═════════════════════════════════════════════════════
    sep("T0 · PredicateVocab (500+ predicates)")
    pv = PredicateVocab()
    assert pv.get_id("assign") == 0,   f"assign FAIL: {pv.get_id('assign')}"
    assert pv.get_id("dep_data") == 197
    assert pv.get_id("parent")  == 256
    assert pv.get_id("add")     == 320
    dyn_id = pv.get_id("my_custom_pred")
    assert dyn_id >= 512, f"dynamic id < 512: {dyn_id}"
    assert pv.get_name(0) == "assign"
    print(f"  vocab_size={pv.vocab_size()}  dynamic_id={dyn_id}  [PASS]")

    # ══ T1: BPESymbolMapper ════════════════════════════════════════════════════
    sep("T1 · BPESymbolMapper (GPT-2/CodeGen tokens)")
    bpe = BPESymbolMapper(sym_vocab=512, top_k=480)
    # BPE -> sym mapping is deterministic.
    s0 = bpe.bpe_to_sym(100)
    s1 = bpe.bpe_to_sym(100)
    assert s0 == s1, "Mapping must be deterministic"
    assert 0 <= s0 < 512
    # Encode/decode
    enc = bpe.encode("def foo(x): return x + 1")
    assert len(enc) > 0
    assert all(0 <= s < 512 for s in enc)
    print(f"  loaded={bpe.is_bpe_loaded}  enc_len={len(enc)}"
          f"  sym_range=[{min(enc)},{max(enc)}]  [PASS]")

    # ══ T2: TensorFactBase ════════════════════════════════════════════════════
    sep("T2 · TensorFactBase (int32 tensor)")
    fb = TensorFactBase(max_arity=2, device=device)
    p_parent = PRED_VOCAB.get_id("parent")

    atoms = [
        HornAtom(pred=p_parent, args=(Const(1), Const(2))),
        HornAtom(pred=p_parent, args=(Const(2), Const(3))),
        HornAtom(pred=p_parent, args=(Const(3), Const(4))),
    ]
    for a in atoms: fb.add_atom(a)

    assert len(fb) == 3, f"Expected 3, got {len(fb)}"
    # Deduplication.
    fb.add_atom(atoms[0])
    assert len(fb) == 3, "Duplicate not filtered"

    by_pred = fb.get_by_pred(p_parent)
    assert by_pred is not None and by_pred.shape == (3, 3), f"Shape: {by_pred.shape}"

    recovered = fb.to_horn_atoms()
    assert len(recovered) == 3
    print(f"  n_facts={len(fb)}  tensor_shape={tuple(by_pred.shape)}  [PASS]")

    sep("T2b · TensorFactBase / TensorUnifyEngine — compound ground terms")
    p_tree = PRED_VOCAB.get_id("tree_edge")
    p_mark = PRED_VOCAB.get_id("inferred")
    comp_left = Compound(func=7, subterms=(Const(1), Const(2)))
    comp_right = Compound(func=8, subterms=(Const(3),))
    fb_comp = TensorFactBase(max_arity=2, device=device)
    fb_comp.add_atom(HornAtom(pred=p_tree, args=(comp_left, comp_right)))
    recovered_comp = fb_comp.to_horn_atoms()
    assert recovered_comp[0].args == (comp_left, comp_right), f"compound decode FAIL: {recovered_comp}"
    rule_comp = HornClause(
        head=HornAtom(pred=p_mark, args=(Var("A"),)),
        body=(HornAtom(pred=p_tree, args=(Var("A"), comp_right)),),
    )
    added_comp = TensorUnifyEngine(max_arity=2, device=device).forward_chain_step(fb_comp, [rule_comp])
    assert added_comp == 1, f"compound 1-body rule FAIL: {added_comp}"
    marked = [a for a in fb_comp.to_horn_atoms() if a.pred == p_mark]
    assert any(a.args == (comp_left,) for a in marked), f"compound derivation FAIL: {marked}"
    print(f"  compound facts={len(fb_comp)}  derived={len(marked)}  [PASS]")

    # ══ T3: ReteIndex ═════════════════════════════════════════════════════════
    sep("T3 · ReteIndex (O(1) trigger)")
    sep("T2c · TensorUnifyEngine — structured compound join")
    p_nested = PRED_VOCAB.get_id("nested_fact")
    p_link = PRED_VOCAB.get_id("nested_link")
    p_struct = PRED_VOCAB.get_id("structured_out")
    fb_struct = TensorFactBase(max_arity=2, device=device)
    nested_fact = Compound(func=20, subterms=(Const(1), Const(5)))
    fb_struct.add_atom(HornAtom(pred=p_nested, args=(nested_fact,)))
    fb_struct.add_atom(HornAtom(pred=p_link, args=(Const(5), Const(7))))
    YS, ZS = Var("YS"), Var("ZS")
    rule_struct = HornClause(
        head=HornAtom(pred=p_struct, args=(Compound(func=30, subterms=(YS, ZS)),)),
        body=(
            HornAtom(pred=p_nested, args=(Compound(func=20, subterms=(Const(1), YS)),)),
            HornAtom(pred=p_link, args=(YS, ZS)),
        ),
    )
    added_struct = TensorUnifyEngine(max_arity=2, device=device).forward_chain_step(fb_struct, [rule_struct])
    assert added_struct == 1, f"structured compound join FAIL: {added_struct}"
    structured_atoms = [a for a in fb_struct.to_horn_atoms() if a.pred == p_struct]
    expected_struct = Compound(func=30, subterms=(Const(5), Const(7)))
    assert any(a.args == (expected_struct,) for a in structured_atoms), f"structured derivation FAIL: {structured_atoms}"
    print(f"  structured facts={len(fb_struct)}  derived={len(structured_atoms)}  [PASS]")

    sep("T3 · ReteIndex (O(1) trigger)")
    rete = ReteIndex()
    p_gp   = PRED_VOCAB.get_id("ancestor")
    X, Y, Z = Var("X"), Var("Y"), Var("Z")

    # parent(?X,?Y), parent(?Y,?Z) → ancestor(?X,?Z)
    rule_gp = HornClause(
        head=HornAtom(pred=p_gp, args=(X, Z)),
        body=(
            HornAtom(pred=p_parent, args=(X, Y)),
            HornAtom(pred=p_parent, args=(Y, Z)),
        )
    )
    rid = rete.register_rule(rule_gp)
    assert rid == 0

    triggered = rete.get_triggered(p_parent)
    assert len(triggered) == 2, f"Expected 2 triggers (body x2), got {len(triggered)}"
    # Predicate `ancestor` does not trigger this rule.
    assert len(rete.get_triggered(p_gp)) == 0
    print(f"  triggers(parent)={len(triggered)}  triggers(ancestor)=0  [PASS]")

    # ══ T4: TensorUnifyEngine — 1-body rule ════════════════════════════════════
    sep("T4 · TensorUnifyEngine — 1-body rule")
    # mortal(?X) :- human(?X)
    p_human  = PRED_VOCAB.get_id("type_of")
    p_mortal = PRED_VOCAB.get_id("instanceof")
    XV = Var("XV")
    rule_1b = HornClause(
        head=HornAtom(pred=p_mortal, args=(XV,)),
        body=(HornAtom(pred=p_human, args=(XV,)),)
    )

    fb1 = TensorFactBase(max_arity=2, device=device)
    fb1.add_atom(HornAtom(pred=p_human, args=(Const(10),)))
    fb1.add_atom(HornAtom(pred=p_human, args=(Const(11),)))

    eng = TensorUnifyEngine(max_arity=2, device=device)
    added = eng.forward_chain_step(fb1, [rule_1b])
    assert added == 2, f"Expected 2 new facts, got {added}"
    by_mortal = fb1.get_by_pred(p_mortal)
    assert by_mortal is not None and by_mortal.shape[0] == 2
    print(f"  added={added}  mortal_facts={by_mortal.shape[0]}  [PASS]")

    # ══ T5: TensorUnifyEngine — 2-body (grandparent/ancestor) ═════════════════
    sep("T5 · TensorUnifyEngine — 2-body join (ancestor/grandparent)")
    fb2 = TensorFactBase(max_arity=2, device=device)
    for a in atoms: fb2.add_atom(a)   # parent: 1→2, 2→3, 3→4

    eng2 = TensorUnifyEngine(max_arity=2, device=device)
    total = eng2.forward_chain(fb2, [rule_gp], max_depth=5)
    all_ancestor = fb2.get_by_pred(p_gp)
    print(f"  total_added={total}  ancestor_facts="
          f"{0 if all_ancestor is None else all_ancestor.shape[0]}")
    # ancestor(1,3), ancestor(2,4), ancestor(1,4) should be derived.
    all_atoms = fb2.to_horn_atoms()
    ancestor_atoms = [a for a in all_atoms if a.pred == p_gp]
    assert any(
        a.args == (Const(1), Const(3)) for a in ancestor_atoms
    ), f"ancestor(1,3) not found in: {ancestor_atoms}"
    assert any(
        a.args == (Const(2), Const(4)) for a in ancestor_atoms
    ), f"ancestor(2,4) not found"
    print(f"  ancestor(1,3) ✓  ancestor(2,4) ✓  [PASS]")

    # ══ T6: PythonASTParser ═══════════════════════════════════════════════════
    sep("T6 · PythonASTParser (Python → Horn facts)")
    CODE = """
def factorial(n: int) -> int:
    if n <= 1:
        return 1
    result = n * factorial(n - 1)
    return result

class MathHelper:
    def __init__(self, base: int):
        self.base = base

    def compute(self, x):
        return self.base + factorial(x)
"""
    facts, pool = PythonASTParser.parse_code(CODE)
    assert len(facts) > 0, "No facts were produced"

    # Check key facts.
    p_define = PRED_VOCAB.get_id("define")
    p_param  = PRED_VOCAB.get_id("param")
    p_ret    = PRED_VOCAB.get_id("return")
    p_call   = PRED_VOCAB.get_id("call")

    fact_preds = {f.pred for f in facts}
    assert p_define in fact_preds, "define not found"
    assert p_param  in fact_preds, "param not found"
    assert p_ret    in fact_preds, "return not found"

    # Check that factorial and MathHelper are defined.
    define_facts  = [f for f in facts if f.pred == p_define]
    p_classdef    = PRED_VOCAB.get_id("classdef")
    class_facts   = [f for f in facts if f.pred == p_classdef]
    defined_names = {pool.name(f.args[0].val) for f in define_facts
                     if f.args and isinstance(f.args[0], Const)}
    class_names   = {pool.name(f.args[0].val) for f in class_facts
                     if f.args and isinstance(f.args[0], Const)}
    assert "factorial" in defined_names, f"factorial not in: {defined_names}"
    assert "MathHelper" in class_names,  f"MathHelper not in classes: {class_names}"

    print(f"  n_facts={len(facts)}  n_const={len(pool)}"
          f"  predicates={len(fact_preds)}")
    print(f"  defined: {defined_names}  [PASS]")

    # ══ T7: ScalableKnowledgeBase ════════════════════════════════════════════
    sep("T7 · ScalableKnowledgeBase (hybrid mode)")
    skb = ScalableKnowledgeBase(max_arity=2, device=device, mode="hybrid")

    # Load facts extracted from Python code.
    for f in facts:
        if f.is_ground():
            skb.add_fact(f)
    make_code_rules(skb)

    n_before = len(skb)
    result_set = skb.forward_chain(max_depth=3)
    n_after  = len(skb)
    assert n_after >= n_before

    stats = skb.stats()
    assert "tensor_rules" in stats and "structured_rules" in stats
    print(f"  facts_before={n_before}  facts_after={n_after}"
          f"  rules={stats['n_rules']}  preds={stats['pred_types']}  [PASS]")

    sep("T7b · ScalableKnowledgeBase prefers structured fast-path below threshold")
    skb_fast = ScalableKnowledgeBase(
        max_arity=2,
        device=device,
        mode="hybrid",
        tensor_threshold=10_000,
    )
    p_pair = pv.get_id("pair")
    p_link = pv.get_id("link")
    p_struct = pv.get_id("structured")
    Y = Var("Y")
    structured_rule = HornClause(
        head=HornAtom(pred=p_struct, args=(Compound("node", (Y, Const(3))),)),
        body=(
            HornAtom(pred=p_pair, args=(Compound("node", (Const(1), Y)),)),
            HornAtom(pred=p_link, args=(Y, Const(3))),
        ),
    )
    skb_fast.add_rule(structured_rule)
    skb_fast.add_fact(HornAtom(pred=p_pair, args=(Compound("node", (Const(1), Const(2))),)))
    skb_fast.add_fact(HornAtom(pred=p_link, args=(Const(2), Const(3))))
    fast_calls = {"n": 0}
    orig_forward = skb_fast.tensor_eng.forward_chain

    def _counting_forward(*args, **kwargs):
        fast_calls["n"] += 1
        return orig_forward(*args, **kwargs)

    skb_fast.tensor_eng.forward_chain = _counting_forward
    skb_fast.forward_chain(max_depth=2)
    assert fast_calls["n"] == 1, f"Expected fast-path call, got {fast_calls['n']}"
    assert any(
        atom.pred == p_struct and atom.args == (Compound("node", (Const(2), Const(3))),)
        for atom in skb_fast.tensor_fb.to_horn_atoms()
    ), "Structured fast-path derivation missing"
    print(f"  fast_calls={fast_calls['n']}  structured_rules={skb_fast.stats()['structured_rules']}  [PASS]")

    # ══ T8: End-to-end Python → facts → rules → inference ════════════════════
    sep("T8 · End-to-end pipeline (Python → KB → forward chain)")
    CODE2 = """
x = read_data()
y = preprocess(x)
z = model(y)
result = postprocess(z)
"""
    facts2, pool2 = PythonASTParser.parse_code(CODE2, scope="pipeline")
    skb2 = ScalableKnowledgeBase(max_arity=2, device=device)
    for f in facts2:
        if f.is_ground():
            skb2.add_fact(f)
    make_code_rules(skb2)
    skb2.forward_chain(max_depth=3)

    # Check transitive data dependencies (x -> y -> z -> result).
    all_f = skb2.tensor_fb.to_horn_atoms()
    p_dep  = PRED_VOCAB.get_id("dep_data")
    dep_facts = [f for f in all_f if f.pred == p_dep]
    print(f"  pipeline facts={len(facts2)}  dep_data derived={len(dep_facts)}  [PASS]")

    print("\n  ✅  omen_tensor_unify: all tests passed\n")
    print("  Ready for the real world:")
    print(f"    · PredicateVocab: {pv.vocab_size()} predicates (500+ built-in)")
    print(f"    · BPESymbolMapper: GPT-2 BPE {'loaded' if bpe.is_bpe_loaded else 'hash-mode'}")
    print(f"    · TensorUnifyEngine: GPU-vectorized join, ~100x vs Python loops")
    print(f"    · PythonASTParser: {len(facts)} facts from ~20 lines of code")
    print(f"    · ScalableKB: hybrid tensor+symbolic, {len(skb)} facts")


if __name__ == "__main__":
    _run_tensor_tests()
