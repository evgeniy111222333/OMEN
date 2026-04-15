from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any, Callable, Dict, Hashable, List, Sequence, Set, Tuple, TypeVar


AtomT = TypeVar("AtomT")
TermConstFn = Callable[[Any], Set[int]]


def atom_const_set(atom: AtomT, term_const_values: TermConstFn) -> Set[int]:
    values: Set[int] = set()
    for arg in getattr(atom, "args", ()):
        values.update(term_const_values(arg))
    return values


def shares_constant(*atoms: AtomT, term_const_values: TermConstFn) -> bool:
    seen: Set[int] = set()
    for atom in atoms:
        for value in atom_const_set(atom, term_const_values):
            if value in seen:
                return True
            seen.add(value)
    return False


def body_is_goal_connected(
    head: AtomT,
    body: Tuple[AtomT, ...],
    term_const_values: TermConstFn,
) -> bool:
    if not body:
        return False
    atoms: Tuple[AtomT, ...] = (head,) + tuple(body)
    frontier: List[int] = [0]
    visited: Set[int] = {0}
    while frontier:
        src = frontier.pop()
        for dst in range(1, len(atoms)):
            if dst in visited:
                continue
            if shares_constant(atoms[src], atoms[dst], term_const_values=term_const_values):
                visited.add(dst)
                frontier.append(dst)
    return len(visited) == len(atoms)


def body_overlap_score(
    head: AtomT,
    body: Tuple[AtomT, ...],
    term_const_values: TermConstFn,
) -> int:
    head_consts = atom_const_set(head, term_const_values)
    body_union: Set[int] = set()
    body_overlap = 0
    link_bonus = 0
    for atom in body:
        atom_consts = atom_const_set(atom, term_const_values)
        body_union.update(atom_consts)
        body_overlap += len(head_consts & atom_consts)
    for atom_a, atom_b in combinations(body, 2):
        if shares_constant(atom_a, atom_b, term_const_values=term_const_values):
            link_bonus += 1
    spans_all_head_consts = 1 if head_consts and head_consts.issubset(body_union) else 0
    return (
        4 * len(head_consts & body_union)
        + 2 * body_overlap
        + 3 * link_bonus
        + 5 * spans_all_head_consts
        - len(body)
    )


def bridge_path_bodies(
    example_head: AtomT,
    facts: Sequence[AtomT],
    term_const_values: TermConstFn,
    max_body_atoms: int = 3,
    max_fact_scan: int = 14,
    max_paths: int = 16,
) -> List[Tuple[int, Tuple[AtomT, ...]]]:
    head_consts = sorted(atom_const_set(example_head, term_const_values))
    if len(head_consts) < 2:
        return []
    fact_subset = list(facts[:max_fact_scan])
    fact_consts = [atom_const_set(atom, term_const_values) for atom in fact_subset]
    adjacency: Dict[int, Set[int]] = defaultdict(set)
    for i, const_i in enumerate(fact_consts):
        for j in range(i + 1, len(fact_consts)):
            if const_i & fact_consts[j]:
                adjacency[i].add(j)
                adjacency[j].add(i)

    ranked: List[Tuple[int, Tuple[AtomT, ...]]] = []
    seen_paths: Set[Tuple[int, ...]] = set()
    for src_const, dst_const in combinations(head_consts, 2):
        start_nodes = [i for i, consts in enumerate(fact_consts) if src_const in consts]
        goal_nodes = {i for i, consts in enumerate(fact_consts) if dst_const in consts}
        if not start_nodes or not goal_nodes:
            continue
        stack: List[Tuple[int, List[int]]] = [(node, [node]) for node in start_nodes]
        while stack and len(ranked) < max_paths:
            node, path = stack.pop()
            if node in goal_nodes:
                key = tuple(path)
                if key in seen_paths:
                    continue
                seen_paths.add(key)
                body = tuple(fact_subset[idx] for idx in path)
                if not body_is_goal_connected(
                    example_head,
                    body,
                    term_const_values=term_const_values,
                ):
                    continue
                score = body_overlap_score(
                    example_head,
                    body,
                    term_const_values=term_const_values,
                ) + 8
                ranked.append((score, body))
                continue
            if len(path) >= max_body_atoms:
                continue
            for nxt in sorted(adjacency[node]):
                if nxt in path:
                    continue
                stack.append((nxt, path + [nxt]))
    ranked.sort(key=lambda item: (item[0], -len(item[1])), reverse=True)
    return ranked


def _term_shape_key(term: Any) -> Hashable:
    if hasattr(term, "func") and hasattr(term, "subterms"):
        return (
            "compound",
            getattr(term, "func"),
            tuple(_term_shape_key(subterm) for subterm in getattr(term, "subterms")),
        )
    if hasattr(term, "val"):
        return ("const",)
    if hasattr(term, "name"):
        return ("var",)
    return ("term", type(term).__name__)


def _body_template_signature(
    head: AtomT,
    body: Tuple[AtomT, ...],
    term_const_values: TermConstFn,
) -> Hashable:
    head_consts = sorted(atom_const_set(head, term_const_values))
    head_roles = {const_value: f"H{idx}" for idx, const_value in enumerate(head_consts)}
    body_counts: Dict[int, int] = defaultdict(int)
    for atom in body:
        for value in atom_const_set(atom, term_const_values):
            if value not in head_roles:
                body_counts[value] += 1
    bridge_values = {
        value: f"B{idx}"
        for idx, value in enumerate(sorted(v for v, count in body_counts.items() if count > 1))
    }

    def arg_signature(term: Any) -> Hashable:
        labels: List[str] = []
        for value in sorted(term_const_values(term)):
            if value in head_roles:
                labels.append(head_roles[value])
            elif value in bridge_values:
                labels.append(bridge_values[value])
            else:
                labels.append("L")
        return (_term_shape_key(term), tuple(labels))

    atom_signatures: List[Hashable] = []
    for atom in body:
        atom_signatures.append(
            (
                int(getattr(atom, "pred")),
                len(getattr(atom, "args", ())),
                tuple(arg_signature(arg) for arg in getattr(atom, "args", ())),
            )
        )
    return tuple(atom_signatures)


def structural_template_bodies(
    example_head: AtomT,
    facts: Sequence[AtomT],
    term_const_values: TermConstFn,
    max_body_atoms: int = 3,
    max_fact_scan: int = 14,
    max_candidates: int = 24,
) -> List[Tuple[int, Tuple[AtomT, ...]]]:
    fact_subset = list(facts[:max_fact_scan])
    best_by_template: Dict[Hashable, Tuple[int, Tuple[AtomT, ...]]] = {}
    for body_len in range(1, max_body_atoms + 1):
        for body in combinations(fact_subset, body_len):
            if not body_is_goal_connected(
                example_head,
                body,
                term_const_values=term_const_values,
            ):
                continue
            template_sig = _body_template_signature(
                example_head,
                body,
                term_const_values=term_const_values,
            )
            bridge_bonus = sum(
                1
                for atom_a, atom_b in combinations(body, 2)
                if shares_constant(atom_a, atom_b, term_const_values=term_const_values)
            )
            score = body_overlap_score(
                example_head,
                body,
                term_const_values=term_const_values,
            ) + 2 * bridge_bonus
            current = best_by_template.get(template_sig)
            if current is None or score > current[0]:
                best_by_template[template_sig] = (score, tuple(body))
    ranked = list(best_by_template.values())
    ranked.sort(key=lambda item: (item[0], -len(item[1])), reverse=True)
    return ranked[:max_candidates]


def rank_goal_directed_bodies(
    example_head: AtomT,
    facts: Sequence[AtomT],
    term_const_values: TermConstFn,
    max_body_atoms: int = 3,
    max_fact_scan: int = 14,
    max_paths: int = 16,
    max_templates: int = 24,
) -> List[Tuple[int, Tuple[AtomT, ...]]]:
    ranked: List[Tuple[int, Tuple[AtomT, ...]]] = []
    seen: Set[Tuple[int, ...]] = set()
    for score, body in bridge_path_bodies(
        example_head,
        facts,
        term_const_values=term_const_values,
        max_body_atoms=max_body_atoms,
        max_fact_scan=max_fact_scan,
        max_paths=max_paths,
    ):
        key = tuple(hash(atom) for atom in body)
        if key in seen:
            continue
        seen.add(key)
        ranked.append((score, body))
    for score, body in structural_template_bodies(
        example_head,
        facts,
        term_const_values=term_const_values,
        max_body_atoms=max_body_atoms,
        max_fact_scan=max_fact_scan,
        max_candidates=max_templates,
    ):
        key = tuple(hash(atom) for atom in body)
        if key in seen:
            continue
        seen.add(key)
        ranked.append((score, body))
    ranked.sort(key=lambda item: (item[0], -len(item[1])), reverse=True)
    return ranked


def rule_template_signature(head: Any, body: Tuple[Any, ...]) -> Hashable:
    var_ids: Dict[str, int] = {}

    def canonical_term(term: Any) -> Hashable:
        if hasattr(term, "name") and not hasattr(term, "func") and not hasattr(term, "val"):
            name = str(getattr(term, "name"))
            if name not in var_ids:
                var_ids[name] = len(var_ids)
            return ("var", var_ids[name])
        if hasattr(term, "func") and hasattr(term, "subterms"):
            return (
                "compound",
                getattr(term, "func"),
                tuple(canonical_term(subterm) for subterm in getattr(term, "subterms")),
            )
        if hasattr(term, "val"):
            return ("const",)
        return ("term", type(term).__name__)

    def canonical_atom(atom: Any) -> Hashable:
        return (
            int(getattr(atom, "pred")),
            len(getattr(atom, "args", ())),
            tuple(canonical_term(arg) for arg in getattr(atom, "args", ())),
        )

    return (
        canonical_atom(head),
        tuple(canonical_atom(atom) for atom in body),
    )


def bridge_variable_count(head: Any, body: Tuple[Any, ...]) -> int:
    head_vars: Set[str] = set()
    body_counts: Dict[str, int] = defaultdict(int)
    for arg in getattr(head, "args", ()):
        if hasattr(arg, "name") and not hasattr(arg, "func") and not hasattr(arg, "val"):
            head_vars.add(str(getattr(arg, "name")))
    for atom in body:
        for arg in getattr(atom, "args", ()):
            if hasattr(arg, "name") and not hasattr(arg, "func") and not hasattr(arg, "val"):
                body_counts[str(getattr(arg, "name"))] += 1
    return sum(
        1
        for name, count in body_counts.items()
        if count > 1 and name not in head_vars
    )
