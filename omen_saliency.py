"""
omen_saliency.py - Saliency Trace bridge for language mode.

Neural attention -> symbolic Horn facts -> S-Core reasoning -> gradients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from omen_prolog import (
    Const,
    HornAtom,
    HornClause,
    Var,
    find_all_substitutions,
    freshen_vars,
)


ROLE_NAMES: Tuple[str, ...] = (
    "agent",
    "patient",
    "action",
    "modifier",
    "coref",
    "other",
)


@dataclass
class SaliencyOutput:
    sal_total: torch.Tensor
    sal_role: torch.Tensor
    sal_struct: torch.Tensor
    sal_consistency_loss: torch.Tensor
    sal_rule_penalty: torch.Tensor
    sal_consistency: float
    sal_tau: float
    sal_observed: int
    sal_expected: int
    sal_edges: int
    sal_abduced: int
    sal_role_targets: torch.Tensor
    sal_graph_latent: torch.Tensor
    sal_raw_facts: List[List[HornAtom]]
    sal_semantic_facts: List[List[HornAtom]]
    sal_expected_facts: List[List[HornAtom]]


class SaliencyGraphEncoder(nn.Module):
    def __init__(self, d_model: int, d_latent: int, max_positions: int, max_preds: int, n_roles: int):
        super().__init__()
        self.pos_emb = nn.Embedding(max_positions, d_model)
        self.pred_emb = nn.Embedding(max_preds, d_model)
        self.role_emb = nn.Embedding(n_roles + 2, d_model)
        self.msg_mlp = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.node_norm = nn.LayerNorm(d_model)
        self.to_latent = nn.Sequential(
            nn.Linear(d_model, d_latent),
            nn.GELU(),
            nn.Linear(d_latent, d_latent),
        )

    def forward(
        self,
        facts_by_batch: Sequence[Sequence[HornAtom]],
        seq_len: int,
        link_pred: int,
        role_pred: int,
        role_id_to_idx: Dict[int, int],
        pred_to_local: Dict[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        if not facts_by_batch:
            return torch.zeros(1, self.to_latent[-1].out_features, device=device)

        latents: List[torch.Tensor] = []
        pos_idx = torch.arange(seq_len, device=device)
        for batch_facts in facts_by_batch:
            nodes = self.pos_emb(pos_idx).clone()
            agg = torch.zeros_like(nodes)
            for fact in batch_facts:
                pred_idx = pred_to_local.get(fact.pred)
                if pred_idx is None:
                    continue
                pred_emb = self.pred_emb(torch.tensor(pred_idx, device=device))
                if fact.pred == role_pred and len(fact.args) >= 2:
                    tok = int(fact.args[0].val)
                    role_idx = role_id_to_idx.get(int(fact.args[1].val), 0)
                    if 0 <= tok < seq_len:
                        role_emb = self.role_emb(torch.tensor(role_idx + 1, device=device))
                        msg = self.msg_mlp(torch.cat([nodes[tok], pred_emb, role_emb], dim=0))
                        agg[tok] = agg[tok] + msg
                    continue

                if len(fact.args) < 2:
                    continue
                a0 = int(fact.args[0].val)
                a1 = int(fact.args[1].val)
                if not (0 <= a0 < seq_len and 0 <= a1 < seq_len):
                    continue
                msg01 = self.msg_mlp(torch.cat([nodes[a0], pred_emb, nodes[a1]], dim=0))
                msg10 = self.msg_mlp(torch.cat([nodes[a1], pred_emb, nodes[a0]], dim=0))
                agg[a1] = agg[a1] + msg01
                if fact.pred != link_pred:
                    agg[a0] = agg[a0] + msg10

            nodes = self.node_norm(nodes + agg)
            latents.append(self.to_latent(nodes.mean(dim=0, keepdim=True)))

        return torch.cat(latents, dim=0)


class SaliencyTraceModule(nn.Module):
    def __init__(self, d_tok: int, d_latent: int, cfg):
        super().__init__()
        if getattr(cfg, "sym_vocab", 0) < 16:
            raise ValueError("SaliencyTraceModule requires sym_vocab >= 16")

        self.d_tok = d_tok
        self.d_latent = d_latent
        self.sym_vocab = int(cfg.sym_vocab)
        self.max_depth = max(int(getattr(cfg, "max_proof_depth", 2)), 1)
        self.top_k = max(int(getattr(cfg, "saliency_top_k", 4)), 1)
        self.max_facts = max(int(getattr(cfg, "saliency_max_facts", 512)), 64)
        self.abduce_every = max(int(getattr(cfg, "saliency_abduce_every", 5)), 1)
        self.consistency_threshold = float(getattr(cfg, "saliency_consistency_threshold", 0.55))
        self.beta_struct = float(getattr(cfg, "saliency_beta_struct", 0.05))
        self.gamma_role = float(getattr(cfg, "saliency_gamma_role", 0.05))
        self.delta_cons = float(getattr(cfg, "saliency_delta_cons", 0.05))
        self.eta_rule = float(getattr(cfg, "saliency_eta_rule", 1e-4))
        tau_init = min(max(float(getattr(cfg, "saliency_tau", 0.20)), 1e-3), 1 - 1e-3)

        self.role_to_idx = {name: idx for idx, name in enumerate(ROLE_NAMES)}
        self.role_classifier = nn.Linear(d_tok, len(ROLE_NAMES))
        self.tau_logit = nn.Parameter(torch.logit(torch.tensor(tau_init)))

        pred_base = self.sym_vocab - 12
        self.pred_link = pred_base + 0
        self.pred_role = pred_base + 1
        self.pred_agent = pred_base + 2
        self.pred_patient = pred_base + 3
        self.pred_modifier = pred_base + 4
        self.pred_coref = pred_base + 5
        self.pred_depends = pred_base + 6

        self._pred_to_local = {
            self.pred_link: 0,
            self.pred_role: 1,
            self.pred_agent: 2,
            self.pred_patient: 3,
            self.pred_modifier: 4,
            self.pred_coref: 5,
            self.pred_depends: 6,
        }
        self._semantic_preds = {
            self.pred_agent,
            self.pred_patient,
            self.pred_modifier,
            self.pred_coref,
            self.pred_depends,
        }
        self._role_const = {
            name: idx + 1 for idx, name in enumerate(ROLE_NAMES)
        }
        self._role_id_to_idx = {
            role_id: idx for idx, role_id in enumerate(self._role_const.values())
        }
        self.graph_encoder = SaliencyGraphEncoder(
            d_model=max(d_latent // 2, 16),
            d_latent=d_latent,
            max_positions=max(int(getattr(cfg, "seq_len", 512)), 32),
            max_preds=len(self._pred_to_local) + 1,
            n_roles=len(ROLE_NAMES),
        )
        self.lang_rules = self._make_language_rules()

    def _make_language_rules(self) -> List[HornClause]:
        X, Y, Z = Var("X"), Var("Y"), Var("Z")
        return [
            HornClause(
                head=HornAtom(self.pred_coref, (X, Z)),
                body=(
                    HornAtom(self.pred_coref, (X, Y)),
                    HornAtom(self.pred_coref, (Y, Z)),
                ),
            ),
            HornClause(
                head=HornAtom(self.pred_depends, (X, Z)),
                body=(
                    HornAtom(self.pred_depends, (X, Y)),
                    HornAtom(self.pred_depends, (Y, Z)),
                ),
            ),
            HornClause(
                head=HornAtom(self.pred_agent, (X, Z)),
                body=(
                    HornAtom(self.pred_agent, (X, Y)),
                    HornAtom(self.pred_coref, (Y, Z)),
                ),
            ),
            HornClause(
                head=HornAtom(self.pred_patient, (X, Z)),
                body=(
                    HornAtom(self.pred_patient, (X, Y)),
                    HornAtom(self.pred_coref, (Y, Z)),
                ),
            ),
            HornClause(
                head=HornAtom(self.pred_modifier, (X, Z)),
                body=(
                    HornAtom(self.pred_modifier, (X, Y)),
                    HornAtom(self.pred_coref, (Y, Z)),
                ),
            ),
        ]

    def _pair_to_semantic_fact(self, src: int, dst: int, src_role: str, dst_role: str) -> Optional[HornAtom]:
        if src_role == "agent" and dst_role == "action":
            return HornAtom(self.pred_agent, (Const(dst), Const(src)))
        if src_role == "action" and dst_role == "patient":
            return HornAtom(self.pred_patient, (Const(src), Const(dst)))
        if src_role == "modifier" and dst_role == "action":
            return HornAtom(self.pred_modifier, (Const(dst), Const(src)))
        if src_role == "coref" and dst_role == "coref":
            return HornAtom(self.pred_coref, (Const(dst), Const(src)))
        if src_role == "action" and dst_role == "action":
            return HornAtom(self.pred_depends, (Const(dst), Const(src)))
        return None

    def _semantic_role_targets(
        self,
        batch_semantic: Sequence[Set[HornAtom]],
        expected_semantic: Sequence[Set[HornAtom]],
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        votes = torch.full(
            (batch_size, seq_len, len(ROLE_NAMES)),
            1e-3,
            dtype=torch.float32,
            device=device,
        )

        def add_vote(batch_idx: int, token_idx: int, role_name: str, weight: float = 1.0) -> None:
            if 0 <= token_idx < seq_len:
                votes[batch_idx, token_idx, self.role_to_idx[role_name]] += weight

        for b_idx, facts in enumerate(batch_semantic):
            for fact in facts:
                if fact.pred == self.pred_agent:
                    add_vote(b_idx, int(fact.args[0].val), "action", 1.0)
                    add_vote(b_idx, int(fact.args[1].val), "agent", 1.0)
                elif fact.pred == self.pred_patient:
                    add_vote(b_idx, int(fact.args[0].val), "action", 1.0)
                    add_vote(b_idx, int(fact.args[1].val), "patient", 1.0)
                elif fact.pred == self.pred_modifier:
                    add_vote(b_idx, int(fact.args[0].val), "action", 1.0)
                    add_vote(b_idx, int(fact.args[1].val), "modifier", 1.0)
                elif fact.pred == self.pred_coref:
                    add_vote(b_idx, int(fact.args[0].val), "coref", 1.0)
                    add_vote(b_idx, int(fact.args[1].val), "coref", 1.0)
                elif fact.pred == self.pred_depends:
                    add_vote(b_idx, int(fact.args[0].val), "action", 1.0)
                    add_vote(b_idx, int(fact.args[1].val), "action", 1.0)

        for b_idx, facts in enumerate(expected_semantic):
            for fact in facts:
                if fact.pred == self.pred_agent:
                    add_vote(b_idx, int(fact.args[0].val), "action", 1.25)
                    add_vote(b_idx, int(fact.args[1].val), "agent", 1.25)
                elif fact.pred == self.pred_patient:
                    add_vote(b_idx, int(fact.args[0].val), "action", 1.25)
                    add_vote(b_idx, int(fact.args[1].val), "patient", 1.25)
                elif fact.pred == self.pred_modifier:
                    add_vote(b_idx, int(fact.args[0].val), "action", 1.25)
                    add_vote(b_idx, int(fact.args[1].val), "modifier", 1.25)
                elif fact.pred == self.pred_coref:
                    add_vote(b_idx, int(fact.args[0].val), "coref", 1.25)
                    add_vote(b_idx, int(fact.args[1].val), "coref", 1.25)
                elif fact.pred == self.pred_depends:
                    add_vote(b_idx, int(fact.args[0].val), "action", 1.25)
                    add_vote(b_idx, int(fact.args[1].val), "action", 1.25)

        return votes / votes.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    @staticmethod
    def _consistency(observed: Set[HornAtom], expected: Set[HornAtom]) -> float:
        union = observed | expected
        if not union:
            return 1.0
        return len(observed & expected) / float(len(union))

    @staticmethod
    def _shares_constant(*atoms: HornAtom) -> bool:
        seen: Set[int] = set()
        for atom in atoms:
            for arg in atom.args:
                val = int(arg.val)
                if val in seen:
                    return True
                seen.add(val)
        return False

    def _generalize_rule(self, head: HornAtom, body: Tuple[HornAtom, HornAtom]) -> Optional[HornClause]:
        const_to_var: Dict[int, Var] = {}
        next_idx = 0

        def map_atom(atom: HornAtom) -> HornAtom:
            nonlocal next_idx
            args: List[Var] = []
            for arg in atom.args:
                key = int(arg.val)
                if key not in const_to_var:
                    const_to_var[key] = Var(f"S{next_idx}")
                    next_idx += 1
                args.append(const_to_var[key])
            return HornAtom(atom.pred, tuple(args))

        head_rule = map_atom(head)
        body_rule = tuple(map_atom(atom) for atom in body)
        head_vars = set(var.name for var in head_rule.args if isinstance(var, Var))
        body_vars = {var.name for atom in body_rule for var in atom.args if isinstance(var, Var)}
        if not head_vars.issubset(body_vars):
            return None
        return HornClause(head=head_rule, body=body_rule)

    def _abduce_trace_rules(
        self,
        observed_semantic: Set[HornAtom],
        expected_semantic: Set[HornAtom],
        prover,
    ) -> int:
        if prover is None:
            return 0
        candidates: List[HornClause] = []
        unexplained = [fact for fact in observed_semantic if fact not in expected_semantic]
        semantic = list(observed_semantic)
        for head in unexplained[:6]:
            for i, b1 in enumerate(semantic):
                for b2 in semantic[i + 1:]:
                    if head == b1 or head == b2:
                        continue
                    if not self._shares_constant(head, b1, b2):
                        continue
                    rule = self._generalize_rule(head, (b1, b2))
                    if rule is not None:
                        candidates.append(rule)

        added = 0
        for rule in candidates[:2]:
            if prover.kb.add_rule(rule):
                added += 1
        return added

    def _reason_expected_facts(
        self,
        facts: Iterable[HornAtom],
        rules: Sequence[HornClause],
    ) -> Set[HornAtom]:
        current: Set[HornAtom] = set(facts)
        max_depth = min(self.max_depth, 3)
        for _ in range(max_depth):
            changed = False
            fact_space = frozenset(current)
            for rule in rules:
                fresh = freshen_vars(rule)
                if not fresh.body:
                    if fresh.head.is_ground() and fresh.head not in current:
                        current.add(fresh.head)
                        changed = True
                    continue
                for sigma in find_all_substitutions(
                    fresh.body,
                    fact_space,
                    max_solutions=128,
                ):
                    derived = sigma.apply_atom(fresh.head)
                    if not derived.is_ground() or derived in current:
                        continue
                    current.add(derived)
                    changed = True
            if not changed:
                break
        return current

    def forward(
        self,
        attn_maps: Optional[torch.Tensor],
        token_hidden: torch.Tensor,
        z_neural: torch.Tensor,
        prover=None,
        train_step: int = 0,
    ) -> SaliencyOutput:
        device = token_hidden.device
        batch_size, seq_len, _ = token_hidden.shape
        zero = torch.zeros(1, device=device).squeeze()

        if attn_maps is None or attn_maps.numel() == 0:
            uniform = torch.full(
                (batch_size, seq_len, len(ROLE_NAMES)),
                1.0 / len(ROLE_NAMES),
                device=device,
            )
            z_sym = torch.zeros(batch_size, self.d_latent, device=device)
            return SaliencyOutput(
                sal_total=zero,
                sal_role=zero,
                sal_struct=zero,
                sal_consistency_loss=zero,
                sal_rule_penalty=zero,
                sal_consistency=1.0,
                sal_tau=float(torch.sigmoid(self.tau_logit).item()),
                sal_observed=0,
                sal_expected=0,
                sal_edges=0,
                sal_abduced=0,
                sal_role_targets=uniform,
                sal_graph_latent=z_sym,
                sal_raw_facts=[[] for _ in range(batch_size)],
                sal_semantic_facts=[[] for _ in range(batch_size)],
                sal_expected_facts=[[] for _ in range(batch_size)],
            )

        tau = torch.sigmoid(self.tau_logit)
        role_logits = self.role_classifier(token_hidden)
        role_probs = role_logits.softmax(dim=-1)
        role_names = ROLE_NAMES
        aggregate = attn_maps.mean(dim=(1, 2))
        aggregate = aggregate.masked_fill(torch.eye(seq_len, device=device, dtype=torch.bool).unsqueeze(0), 0.0)
        top_vals, top_idx = aggregate.topk(k=min(self.top_k, seq_len), dim=-1)
        keep = top_vals > tau

        batch_raw_facts: List[List[HornAtom]] = []
        batch_semantic_sets: List[Set[HornAtom]] = []
        batch_graph_facts: List[List[HornAtom]] = []
        total_edges = 0

        for b_idx in range(batch_size):
            raw_facts: List[HornAtom] = []
            semantic: Set[HornAtom] = set()
            graph_facts: List[HornAtom] = []
            role_idx = role_probs[b_idx].argmax(dim=-1)

            for tok_idx, ridx in enumerate(role_idx.tolist()):
                role_id = self._role_const[role_names[ridx]]
                fact = HornAtom(self.pred_role, (Const(tok_idx), Const(role_id)))
                raw_facts.append(fact)
                graph_facts.append(fact)

            for dst_idx in range(seq_len):
                for edge_rank in range(top_vals.size(-1)):
                    if not keep[b_idx, dst_idx, edge_rank]:
                        continue
                    src_idx = int(top_idx[b_idx, dst_idx, edge_rank].item())
                    raw_link = HornAtom(self.pred_link, (Const(src_idx), Const(dst_idx)))
                    raw_facts.append(raw_link)
                    graph_facts.append(raw_link)
                    total_edges += 1

                    src_role = role_names[int(role_idx[src_idx].item())]
                    dst_role = role_names[int(role_idx[dst_idx].item())]
                    sem_fact = self._pair_to_semantic_fact(src_idx, dst_idx, src_role, dst_role)
                    if sem_fact is not None:
                        semantic.add(sem_fact)
                        graph_facts.append(sem_fact)

            batch_raw_facts.append(raw_facts)
            batch_semantic_sets.append(semantic)
            batch_graph_facts.append(graph_facts[:self.max_facts])

        expected_semantic: List[Set[HornAtom]] = []
        consistency_scores: List[float] = []
        rule_penalties: List[float] = []
        abduced_total = 0

        copied_rules = list(prover.kb.rules) if prover is not None else []
        reasoning_rules = list(self.lang_rules) + copied_rules
        if prover is not None:
            base_rule_penalty = float(prover.kb.utility_adjusted_penalty(eta=0.1))
        else:
            base_rule_penalty = 0.0
        base_rule_penalty += float(sum(rule.complexity() for rule in self.lang_rules))
        for b_idx in range(batch_size):
            all_facts = self._reason_expected_facts(
                list(batch_raw_facts[b_idx]) + list(batch_semantic_sets[b_idx]),
                reasoning_rules,
            )
            entailed_sem = {
                fact for fact in all_facts
                if fact.pred in self._semantic_preds
            }
            expected_semantic.append(entailed_sem)
            consistency = self._consistency(batch_semantic_sets[b_idx], entailed_sem)
            consistency_scores.append(consistency)
            rule_penalties.append(base_rule_penalty)

            if consistency < self.consistency_threshold and (train_step % self.abduce_every == 0):
                abduced_total += self._abduce_trace_rules(
                    batch_semantic_sets[b_idx],
                    entailed_sem,
                    prover,
                )

        role_targets = self._semantic_role_targets(
            batch_semantic_sets,
            expected_semantic,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
        )
        role_loss = F.kl_div(
            F.log_softmax(role_logits, dim=-1),
            role_targets,
            reduction="batchmean",
        )

        z_graph = self.graph_encoder(
            batch_graph_facts,
            seq_len=seq_len,
            link_pred=self.pred_link,
            role_pred=self.pred_role,
            role_id_to_idx=self._role_id_to_idx,
            pred_to_local=self._pred_to_local,
            device=device,
        )
        struct_loss = F.mse_loss(z_neural, z_graph)

        mean_consistency = sum(consistency_scores) / max(len(consistency_scores), 1)
        consistency_loss = -torch.log(torch.tensor(mean_consistency, device=device).clamp_min(1e-6))
        rule_penalty = torch.tensor(sum(rule_penalties) / max(len(rule_penalties), 1), device=device)

        total = (
            self.beta_struct * struct_loss
            + self.gamma_role * role_loss
            + self.delta_cons * consistency_loss
            + self.eta_rule * rule_penalty
        )

        return SaliencyOutput(
            sal_total=total,
            sal_role=role_loss,
            sal_struct=struct_loss,
            sal_consistency_loss=consistency_loss,
            sal_rule_penalty=rule_penalty,
            sal_consistency=float(mean_consistency),
            sal_tau=float(tau.item()),
            sal_observed=sum(len(facts) for facts in batch_semantic_sets),
            sal_expected=sum(len(facts) for facts in expected_semantic),
            sal_edges=total_edges,
            sal_abduced=abduced_total,
            sal_role_targets=role_targets,
            sal_graph_latent=z_graph,
            sal_raw_facts=[list(facts) for facts in batch_raw_facts],
            sal_semantic_facts=[list(facts) for facts in batch_semantic_sets],
            sal_expected_facts=[list(facts) for facts in expected_semantic],
        )


def run_saliency_tests() -> None:
    from omen_scale_config import OMENScaleConfig

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OMENScaleConfig.demo()
    mod = SaliencyTraceModule(d_tok=8, d_latent=16, cfg=cfg).to(device)

    with torch.no_grad():
        mod.role_classifier.weight.zero_()
        mod.role_classifier.bias.zero_()
        for ridx in range(len(ROLE_NAMES)):
            mod.role_classifier.weight[ridx, ridx] = 4.0

    token_hidden = torch.zeros(1, 5, 8, device=device, requires_grad=True)
    token_hidden.data[0, 0, mod.role_to_idx["action"]] = 1.0
    token_hidden.data[0, 1, mod.role_to_idx["action"]] = 1.0
    token_hidden.data[0, 2, mod.role_to_idx["action"]] = 1.0
    token_hidden.data[0, 3, mod.role_to_idx["agent"]] = 1.0
    token_hidden.data[0, 4, mod.role_to_idx["patient"]] = 1.0
    z_neural = torch.randn(1, 16, device=device, requires_grad=True)

    attn = torch.zeros(1, 2, 2, 5, 5, device=device)
    attn[:, :, :, 0, 3] = 0.95   # agent(0,3)
    attn[:, :, :, 4, 0] = 0.95   # patient(0,4)
    attn[:, :, :, 1, 0] = 0.95   # depends(1,0)
    attn[:, :, :, 2, 1] = 0.95   # depends(2,1)

    out = mod(attn, token_hidden, z_neural, prover=None, train_step=0)
    assert out.sal_observed >= 4, f"Observed facts too small: {out.sal_observed}"
    assert out.sal_expected >= 1, f"Expected facts too small: {out.sal_expected}"
    assert out.sal_consistency > 0.15, f"Consistency too low: {out.sal_consistency}"
    assert out.sal_role_targets.shape == (1, 5, len(ROLE_NAMES))
    loss = out.sal_total + out.sal_struct + out.sal_role
    loss.backward()
    assert token_hidden.grad is not None and token_hidden.grad.norm().item() > 0
    assert z_neural.grad is not None and z_neural.grad.norm().item() > 0

    head = HornAtom(mod.pred_depends, (Const(2), Const(0)))
    body = (
        HornAtom(mod.pred_depends, (Const(2), Const(1))),
        HornAtom(mod.pred_depends, (Const(1), Const(0))),
    )
    rule = mod._generalize_rule(head, body)
    assert rule is not None and len(rule.body) == 2
    print("Saliency tests passed.")


if __name__ == "__main__":
    run_saliency_tests()
