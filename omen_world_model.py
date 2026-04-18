from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from omen_symbolic.world_graph import WorldGraphBatch


class MemoryInterface(Protocol):
    def read(self, z_query: torch.Tensor) -> torch.Tensor: ...

    def episodic_recall(self, z_query: torch.Tensor, k: int = 4) -> torch.Tensor: ...


@dataclass
class OMENCoreConfig:
    vocab_size: int = 256
    d_latent: int = 64
    world_rnn_hidden: int = 128
    world_graph_transition_mix: float = 0.2
    epistemic_tau: float = 0.3
    epistemic_exact_grad: bool = False
    n_counterfactual: int = 2


@dataclass
class WorldTransitionResult:
    z_next: torch.Tensor
    hidden: torch.Tensor
    causal_error: torch.Tensor
    graph_alignment: torch.Tensor
    state_residual: torch.Tensor


class WorldRNN(nn.Module):
    def __init__(self, cfg: OMENCoreConfig):
        super().__init__()
        self.act_emb = nn.Embedding(cfg.vocab_size, cfg.d_latent)
        self.gru = nn.GRUCell(cfg.d_latent * 2, cfg.world_rnn_hidden)
        self.out = nn.Sequential(
            nn.Linear(cfg.world_rnn_hidden, cfg.d_latent * 2),
            nn.GELU(),
            nn.Linear(cfg.d_latent * 2, cfg.d_latent),
        )
        self.graph_memory_proj = nn.Sequential(
            nn.Linear(cfg.d_latent * 3, cfg.d_latent),
            nn.GELU(),
            nn.Linear(cfg.d_latent, cfg.d_latent),
        )
        self.graph_memory_gate = nn.Sequential(
            nn.Linear(cfg.d_latent * 3, cfg.d_latent),
            nn.GELU(),
            nn.Linear(cfg.d_latent, cfg.d_latent),
            nn.Sigmoid(),
        )
        self.state_refine_gate = nn.Sequential(
            nn.Linear(cfg.d_latent * 3, cfg.d_latent),
            nn.GELU(),
            nn.Linear(cfg.d_latent, cfg.d_latent),
            nn.Sigmoid(),
        )
        self.h0 = nn.Parameter(torch.zeros(1, cfg.world_rnn_hidden))
        self.graph_transition_mix = float(getattr(cfg, "world_graph_transition_mix", 0.2))

    @staticmethod
    def _fit_graph_sequence(
        seq: Optional[torch.Tensor],
        steps: int,
        *,
        pad_with_first: bool = True,
    ) -> Optional[torch.Tensor]:
        if seq is None or seq.numel() == 0 or steps <= 0:
            return None
        if seq.size(0) >= steps:
            return seq[-steps:]
        pad_steps = steps - seq.size(0)
        pad_ref = seq[:1] if pad_with_first else seq[-1:]
        return torch.cat([pad_ref.expand(pad_steps, -1), seq], dim=0)

    def _resolve_graph_contexts(
        self,
        world_graph_batch: Optional[WorldGraphBatch],
        steps: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        pad_with_first: bool = True,
    ) -> Optional[torch.Tensor]:
        if (
            steps <= 0
            or world_graph_batch is None
            or not world_graph_batch.graphs
            or world_graph_batch.pooled_states.numel() == 0
        ):
            return None
        graph_contexts = world_graph_batch.pooled_states.to(device=device, dtype=dtype)
        graph_contexts = graph_contexts.unsqueeze(1).expand(-1, steps, -1).clone()
        for batch_idx, graph in enumerate(world_graph_batch.graphs):
            trace_context = self._fit_graph_sequence(
                None if graph.transition_states is None else graph.transition_states.detach().to(device=device, dtype=dtype),
                steps,
                pad_with_first=pad_with_first,
            )
            if trace_context is not None:
                graph_contexts[batch_idx] = trace_context
        return graph_contexts

    def _graph_conditioned_state(
        self,
        z: torch.Tensor,
        graph_context: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if graph_context is None or self.graph_transition_mix <= 0.0:
            return z
        graph_context = graph_context.to(device=z.device, dtype=z.dtype)
        graph_delta = torch.tanh(graph_context - z)
        return z + self.graph_transition_mix * graph_delta

    def _machine_state(
        self,
        z: torch.Tensor,
        graph_context: Optional[torch.Tensor],
    ) -> torch.Tensor:
        base_state = self._graph_conditioned_state(z, graph_context)
        if graph_context is None:
            return base_state
        graph_context = graph_context.to(device=z.device, dtype=z.dtype)
        features = torch.cat(
            [base_state, graph_context, graph_context - base_state],
            dim=-1,
        )
        graph_memory = self.graph_memory_proj(features)
        graph_gate = self.graph_memory_gate(features)
        return base_state + graph_gate * graph_memory

    def _refine_state(
        self,
        z_next: torch.Tensor,
        graph_context: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if graph_context is None:
            return z_next
        graph_context = graph_context.to(device=z_next.device, dtype=z_next.dtype)
        features = torch.cat(
            [z_next, graph_context, graph_context - z_next],
            dim=-1,
        )
        refine_gate = self.state_refine_gate(features)
        refined = z_next + refine_gate * (graph_context - z_next)
        return self._graph_conditioned_state(refined, graph_context)

    def _action_state(
        self,
        *,
        action: Optional[torch.Tensor],
        action_probs: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if action_probs is not None:
            probs = action_probs.to(device=device, dtype=dtype)
            if probs.dim() == 1:
                probs = probs.unsqueeze(0).expand(batch_size, -1)
            elif probs.size(0) != batch_size:
                probs = probs.expand(batch_size, -1)
            vocab = self.act_emb.num_embeddings
            if probs.size(-1) < vocab:
                probs = F.pad(probs, (0, vocab - probs.size(-1)))
            elif probs.size(-1) > vocab:
                probs = probs[..., :vocab]
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            weight = self.act_emb.weight.to(device=device, dtype=dtype)
            return probs @ weight
        if action is None:
            return torch.zeros(batch_size, self.act_emb.embedding_dim, device=device, dtype=dtype)
        action = action.to(device=device, dtype=torch.long)
        if action.dim() == 0:
            action = action.view(1).expand(batch_size)
        elif action.dim() == 1 and action.size(0) == 1 and batch_size > 1:
            action = action.expand(batch_size)
        return self.act_emb(action)

    def _causal_diagnostics(
        self,
        z_next: torch.Tensor,
        *,
        graph_context: Optional[torch.Tensor],
        target_state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reference = target_state if target_state is not None else graph_context
        if reference is None:
            zeros = torch.zeros(z_next.size(0), device=z_next.device, dtype=z_next.dtype)
            return zeros, torch.ones_like(zeros), torch.zeros_like(z_next)
        reference = reference.to(device=z_next.device, dtype=z_next.dtype)
        if reference.shape != z_next.shape:
            reference = reference.expand_as(z_next)
        state_residual = z_next - reference
        smooth_error = F.smooth_l1_loss(z_next, reference, reduction="none").mean(dim=-1)
        graph_alignment = (
            F.cosine_similarity(z_next, reference, dim=-1)
            .clamp(-1.0, 1.0)
            .add(1.0)
            .mul(0.5)
        )
        causal_error = (0.5 * smooth_error + 0.5 * (1.0 - graph_alignment)).clamp(0.0, 1.0)
        return causal_error, graph_alignment, state_residual

    def transition(
        self,
        z: torch.Tensor,
        *,
        action: Optional[torch.Tensor] = None,
        action_probs: Optional[torch.Tensor] = None,
        h: Optional[torch.Tensor] = None,
        graph_context: Optional[torch.Tensor] = None,
        target_state: Optional[torch.Tensor] = None,
    ) -> WorldTransitionResult:
        batch_size = z.size(0)
        action_state = self._action_state(
            action=action,
            action_probs=action_probs,
            batch_size=batch_size,
            device=z.device,
            dtype=z.dtype,
        )
        hidden = self.h0.expand(batch_size, -1) if h is None else h
        machine_state = self._machine_state(z, graph_context)
        h2 = self.gru(torch.cat([machine_state, action_state], dim=-1), hidden)
        z_next = self._refine_state(self.out(h2), graph_context)
        causal_error, graph_alignment, state_residual = self._causal_diagnostics(
            z_next,
            graph_context=graph_context,
            target_state=target_state,
        )
        return WorldTransitionResult(
            z_next=z_next,
            hidden=h2,
            causal_error=causal_error,
            graph_alignment=graph_alignment,
            state_residual=state_residual,
        )

    def forward(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        graph_context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        result = self.transition(
            z,
            action=action,
            h=h,
            graph_context=graph_context,
        )
        return result.z_next, result.hidden

    def simulate_sequence(
        self,
        z0: torch.Tensor,
        actions: torch.Tensor,
        teacher_forcing_ratio: float = 0.0,
        teacher_states: Optional[torch.Tensor] = None,
        graph_contexts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, steps = actions.shape
        hidden = self.h0.expand(batch_size, -1).contiguous()
        trajectory = []
        z_prev = z0
        if teacher_states is not None and teacher_states.shape[:2] != (batch_size, steps):
            raise ValueError("teacher_states must have shape (B, T, d_latent)")
        if graph_contexts is not None and graph_contexts.shape[:2] != (batch_size, steps):
            raise ValueError("graph_contexts must have shape (B, T, d_latent)")
        for step in range(steps):
            z_in = z_prev
            if teacher_states is not None:
                z_teacher = teacher_states[:, step]
            else:
                z_teacher = z_prev if step > 0 else z0
            if teacher_states is not None and step == 0:
                z_in = z_teacher
            elif step > 0 and teacher_forcing_ratio > 0.0 and self.training:
                use_teacher = (
                    torch.rand(batch_size, 1, device=z0.device) < teacher_forcing_ratio
                ).to(z0.dtype)
                z_in = use_teacher * z_teacher + (1.0 - use_teacher) * z_prev
            graph_context = None if graph_contexts is None else graph_contexts[:, step]
            result = self.transition(
                z_in,
                action=actions[:, step],
                h=hidden,
                graph_context=graph_context,
                target_state=z_teacher if teacher_states is not None else graph_context,
            )
            hidden = result.hidden
            z_prev = result.z_next
            trajectory.append(z_prev)
        return torch.stack(trajectory, dim=1)

    def simulate_graph_sequence(
        self,
        actions: torch.Tensor,
        *,
        world_graph_batch: Optional[WorldGraphBatch],
        z0: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
        teacher_states: Optional[torch.Tensor] = None,
        graph_contexts: Optional[torch.Tensor] = None,
        pad_with_first: bool = True,
    ) -> torch.Tensor:
        batch_size, steps = actions.shape
        if graph_contexts is None:
            ref_dtype = z0.dtype if z0 is not None else (
                teacher_states.dtype if teacher_states is not None else self.h0.dtype
            )
            graph_contexts = self._resolve_graph_contexts(
                world_graph_batch,
                steps,
                device=actions.device,
                dtype=ref_dtype,
                pad_with_first=pad_with_first,
            )
        if z0 is None:
            if teacher_states is not None:
                z0 = teacher_states[:, 0]
            elif graph_contexts is not None:
                z0 = graph_contexts[:, 0]
            else:
                raise ValueError("z0 or graph_contexts/teacher_states are required for graph rollout")
        if z0.size(0) != batch_size:
            raise ValueError("z0 batch size must match actions batch size")
        return self.simulate_sequence(
            z0,
            actions,
            teacher_forcing_ratio=teacher_forcing_ratio,
            teacher_states=teacher_states,
            graph_contexts=graph_contexts,
        )


class EpistemicGapDetector(nn.Module):
    def __init__(self, cfg: OMENCoreConfig):
        super().__init__()
        self.tau = cfg.epistemic_tau
        self.exact_grad = bool(getattr(cfg, "epistemic_exact_grad", False))
        self.d = cfg.d_latent

    def compute(
        self,
        z: torch.Tensor,
        world_rnn: WorldRNN,
        z_sim: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.exact_grad and torch.is_grad_enabled() and z.requires_grad:
            world_loss = F.mse_loss(z, z_sim)
            grad_z = torch.autograd.grad(
                world_loss,
                z,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )[0]
            epistemic_map = grad_z.detach().pow(2)
            gap_norm = epistemic_map.sum(dim=-1).sqrt()
        else:
            diff = z.detach() - z_sim.detach()
            alignment_gap = (
                1.0 - F.cosine_similarity(z.detach(), z_sim.detach(), dim=-1).clamp(-1.0, 1.0)
            ).unsqueeze(-1).mul(0.5)
            epistemic_map = diff.pow(2) * (1.0 + alignment_gap)
            gap_norm = epistemic_map.sum(dim=-1).sqrt() + alignment_gap.squeeze(-1)

        threshold = epistemic_map.quantile(0.75, dim=-1, keepdim=True)
        hot_dims = (epistemic_map >= threshold).float()
        return epistemic_map.detach(), gap_norm.detach(), hot_dims.detach()


class CuriosityModule(nn.Module):
    def __init__(self, cfg: OMENCoreConfig):
        super().__init__()
        d = cfg.d_latent
        self.query_proj = nn.Linear(d, d)
        self.fusion = nn.Linear(d * 2, d)
        self.n_cf = cfg.n_counterfactual
        self.tau = cfg.epistemic_tau
        self.vocab_size = cfg.vocab_size
        self.unknown_flag_count = 0

    def forward(
        self,
        z: torch.Tensor,
        epistemic_map: torch.Tensor,
        hot_dims: torch.Tensor,
        gap_norm: torch.Tensor,
        memory: MemoryInterface,
        world_rnn: WorldRNN,
        counterfactual_actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del epistemic_map
        batch_size = z.size(0)
        active = gap_norm > self.tau
        if not active.any():
            return z, torch.tensor(0.0, device=z.device)

        query = self.query_proj(z * hot_dims)
        v_mem = memory.read(query)
        v_epi = memory.episodic_recall(query, k=4)
        v_combined = 0.5 * (v_mem + v_epi)

        cf_loss = torch.tensor(0.0, device=z.device)
        if self.n_cf > 0 and self.training:
            if counterfactual_actions is not None and counterfactual_actions.numel() > 0:
                noise_actions = counterfactual_actions.to(device=z.device, dtype=torch.long)
                if noise_actions.dim() == 1:
                    noise_actions = noise_actions.unsqueeze(0).expand(batch_size, -1)
                if noise_actions.size(1) < self.n_cf:
                    repeats = math.ceil(self.n_cf / max(noise_actions.size(1), 1))
                    noise_actions = noise_actions.repeat(1, repeats)
                noise_actions = noise_actions[:, : self.n_cf]
            else:
                noise_actions = torch.randint(
                    0,
                    self.vocab_size,
                    (batch_size, self.n_cf),
                    device=z.device,
                )
            z_cf_traj = world_rnn.simulate_sequence(z.detach(), noise_actions)
            z_target = (z + v_combined).detach()
            cf_loss = F.mse_loss(z_cf_traj.mean(dim=1), z_target)

        mem_signal_norm = v_combined.norm(dim=-1)
        unknown = active & (mem_signal_norm < 1e-3)
        if unknown.any():
            self.unknown_flag_count += int(unknown.sum().item())

        z_enriched = self.fusion(torch.cat([z, v_combined], dim=-1))
        z_out = torch.where(active.unsqueeze(-1), z_enriched, z)
        return z_out, cf_loss
