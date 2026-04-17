from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryInterface(Protocol):
    def read(self, z_query: torch.Tensor) -> torch.Tensor: ...

    def episodic_recall(self, z_query: torch.Tensor, k: int = 4) -> torch.Tensor: ...


@dataclass
class OMENCoreConfig:
    vocab_size: int = 256
    d_latent: int = 64
    world_rnn_hidden: int = 128
    epistemic_tau: float = 0.3
    epistemic_exact_grad: bool = False
    n_counterfactual: int = 2


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
        self.h0 = nn.Parameter(torch.zeros(1, cfg.world_rnn_hidden))

    def forward(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        a = self.act_emb(action)
        h = self.h0.expand(z.size(0), -1) if h is None else h
        h2 = self.gru(torch.cat([z, a], dim=-1), h)
        return self.out(h2), h2

    def simulate_sequence(
        self,
        z0: torch.Tensor,
        actions: torch.Tensor,
        teacher_forcing_ratio: float = 0.0,
        teacher_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, steps = actions.shape
        action_embs = self.act_emb(actions)
        hidden = self.h0.expand(batch_size, -1).contiguous()
        trajectory = []
        z_prev = z0
        if teacher_states is not None and teacher_states.shape[:2] != (batch_size, steps):
            raise ValueError("teacher_states must have shape (B, T, d_latent)")
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
            hidden = self.gru(torch.cat([z_in, action_embs[:, step]], dim=-1), hidden)
            z_prev = self.out(hidden)
            trajectory.append(z_prev)
        return torch.stack(trajectory, dim=1)


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
            epistemic_map = diff.pow(2)
            gap_norm = epistemic_map.sum(dim=-1).sqrt()

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
