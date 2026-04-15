from __future__ import annotations

import torch
import torch.nn as nn


class SymbolicStateIntegrator(nn.Module):
    """
    Two-stage concept/symbolic fusion.

    Stage 1 enriches the concept state with memory before symbolic reasoning.
    Stage 2 lets symbolic state overwrite inconsistent concept dimensions
    instead of acting as a weak additive bias.
    """

    def __init__(self, d_latent: int):
        super().__init__()
        self.pre_mem_gate = nn.Sequential(
            nn.Linear(d_latent * 2, d_latent),
            nn.GELU(),
            nn.Linear(d_latent, d_latent),
            nn.Sigmoid(),
        )
        self.post_mem_gate = nn.Sequential(
            nn.Linear(d_latent * 3, d_latent),
            nn.GELU(),
            nn.Linear(d_latent, d_latent),
            nn.Sigmoid(),
        )
        self.sym_override_gate = nn.Sequential(
            nn.Linear(d_latent * 3, d_latent),
            nn.GELU(),
            nn.Linear(d_latent, d_latent),
            nn.Sigmoid(),
        )

    def pre_symbolic(
        self,
        z_concept: torch.Tensor,
        v_mem: torch.Tensor,
    ) -> torch.Tensor:
        mem_gate = self.pre_mem_gate(torch.cat([z_concept, v_mem], dim=-1))
        return z_concept + mem_gate * v_mem

    def post_symbolic(
        self,
        z_concept: torch.Tensor,
        z_symbolic: torch.Tensor,
        v_mem: torch.Tensor,
    ) -> torch.Tensor:
        mem_gate = self.post_mem_gate(torch.cat([z_concept, z_symbolic, v_mem], dim=-1))
        base_state = z_concept + mem_gate * v_mem
        sym_gate = self.sym_override_gate(
            torch.cat([base_state, z_symbolic, z_symbolic - base_state], dim=-1)
        )
        return base_state + sym_gate * (z_symbolic - base_state)
