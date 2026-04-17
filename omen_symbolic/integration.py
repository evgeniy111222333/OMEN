from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from omen_symbolic.world_graph import WorldGraphState


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
        self.graph_query = nn.Sequential(
            nn.Linear(d_latent * 4, d_latent),
            nn.GELU(),
            nn.Linear(d_latent, d_latent),
        )
        self.graph_state_gate = nn.Sequential(
            nn.Linear(d_latent * 4, d_latent),
            nn.GELU(),
            nn.Linear(d_latent, d_latent),
            nn.Sigmoid(),
        )
        self.program_state_gate = nn.Sequential(
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

    def graph_centered(
        self,
        z_query: torch.Tensor,
        graphs: Sequence[WorldGraphState],
        *,
        program_state: Optional[torch.Tensor] = None,
        graph_mix: float = 0.55,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not graphs:
            zeros = torch.zeros_like(z_query)
            anchor = torch.zeros(
                z_query.size(0),
                device=z_query.device,
                dtype=z_query.dtype,
            )
            return z_query, zeros, anchor

        if program_state is None:
            program_state = torch.zeros_like(z_query)
            use_program_state = False
        else:
            program_state = program_state.to(device=z_query.device, dtype=z_query.dtype)
            use_program_state = True

        graph_readouts = []
        graph_anchors = []
        scale = math.sqrt(float(z_query.size(-1)))
        for idx in range(z_query.size(0)):
            graph = graphs[min(idx, len(graphs) - 1)]
            nodes = graph.node_states.to(device=z_query.device, dtype=z_query.dtype)
            pooled = graph.pooled_state.to(device=z_query.device, dtype=z_query.dtype)
            if nodes.dim() == 1:
                nodes = nodes.unsqueeze(0)
            query = self.graph_query(
                torch.cat(
                    [
                        z_query[idx],
                        pooled,
                        program_state[idx],
                        z_query[idx] - pooled,
                    ],
                    dim=-1,
                )
            )
            attn_scores = torch.matmul(nodes, query) / max(scale, 1.0)
            attn_weights = torch.softmax(attn_scores, dim=0)
            graph_readouts.append(torch.sum(attn_weights.unsqueeze(-1) * nodes, dim=0))
            graph_anchors.append(attn_weights.max())

        graph_readout = torch.stack(graph_readouts, dim=0)
        graph_anchor = torch.stack(graph_anchors, dim=0)
        graph_gate = self.graph_state_gate(
            torch.cat(
                [
                    z_query,
                    graph_readout,
                    program_state,
                    z_query - graph_readout,
                ],
                dim=-1,
            )
        )
        graph_centered_state = z_query + float(graph_mix) * graph_gate * (graph_readout - z_query)
        if use_program_state:
            program_gate = self.program_state_gate(
                torch.cat(
                    [
                        graph_centered_state,
                        graph_readout,
                        program_state,
                    ],
                    dim=-1,
                )
            )
            graph_centered_state = graph_centered_state + program_gate * (
                program_state - graph_centered_state
            )
        return graph_centered_state, graph_readout, graph_anchor
