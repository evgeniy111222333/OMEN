from __future__ import annotations

import math
from typing import Optional

import torch


LOG2E = 1.0 / math.log(2.0)


def elias_gamma_bits(value: int) -> float:
    value = max(int(value), 1)
    magnitude = int(math.floor(math.log2(value)))
    return float(2 * magnitude + 1)


def universal_int_bits(value: int) -> float:
    return elias_gamma_bits(max(int(value), 0) + 1)


def universal_float_bits(
    value: float,
    sigma: float = 1.0,
    min_sigma: float = 1e-6,
) -> float:
    sigma = max(float(sigma), float(min_sigma))
    nll_nats = 0.5 * (
        math.log(2.0 * math.pi * sigma * sigma)
        + (float(value) * float(value)) / (sigma * sigma)
    )
    return float(nll_nats * LOG2E)


def gaussian_tensor_bits(
    tensor: torch.Tensor,
    sigma: float = 1.0,
    min_sigma: float = 1e-6,
) -> torch.Tensor:
    sigma = max(float(sigma), float(min_sigma))
    sigma_sq = sigma * sigma
    const_bits = 0.5 * math.log2(2.0 * math.pi * sigma_sq)
    bits = const_bits + 0.5 * tensor.pow(2) / sigma_sq * LOG2E
    return bits.clamp_min(0.0).sum()


def gaussian_nll_bits(
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    nll_nats = 0.5 * (
        logvar
        + (x - mu).pow(2) / logvar.exp().clamp_min(1e-6)
        + math.log(2.0 * math.pi)
    )
    nll_bits = nll_nats * LOG2E
    if reduction == "sum":
        return nll_bits.sum()
    if reduction == "none":
        return nll_bits
    if reduction != "mean":
        raise ValueError(f"Unsupported reduction: {reduction}")
    return nll_bits.mean()


def gaussian_kl_bits(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: Optional[torch.Tensor] = None,
    logvar_p: Optional[torch.Tensor] = None,
    free_bits: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    if mu_p is None:
        mu_p = torch.zeros_like(mu_q)
    if logvar_p is None:
        logvar_p = torch.zeros_like(logvar_q)
    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    kl_nats = 0.5 * (
        logvar_p - logvar_q
        + (var_q + (mu_q - mu_p).pow(2)) / var_p.clamp_min(1e-6)
        - 1.0
    )
    kl_bits = kl_nats * LOG2E
    if free_bits > 0.0:
        kl_bits = kl_bits.clamp_min(float(free_bits))
    if reduction == "sum":
        return kl_bits.sum()
    if reduction == "none":
        return kl_bits
    if reduction != "mean":
        raise ValueError(f"Unsupported reduction: {reduction}")
    return kl_bits.mean(dim=-1).mean()
