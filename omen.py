from __future__ import annotations

from typing import Any

from omen_canonical import (
    CANONICAL_OMEN_SPEC,
    OMEN_REPOSITORY_AXIS,
    CanonicalArchitectureSpec,
    RepositoryAxisSpec,
    canonical_module_role,
)
from omen_scale import OMENScale
from omen_scale_config import OMENScaleConfig


OMEN = OMENScale
OMENConfig = OMENScaleConfig


def build_omen(
    cfg: OMENScaleConfig | None = None,
    *,
    device: Any | None = None,
    eval_mode: bool = False,
) -> OMENScale:
    model = OMENScale(cfg if cfg is not None else OMENScaleConfig.demo())
    if device is not None:
        model = model.to(device)
    if eval_mode:
        model.eval()
    return model


def canonical_architecture() -> CanonicalArchitectureSpec:
    return CANONICAL_OMEN_SPEC


def repository_axis() -> RepositoryAxisSpec:
    return OMEN_REPOSITORY_AXIS


def module_role(module_path: str) -> str:
    return canonical_module_role(module_path)


__all__ = [
    "OMEN",
    "OMENConfig",
    "build_omen",
    "canonical_architecture",
    "module_role",
    "repository_axis",
]
