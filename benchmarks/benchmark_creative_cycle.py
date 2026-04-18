from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omen_prolog import Const, DifferentiableProver, EpistemicStatus, HornAtom, HornClause, SymbolicTaskContext, Var


def atom(pred: int, *args) -> HornAtom:
    return HornAtom(pred=pred, args=tuple(args))


def rule(head: HornAtom, *body: HornAtom) -> HornClause:
    return HornClause(head=head, body=tuple(body))


def build_prover(*, creative_enabled: bool) -> DifferentiableProver:
    prover = DifferentiableProver(
        d_latent=16,
        sym_vocab=64,
        max_rules=64,
        max_depth=2,
        n_cands=2,
    )
    prover.eval()
    prover.configure_creative_cycle(
        enabled=creative_enabled,
        cycle_every=1,
        max_selected_rules=2,
        tau_analogy=0.10,
        analogy_contrastive_steps=0,
        aee_generations=1,
        aee_population=6,
    )
    x, y, z = Var("X"), Var("Y"), Var("Z")
    for clause in [
        rule(atom(1, x, y, z), atom(1, y, x, z)),
        rule(atom(101, x, z), atom(1, x, y, z)),
        rule(atom(2, x, y, z), atom(202, x, y, z)),
        rule(atom(102, x, z), atom(2, x, y, z)),
    ]:
        prover.kb.add_rule(clause, status=EpistemicStatus.verified)
    goal = atom(102, Const(1), Const(3))
    prover.set_task_context(
        SymbolicTaskContext(
            observed_facts=frozenset(
                {
                    atom(2, Const(1), Const(2), Const(3)),
                    atom(202, Const(1), Const(2), Const(3)),
                }
            ),
            goal=goal,
            target_facts=frozenset({goal}),
            provenance="benchmark",
            metadata={"gap_norm": 0.9},
        )
    )
    return prover


def main() -> None:
    device = torch.device("cpu")
    z = torch.randn(1, 16, device=device)
    world_error = torch.tensor(0.0, device=device)

    reports = {}
    for label, enabled in (("creative_on", True), ("creative_off", False)):
        prover = build_prover(creative_enabled=enabled).to(device)
        timings_ms = []
        for _ in range(5):
            start = time.perf_counter()
            prover(z, world_error)
            timings_ms.append((time.perf_counter() - start) * 1_000.0)
        reports[label] = {
            "timings_ms": timings_ms,
            "metrics": {
                key: prover.last_forward_info[key]
                for key in sorted(prover.last_forward_info)
                if key.startswith("creative_")
            },
        }

    print("creative-cycle benchmark")
    for label, payload in reports.items():
        timings_ms = payload["timings_ms"]
        print(f"{label}.runs={len(timings_ms)}")
        print(f"{label}.mean_ms={statistics.mean(timings_ms):.3f}")
        print(f"{label}.median_ms={statistics.median(timings_ms):.3f}")
        print(f"{label}.max_ms={max(timings_ms):.3f}")
        for key, value in payload["metrics"].items():
            print(f"{label}.{key}={value}")
    creative_on = reports["creative_on"]["metrics"]
    creative_off = reports["creative_off"]["metrics"]
    for key in sorted(set(creative_on) | set(creative_off)):
        delta = float(creative_on.get(key, 0.0)) - float(creative_off.get(key, 0.0))
        print(f"delta.{key}={delta}")


if __name__ == "__main__":
    main()
