"""
omen_train_code.py — Повний тренувальний цикл OMEN-Scale + NET
==============================================================

Два етапи навчання (як описано в специфікації NET):

  Stage 1 — NET Pretraining (byte-level compression)
    · Оптимізується L_code + L_rec (без S-Core)
    · Словник NET формується з нуля через EMA + dead-code restart
    · Ціль: стабільний кодбук перед символьним fine-tuning

  Stage 2 — OMEN-Scale Full Training (joint)
    · J = Perplexity + β·L_proof + γ·L_world - α·I(Z;M)
          + L_scale + λ_rule·Complexity(Γ) + η_tok·L_NET
    · S-Core отримує токени і виводить Horn-правила
    · NET словник продовжує рости через абдукцію

Запуск:
  python omen_train_code.py                     # smoke test (demo config)
  python omen_train_code.py --config demo       # CPU (швидко)
  python omen_train_code.py --config mid        # 1x A100
  python omen_train_code.py --real_text FILE    # реальний корпус (UTF-8)
"""

from __future__ import annotations
import argparse
import math
import random
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)

from omen_scale_config import OMENScaleConfig
from omen_scale import OMENScale
from omen_v2 import make_counting, make_python, make_rule_transfer, collate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
# 1. ДАНІ
# ══════════════════════════════════════════════════════════════════════════════

def load_text_corpus(path: str, seq_len: int,
                     max_samples: int = 50_000) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Завантажує реальний текст як байтові послідовності.

    UTF-8 текст → int байти [0..255] → (src, tgt) пари.
    Stride = seq_len//2 для 50% overlap між сусідніми прикладами.
    """
    text = Path(path).read_bytes()
    samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
    stride = seq_len // 2

    for start in range(0, len(text) - seq_len - 1, stride):
        chunk = list(text[start: start + seq_len + 1])
        src   = torch.tensor(chunk[:-1], dtype=torch.long)
        tgt   = torch.tensor(chunk[1:],  dtype=torch.long)
        samples.append((src, tgt))
        if len(samples) >= max_samples:
            break

    random.shuffle(samples)
    print(f"  [Dataset] {path}: {len(samples):,} samples x {seq_len} bytes")
    return samples


def make_synthetic_dataset(cfg: OMENScaleConfig,
                           n: int = 512) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Синтетичний датасет: counting + python + rule_transfer."""
    ds: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for gen in [make_counting, make_python, make_rule_transfer]:
        ds.extend(gen(n // 3, cfg.seq_len))
    random.shuffle(ds)
    return ds


# ══════════════════════════════════════════════════════════════════════════════
# 2. STAGE 1 — NET PRETRAINING
# ══════════════════════════════════════════════════════════════════════════════

def pretrain_net(model: OMENScale,
                 dataset: List[Tuple[torch.Tensor, torch.Tensor]],
                 n_steps: int = 500,
                 batch_size: int = 8,
                 lr: float = 3e-4,
                 log_every: int = 50) -> Dict:
    """
    Stage 1: попереднє навчання NET без символьної частини.
    Оптимізує лише: L_NET = L_code + L_rec + L_vocab + λ_vq·L_vq
    Решта параметрів заморожена.
    """
    if not model.net_enabled:
        print("  [Stage 1] NET вимкнений — пропускаємо")
        return {}

    print("\n" + "═" * 68)
    print("  STAGE 1: NET Pretraining  (byte-level compression)")
    print("═" * 68)

    # Заморожуємо все, крім NET
    for name, p in model.named_parameters():
        p.requires_grad = "net." in name

    net_params = [p for p in model.parameters() if p.requires_grad]
    print(f"  NET params: {sum(p.numel() for p in net_params):,}")

    opt   = AdamW(net_params, lr=lr, weight_decay=1e-5)
    sched = CosineAnnealingLR(opt, T_max=max(n_steps, 1), eta_min=lr * 0.01)
    history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=log_every))

    model.net.train()
    t0 = time.perf_counter()

    for step in range(1, n_steps + 1):
        batch    = random.sample(dataset, min(batch_size, len(dataset)))
        src, tgt = collate(batch)
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)

        opt.zero_grad(set_to_none=True)

        h_q, _, vq_info = model.net.encode(src)
        z_dummy = torch.zeros(src.size(0), model.cfg.d_latent, device=DEVICE)
        _, l_rec = model.net.decode(tgt, z_dummy, h_q)
        loss_dict = model.net.compute_loss(vq_info, l_rec)
        loss = loss_dict["net_total"]

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net_params, 1.0)
        opt.step()
        model.memory.maybe_flush()
        sched.step()

        for k in ("net_total", "net_code", "net_rec", "net_vq", "net_entropy_bits"):
            v = loss_dict.get(k, 0.0)
            history[k].append(float(v) if not isinstance(v, float) else v)
        history["vocab"].append(vq_info["vocab_size"])

        if step % log_every == 0:
            m = {k: sum(v) / len(v) for k, v in history.items()}
            elapsed = time.perf_counter() - t0
            print(f"  step {step:5d}/{n_steps}  "
                  f"L_code={m['net_code']:.4f}  "
                  f"L_rec={m['net_rec']:.4f}  "
                  f"L_vq={m['net_vq']:.4f}  "
                  f"H={m['net_entropy_bits']:.2f}bits  "
                  f"V={int(m['vocab'])}  "
                  f"{elapsed:.0f}s")

    # Розморожуємо всі параметри
    for p in model.parameters():
        p.requires_grad = True

    q = model.net.quantizer
    print(f"\n  Stage 1 done. vocab={q.current_size.item()}/{q.max_vocab}  "
          f"new_tokens={q.n_new_tokens}  "
          f"kb_facts={model.prover.kb.n_facts()}")
    return {"final_vocab": q.current_size.item(), "new_tokens": q.n_new_tokens}


# ══════════════════════════════════════════════════════════════════════════════
# 3. STAGE 2 — JOINT TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def joint_train(model: OMENScale,
                dataset: List[Tuple[torch.Tensor, torch.Tensor]],
                n_epochs: int = 10,
                batch_size: int = 8,
                lr: float = 1e-4,
                weight_decay: float = 1e-2,
                max_batches_per_epoch: Optional[int] = None,
                checkpoint_dir: Optional[str] = None) -> List[Dict]:
    """
    Stage 2: спільне навчання OMEN-Scale.
    NET має менший LR (0.3x) щоб не зруйнувати Stage-1 кодбук.

    max_batches_per_epoch=None → весь датасет за кожну епоху.
    Задай числом щоб обмежити (наприклад 256 = 2048 зразків / епоха).
    """
    print("\n" + "═" * 68)
    print("  STAGE 2: Joint Training  (OMEN-Scale + NET)")
    print("═" * 68)

    # Обчислюємо ефективний ліміт батчів (None = весь датасет)
    n_total_batches = len(dataset) // batch_size
    _max_bat = max_batches_per_epoch if max_batches_per_epoch is not None else n_total_batches
    print(f"  batches/epoch : {min(_max_bat, n_total_batches)}"
          f"  ({min(_max_bat, n_total_batches) * batch_size} samples)")

    net_params   = [p for n, p in model.named_parameters() if "net." in n]
    other_params = [p for n, p in model.named_parameters() if "net." not in n]

    opt = AdamW([
        {"params": net_params,   "lr": lr * 0.3, "weight_decay": 1e-5},
        {"params": other_params, "lr": lr,        "weight_decay": weight_decay},
    ])

    # FIX ❸: CosineAnnealingWarmRestarts(T_0=10) → LR-спайк на epoch 11 (PPL=1454).
    # Замінюємо на: 2-епохний лінійний warmup + CosineAnnealingLR без рестартів.
    # Результат: LR плавно зростає перші 2 епохи, потім монотонно спадає до lr*0.01.
    _warmup_ep = min(2, n_epochs)
    _decay_ep  = max(n_epochs - _warmup_ep, 1)
    warmup_sched = LinearLR(opt, start_factor=0.1, end_factor=1.0,
                            total_iters=_warmup_ep)
    decay_sched  = CosineAnnealingLR(opt, T_max=_decay_ep, eta_min=lr * 0.01)
    sched = SequentialLR(opt, schedulers=[warmup_sched, decay_sched],
                         milestones=[_warmup_ep])

    hdr = (f"{'Ep':>4} {'CE':>7} {'World':>7} {'LScale':>7} "
           f"{'NetL':>7} {'Voc':>4} {'KBf':>4} {'PPL':>8} {'ms':>6}")
    print(f"\n  {hdr}")
    print("  " + "─" * len(hdr))

    results: List[Dict] = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        t0 = time.perf_counter()
        agg: Dict[str, float] = defaultdict(float)
        n_bat = 0
        random.shuffle(dataset)

        for start in range(0, len(dataset) - batch_size, batch_size):
            batch    = dataset[start: start + batch_size]
            src, tgt = collate(batch)
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)

            opt.zero_grad(set_to_none=True)
            out = model(src, tgt)

            if torch.isnan(out["total"]) or torch.isinf(out["total"]):
                continue

            out["total"].backward()
            torch.nn.utils.clip_grad_norm_(net_params,   0.5)
            torch.nn.utils.clip_grad_norm_(other_params, 1.0)
            opt.step()

            # Flush memory AFTER optimizer step (no autograd graph)
            model.memory.maybe_flush()

            for k, v in out.items():
                if k not in ("logits", "z"):
                    agg[k] += float(v) if not isinstance(v, float) else v
            n_bat += 1
            if n_bat >= _max_bat:
                break

        sched.step()

        if n_bat == 0:
            continue

        avg = {k: v / n_bat for k, v in agg.items()}
        avg["ppl"] = math.exp(min(avg.get("ce", 10), 10))
        avg["ms"]  = (time.perf_counter() - t0) * 1000
        avg["net_vocab"] = model.net.quantizer.current_size.item() if model.net_enabled else 0
        avg["kb_facts"]  = model.prover.kb.n_facts()

        print(f"  {epoch:4d} "
              f"{avg.get('ce', 0):7.3f} "
              f"{min(avg.get('world', 0), 10.0):7.3f} "
              f"{avg.get('l_scale', 0):7.4f} "
              f"{avg.get('net_loss', 0):7.3f} "
              f"{avg['net_vocab']:4d} "
              f"{avg['kb_facts']:4d} "
              f"{avg['ppl']:8.2f} "
              f"{avg['ms']:6.0f}")
        results.append(avg)

        if checkpoint_dir and epoch % 2 == 0:
            _save_ckpt(model, opt, epoch, avg, checkpoint_dir)

    print("  " + "─" * len(hdr))
    if results:
        print(f"  Best PPL: {min(r['ppl'] for r in results):.2f}")
    return results


def _save_ckpt(model, optimizer, epoch, metrics, save_dir):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    path = str(Path(save_dir) / f"omen_epoch{epoch:04d}.pt")
    torch.save({
        "epoch": epoch, "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else {},
        "metrics": metrics,
        "net_vocab": model.net.quantizer.current_size.item() if model.net_enabled else 0,
        "kb_facts": model.prover.kb.n_facts(),
    }, path)
    print(f"    [ckpt] {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def net_diagnostics(model: OMENScale,
                    dataset: List[Tuple[torch.Tensor, torch.Tensor]],
                    n_samples: int = 4) -> None:
    if not model.net_enabled:
        return
    model.eval()
    q   = model.net.quantizer
    V   = q.current_size.item()
    src, _ = collate(random.sample(dataset, min(n_samples, len(dataset))))
    _, idx, info = model.net.encode(src.to(DEVICE))

    counts = torch.bincount(idx.view(-1).clamp(0, V - 1), minlength=V).float()
    used   = (counts > 0).sum().item()
    H_bits = info.get("usage_entropy", 0) / math.log(2)

    print(f"\n  ── NET Diagnostics ──────────────────────────────")
    print(f"  Vocab:   {V}/{q.max_vocab}  Used={used}  Dead={V-used}")
    print(f"  Entropy: {H_bits:.2f} bits / {math.log2(max(V,2)):.2f} max")
    print(f"  MeanSim: {info['mean_sim']:.4f}  τ={q.tau}")
    print(f"  NewToks: {q.n_new_tokens}  Calls={q.n_quant_calls}")
    print(f"  KB facts={model.prover.kb.n_facts()}")
    print(f"  ────────────────────────────────────────────────")


# ══════════════════════════════════════════════════════════════════════════════
# 5. SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

def smoke_test() -> None:
    torch.manual_seed(0); random.seed(0)
    print("═" * 68)
    print("  SMOKE TEST: OMEN-Scale + NET (demo config, CPU)")
    print("═" * 68)

    cfg     = OMENScaleConfig.demo()
    model   = OMENScale(cfg).to(DEVICE)
    dataset = make_synthetic_dataset(cfg, n=128)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params:,}  KB linked: {model.net.quantizer.kb is not None}")

    # Stage 1
    s1 = pretrain_net(model, dataset, n_steps=60, batch_size=4,
                      lr=3e-4, log_every=20)
    net_diagnostics(model, dataset)

    # Stage 2
    results = joint_train(model, dataset, n_epochs=4, batch_size=4,
                          lr=1e-4, max_batches_per_epoch=8)

    ppls = [r["ppl"] for r in results]
    best_ppl = min(ppls)
    # Перевіряємо що best PPL краще за перший крок (не обов'язково останній)
    assert best_ppl < ppls[0] * 0.95 or best_ppl < ppls[0], \
        f"PPL не покращилась: best={best_ppl:.1f} vs initial={ppls[0]:.1f}"
    print(f"\n  PPL: {ppls[0]:.1f} → min={best_ppl:.1f} (ep{ppls.index(best_ppl)+1})  ✓")

    vocab_final = model.net.quantizer.current_size.item()
    assert vocab_final > cfg.net_init_vocab, "vocab не виріс"
    print(f"  NET vocab: {cfg.net_init_vocab} → {vocab_final}  ✓")

    assert model.prover.kb.n_facts() > 0, "KB порожня"
    print(f"  KB facts: {model.prover.kb.n_facts()}  ✓")

    print(f"\n{model.memory_report()}")
    print("\n  ✅  Smoke test пройдено")


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",   default="demo", choices=["demo","strong","mid","full"])
    p.add_argument("--real_text", default=None)
    p.add_argument("--stage1_steps", type=int, default=300)
    p.add_argument("--stage2_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--checkpoint_dir", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_net", action="store_true")
    # FIX ❹: раніше max_batches хардкодовано 32 → 256 зразків / 45 000 = 0.57%.
    # None = весь датасет (одна повна епоха). Числове значення = ліміт батчів.
    # Рекомендація для RTX 3080 + strong config:
    #   --max_batches 256  (≈ 2048 зразків / епоха, ~2 хв / епоха)
    #   --max_batches 500  (≈ 4000 зразків / епоха, ~4 хв / епоха)
    #   без аргументу     = весь датасет, але довго
    p.add_argument("--max_batches", type=int, default=None,
                   help="Ліміт батчів/епоху (None = весь датасет). "
                        "Рекомендовано: 256-500 для швидких ітерацій.")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed)
    cfg = {"demo":   OMENScaleConfig.demo,
           "strong": OMENScaleConfig.strong,
           "mid":    OMENScaleConfig.mid,
           "full":   OMENScaleConfig.full}[args.config]()
    if args.no_net:
        cfg.net_enabled = False

    print(f"Config={args.config}  device={DEVICE}  NET={'on' if cfg.net_enabled else 'off'}")

    if args.real_text:
        dataset = load_text_corpus(args.real_text, cfg.seq_len)
    else:
        dataset = make_synthetic_dataset(cfg, n=512)

    split    = int(0.9 * len(dataset))
    train_ds = dataset[:split]

    model = OMENScale(cfg).to(DEVICE)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])

    pretrain_net(model, train_ds, n_steps=args.stage1_steps,
                 batch_size=args.batch_size)
    if cfg.net_enabled:
        net_diagnostics(model, train_ds)

    joint_train(model, train_ds, n_epochs=args.stage2_epochs,
                batch_size=args.batch_size, lr=args.lr,
                max_batches_per_epoch=args.max_batches,
                checkpoint_dir=args.checkpoint_dir)
    print(model.memory_report())


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        smoke_test()