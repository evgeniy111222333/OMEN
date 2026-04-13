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

  # Рекомендований запуск на GPU (RTX 3080/4080):
  python omen_train_code.py \\
      --real_text codesearchnet_python.txt \\
      --config strong --stage1_steps 1000 \\
      --stage2_epochs 20 --batch_size 8 \\
      --amp --grad_accum 2 \\
      --checkpoint_dir ./checkpoints

  Оптимізації Stage 2 (без втрати якості):
    --amp            : FP16 autocast на GPU  → ~2x прискорення матричних операцій
    --grad_accum N   : накопичення градієнтів → менше opt.step (рекомендовано: 2–4)
    --num_workers N  : DataLoader workers для паралельного prefetch даних (2–4 на GPU)
    --max_batches N  : ліміт батчів/епоху для коротших ітерацій
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
from torch.utils.data import DataLoader, Dataset

import contextlib
import pickle

from omen_scale_config import OMENScaleConfig
from omen_scale import OMENScale
from omen_v2 import make_counting, make_python, make_rule_transfer, collate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# AMP доступний тільки на CUDA (на CPU — no-op через enabled=False)
_AMP_AVAILABLE = torch.cuda.is_available()


# ══════════════════════════════════════════════════════════════════════════════
# 0. STACKED DATASET  (zero-copy batching)
# ══════════════════════════════════════════════════════════════════════════════

class StackedDataset(Dataset):
    """
    Перетворює List[Tuple[Tensor,Tensor]] у два великих тензори (N,T).
    Це виконується ONE TIME, після чого кожен батч — просто slice (zero-copy).

    Порівняно з поточним підходом (torch.stack кожен батч):
      · Старий: N_batches × torch.stack(8 tensors) = N_batches × ~0.1ms = накопичується
      · Новий:  stack один раз при ініціалізації, далі __getitem__ = O(1) indirection

    На GPU з pin_memory=True DataLoader асинхронно переносить дані CPU→GPU
    паралельно з тим, як GPU обраховує попередній батч (реальний overlap).
    """
    def __init__(self, data: List[Tuple[torch.Tensor, torch.Tensor]]):
        t0 = time.perf_counter()
        # Підтримуємо два формати (як collate):
        #   List[Tuple[src, tgt]] — реальний текст (load_text_corpus)
        #   List[Tensor]          — синтетичний (make_counting, make_python, …)
        if isinstance(data[0], (tuple, list)):
            self.src = torch.stack([x[0] for x in data])   # (N, T)
            self.tgt = torch.stack([x[1] for x in data])   # (N, T)
        else:
            stacked  = torch.stack(data)                    # (N, T+1)
            self.src = stacked[:, :-1]                      # (N, T)
            self.tgt = stacked[:, 1:]                       # (N, T)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  [StackedDataset] {len(self.src):,} зразків x T={self.src.shape[1]} "
              f"стеккінг за {elapsed:.0f}ms  "
              f"({self.src.numel()*2*4/1e6:.1f} MB RAM)")

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.src[i], self.tgt[i]


def build_loader(dataset_obj: StackedDataset,
                 batch_size: int,
                 num_workers: int = 0,
                 shuffle: bool = True) -> DataLoader:
    """
    Будує DataLoader з pin_memory (якщо CUDA) і prefetch_factor.
    num_workers=0 — безпечно з будь-яким GPU/CPU, >0 лише якщо CUDA і стабільно.
    """
    return DataLoader(
        dataset_obj,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=(DEVICE.type == "cuda"),
        num_workers=num_workers,
        prefetch_factor=(2 if num_workers > 0 else None),
        drop_last=True,
    )


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

        # Non-autoregressive reconstruction loss (позиційна, без causal bypass).
        # Без цього decoder навчається передбачати tgt[i+1] з tgt[0..i]
        # БЕЗ використання h_q → gradient через h_q до encoder слабшає → collapse.
        # stage1_rec_loss форсує h_q[i] кодувати саме src[i].
        l_nonauto = model.net.stage1_rec_loss(h_q, src)

        loss_dict = model.net.compute_loss(vq_info, l_rec)
        loss = loss_dict["net_total"] + 0.5 * l_nonauto

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
                  f"{elapsed:.0f}s", flush=True)

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
                checkpoint_dir: Optional[str] = None,
                use_amp: bool = False,
                grad_accum: int = 1,
                num_workers: int = 0,
                opt_state: Optional[dict] = None) -> List[Dict]:
    """
    Stage 2: спільне навчання OMEN-Scale.
    NET має менший LR (0.3x) щоб не зруйнувати Stage-1 кодбук.

    Оптимізації (без втрати якості):
      · StackedDataset + DataLoader → zero-copy batching + pin_memory async transfer
      · AMP (autocast + GradScaler) → FP16 matmul на GPU (40–100% прискорення)
      · grad_accum > 1 → менше optimizer.step() викликів (effective_batch = batch×accum)
      · Throughput звіт: samples/sec, tokens/sec, ms/batch

    max_batches_per_epoch=None → весь датасет за кожну епоху.
    Задай числом щоб обмежити (наприклад 256 = 2048 зразків / епоха).
    """
    use_amp = use_amp and _AMP_AVAILABLE  # AMP тільки на CUDA
    _device_type = "cuda" if DEVICE.type == "cuda" else "cpu"

    # torch.autocast — уніфікований API (PyTorch ≥ 2.0).
    # enabled=False → nullcontext-like no-op, безпечно на CPU/GPU.
    def _amp_ctx():
        return torch.autocast(device_type=_device_type,
                              dtype=torch.float16 if use_amp else torch.float32,
                              enabled=use_amp)

    print("\n" + "═" * 68)
    print("  STAGE 2: Joint Training  (OMEN-Scale + NET)")
    print("═" * 68)

    # ── StackedDataset + DataLoader ───────────────────────────────────────────
    stacked = StackedDataset(dataset)
    loader  = build_loader(stacked, batch_size, num_workers=num_workers, shuffle=True)

    # Обчислюємо ефективний ліміт батчів
    n_total_batches = len(loader)
    _max_bat = max_batches_per_epoch if max_batches_per_epoch is not None else n_total_batches
    _max_bat = min(_max_bat, n_total_batches)

    effective_batch = batch_size * grad_accum
    print(f"  batches/epoch : {_max_bat}  ({_max_bat * batch_size} samples)")
    print(f"  effective_batch: {effective_batch}  (batch_size={batch_size} × grad_accum={grad_accum})")
    print(f"  AMP: {'✓ FP16' if use_amp else '✗ FP32'}  "
          f"pin_memory: {'✓' if DEVICE.type=='cuda' else '✗'}  "
          f"workers: {num_workers}")

    net_params   = [p for n, p in model.named_parameters() if "net." in n]
    other_params = [p for n, p in model.named_parameters() if "net." not in n]

    opt = AdamW([
        {"params": net_params,   "lr": lr * 0.3, "weight_decay": 1e-5},
        {"params": other_params, "lr": lr,        "weight_decay": weight_decay},
    ])
    # Bug fix: відновлюємо optimizer state (Adam m1/m2) після resume
    if opt_state:
        try:
            opt.load_state_dict(opt_state)
            print("  [resume] Optimizer state відновлено ✓")
        except Exception as e:
            print(f"  [resume] Optimizer state не вдалось відновити: {e} — продовжуємо з нуля")

    # FIX ❸: CosineAnnealingWarmRestarts(T_0=10) → LR-спайк на epoch 11 (PPL=1454).
    _warmup_ep = min(2, n_epochs)
    _decay_ep  = max(n_epochs - _warmup_ep, 1)
    warmup_sched = LinearLR(opt, start_factor=0.1, end_factor=1.0,
                            total_iters=_warmup_ep)
    decay_sched  = CosineAnnealingLR(opt, T_max=_decay_ep, eta_min=lr * 0.01)
    sched = SequentialLR(opt, schedulers=[warmup_sched, decay_sched],
                         milestones=[_warmup_ep])

    # AMP GradScaler (no-op якщо use_amp=False)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    hdr = (f"{'Ep':>4} {'CE':>7} {'World':>7} {'LScale':>7} "
           f"{'NetL':>7} {'Voc':>4} {'KBf':>4} {'PPL':>8} {'ms/b':>6} {'tok/s':>7}")
    print(f"\n  {hdr}")
    print("  " + "─" * len(hdr))

    results: List[Dict] = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        t0_epoch = time.perf_counter()
        agg: Dict[str, float] = defaultdict(float)
        n_bat = 0
        total_tokens = 0

        # timing buckets для throughput аналізу
        t_data, t_fwd, t_bwd, t_opt = 0.0, 0.0, 0.0, 0.0

        opt.zero_grad(set_to_none=True)

        t_iter_start = time.perf_counter()
        for src, tgt in loader:
            t_data += time.perf_counter() - t_iter_start  # data prefetch time

            src = src.to(DEVICE, non_blocking=True)
            tgt = tgt.to(DEVICE, non_blocking=True)

            # ── Forward (з AMP якщо use_amp) ─────────────────────────────────
            t1 = time.perf_counter()
            with _amp_ctx():
                out = model(src, tgt)

            # ── NaN/Inf guard з покомпонентною діагностикою ───────────────────
            if torch.isnan(out["total"]) or torch.isinf(out["total"]):
                # Знаходимо проблемний компонент
                bad = [k for k, v in out.items()
                       if k not in ("logits", "z", "emc_stop")
                       and isinstance(v, float) and (math.isnan(v) or math.isinf(v))]
                print(f"\n  ⚠️  NaN/Inf у total (batch {n_bat+1}), пропускаємо. "
                      f"Проблемні ключі: {bad if bad else 'невідомо'}")
                t_iter_start = time.perf_counter()
                continue
            t_fwd += time.perf_counter() - t1

            # ── Backward (масштабований для AMP) ─────────────────────────────
            t2 = time.perf_counter()
            loss = out["total"] / grad_accum
            scaler.scale(loss).backward()
            t_bwd += time.perf_counter() - t2

            n_bat += 1
            total_tokens += src.numel()

            for k, v in out.items():
                if k not in ("logits", "z", "emc_stop"):
                    try:
                        fv = float(v) if not isinstance(v, float) else v
                        if not (math.isnan(fv) or math.isinf(fv)):
                            agg[k] += fv
                    except (TypeError, ValueError):
                        pass

            # ── Optimizer step кожні grad_accum кроків ───────────────────────
            if n_bat % grad_accum == 0:
                t3 = time.perf_counter()
                scaler.unscale_(opt)
                # Вимірюємо grad norm ДО кліпу для діагностики
                gnorm_net   = torch.nn.utils.clip_grad_norm_(net_params,   0.5)
                gnorm_other = torch.nn.utils.clip_grad_norm_(other_params, 1.0)
                agg["gnorm_net"]   += float(gnorm_net)
                agg["gnorm_other"] += float(gnorm_other)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                model.memory.maybe_flush()
                t_opt += time.perf_counter() - t3

            # ── Прогрес-лічильник ─────────────────────────────────────────────
            elapsed_bat = time.perf_counter() - t0_epoch
            ms_per_bat  = elapsed_bat / n_bat * 1000
            eta_s       = ms_per_bat * (_max_bat - n_bat) / 1000
            ce_cur      = agg.get("ce", 0.0) / n_bat
            print(f"  \r  Ep {epoch:2d}  [{n_bat:4d}/{_max_bat}]  "
                  f"{n_bat / _max_bat * 100:5.1f}%  "
                  f"CE={ce_cur:.3f}  "
                  f"{ms_per_bat:5.0f}ms/b  "
                  f"ETA {eta_s:5.0f}s",
                  end="", flush=True)

            if n_bat >= _max_bat:
                break

            t_iter_start = time.perf_counter()

        # Final flush якщо залишились акумульовані градієнти
        if n_bat % grad_accum != 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net_params,   0.5)
            torch.nn.utils.clip_grad_norm_(other_params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            model.memory.maybe_flush()

        sched.step()

        if n_bat == 0:
            continue

        epoch_time = time.perf_counter() - t0_epoch
        n_opt_steps = max(n_bat // grad_accum, 1)
        avg = {k: v / n_bat for k, v in agg.items()}
        # grad norms усереднені по кількості optimizer steps
        avg["gnorm_net"]   = agg.get("gnorm_net",   0.0) / n_opt_steps
        avg["gnorm_other"] = agg.get("gnorm_other", 0.0) / n_opt_steps
        avg["ppl"]       = math.exp(min(avg.get("ce", 10), 10))
        avg["ms"]        = (epoch_time / n_bat) * 1000
        avg["tok_s"]     = total_tokens / epoch_time
        avg["net_vocab"] = model.net.quantizer.current_size.item() if model.net_enabled else 0
        avg["kb_facts"]  = model.prover.kb.n_facts()
        avg["lr"]        = opt.param_groups[-1]["lr"]  # LR основної групи

        # Throughput breakdown (тільки epoch 1)
        if epoch == 1:
            print(f"\n  ── Epoch 1 throughput breakdown ──")
            print(f"     data prefetch : {t_data*1000/n_bat:5.1f} ms/batch")
            print(f"     forward       : {t_fwd*1000/n_bat:5.1f} ms/batch")
            print(f"     backward      : {t_bwd*1000/n_bat:5.1f} ms/batch")
            print(f"     opt+step      : {t_opt*1000/n_bat:5.1f} ms/batch  "
                  f"(кожні {grad_accum} batches)")
            print(f"     total/batch   : {avg['ms']:5.1f} ms/batch")
            print(f"     throughput    : {avg['tok_s']:,.0f} tok/s\n")
            print(f"  {hdr}")
            print("  " + "─" * len(hdr))

        # ── PPL spike detector ────────────────────────────────────────────────
        _prev_ppl = results[-1]["ppl"] if results else avg["ppl"]
        _spike = avg["ppl"] > _prev_ppl * 3.0 and avg["ppl"] > 20.0
        if _spike:
            print(f"\n  🔴 PPL SPIKE: {_prev_ppl:.1f} → {avg['ppl']:.1f}  Компоненти лосу:")
            diag_keys = ["ce", "world", "l_scale", "sym_ground", "ltm_pen",
                         "ltm_pen_raw", "net_loss", "meta_loss", "traj_reward",
                         "curiosity", "recall", "novelty", "vem_pen",
                         "gnorm_net", "gnorm_other", "lr",
                         # OSF: компоненти J_OSF для діагностики стрибків
                         "osf_l_plan", "osf_l_sim", "osf_l_refl", "osf_l_meta",
                         "osf_l_intent", "osf_struct", "osf_plan_rl"]
            for dk in diag_keys:
                v = avg.get(dk, 0.0)
                flag = " ⚠️" if abs(v) > 5.0 else ""
                print(f"     {dk:<18}: {v:+10.5f}{flag}")

        # ── Компактний рядок епохи ─────────────────────────────────────────────
        print(f"\r  {epoch:4d} "
              f"{avg.get('ce', 0):7.3f} "
              f"{min(avg.get('world', 0), 10.0):7.3f} "
              f"{avg.get('l_scale', 0):7.4f} "
              f"{avg.get('net_loss', 0):7.3f} "
              f"{avg['net_vocab']:4d} "
              f"{avg['kb_facts']:4d} "
              f"{avg['ppl']:8.2f} "
              f"{avg['ms']:6.0f} "
              f"{avg['tok_s']:7.0f}          ", flush=True)

        # ── Детальний рядок loss-компонентів ─────────────────────────────────
        print(f"       "
              f"sym={avg.get('sym_ground',0):.3f} "
              f"ltm={avg.get('ltm_pen',0):.3f}({avg.get('ltm_pen_raw',0):.1f}) "
              f"meta={avg.get('meta_loss',0):.3f} "
              f"traj_r={avg.get('traj_reward',0):.3f} "
              f"vem={avg.get('vem_pen',0):.4f} "
              f"cur={avg.get('curiosity',0):.3f} "
              f"|∇net|={avg.get('gnorm_net',0):.3f} "
              f"|∇oth|={avg.get('gnorm_other',0):.3f} "
              f"lr={avg.get('lr',0):.2e}")

        # ── EMC рядок (якщо ввімкнено) ────────────────────────────────────────
        if getattr(model.cfg, 'emc_enabled', False):
            emc_steps  = avg.get('emc_steps', 0.0)
            meta_loss  = avg.get('meta_loss', 0.0)
            traj_r     = avg.get('emc_traj_r', 0.0)
            pct_proved = avg.get('emc_proved', 0.0) * 100
            a_stop     = avg.get('emc_a_stop', 0.0) * 100
            a_recall   = avg.get('emc_a_recall', 0.0) * 100
            a_fc       = avg.get('emc_a_fc', 0.0) * 100
            a_abd      = avg.get('emc_a_abduce', 0.0) * 100
            print(f"       [EMC] steps={emc_steps:.1f} "
                  f"proved={pct_proved:.0f}% "
                  f"stop={a_stop:.0f}% recall={a_recall:.0f}% "
                  f"fc={a_fc:.0f}% abd={a_abd:.0f}% "
                  f"mdl={avg.get('emc_mdl',0):.2f}")

        # ── OSF рядок (якщо ввімкнено) ────────────────────────────────────────
        # Відображає всі компоненти J_OSF + стан мета-контролера стратегій.
        # σ: 0=Fast, 1=Careful, 2=Exploratory — середнє за епоху.
        if getattr(model.cfg, 'osf_enabled', False):
            _osf_plan   = avg.get('osf_l_plan',    0.0)
            _osf_sim    = avg.get('osf_l_sim',     0.0)
            _osf_refl   = avg.get('osf_l_refl',    0.0)
            _osf_meta   = avg.get('osf_l_meta',    0.0)
            _osf_intent = avg.get('osf_l_intent',  0.0)
            _osf_struct = avg.get('osf_struct',    0.0)
            _osf_rl     = avg.get('osf_plan_rl',   0.0)
            _osf_depth  = avg.get('osf_plan_depth',0.0)
            _osf_entr   = avg.get('osf_goal_entropy', 0.0)
            _osf_sigma  = avg.get('osf_strategy',  1.0)
            _osf_rce    = getattr(model, '_osf_running_ce', 5.0)
            _freq_fast  = avg.get('meta_freq_Fast',        0.0) * 100
            _freq_care  = avg.get('meta_freq_Careful',     0.0) * 100
            _freq_expl  = avg.get('meta_freq_Exploratory', 0.0) * 100
            print(f"       [OSF] plan={_osf_plan:.3f} sim={_osf_sim:.3f} "
                  f"refl={_osf_refl:.3f} meta={_osf_meta:.3f} "
                  f"intent={_osf_intent:.3f} struct={_osf_struct:.3f} "
                  f"rl={_osf_rl:.3f} | "
                  f"depth={_osf_depth:.1f} H={_osf_entr:.2f} "
                  f"σ={_osf_sigma:.1f} CE_ema={_osf_rce:.3f} | "
                  f"Fast={_freq_fast:.0f}% Care={_freq_care:.0f}% Expl={_freq_expl:.0f}%")

        results.append(avg)

        if checkpoint_dir and (epoch % 2 == 0 or epoch == n_epochs):
            _save_ckpt(model, opt, epoch, avg, checkpoint_dir)

    print("  " + "─" * len(hdr))
    if results:
        print(f"  Best PPL: {min(r['ppl'] for r in results):.2f}")
    return results


def _serialize_kb(kb) -> dict:
    """Повна серіалізація TensorKnowledgeBase: факти, правила, records, лічильники."""
    return {
        "fact_buf":   kb._fact_buf[:kb._n_facts].cpu().clone(),   # (n_facts, 3) int64
        "rules":      pickle.dumps(list(kb.rules)),               # bytes → safe for weights_only=True
        "n_facts":    kb._n_facts,
        "n_rules":    kb._n_rules,
        "n_proposed": kb._n_proposed,
        "n_verified": kb._n_verified,
        "records":    pickle.dumps(dict(kb._records)),             # bytes → safe (RuleRecord містить HornClause)
        "fact_set":   set(kb._fact_set),
        "rule_hash_set": set(kb._rule_hash_set),
    }


def _restore_kb(model, kb_state: dict, device) -> None:
    """Відновлює TensorKnowledgeBase з серіалізованого стану."""
    kb = model.prover.kb
    # Відновлюємо факти
    n_f = kb_state["n_facts"]
    if n_f > 0:
        kb._fact_buf[:n_f] = kb_state["fact_buf"].to(device)
    kb._n_facts       = n_f
    kb._fact_set      = kb_state.get("fact_set", set())
    # Відновлюємо правила
    kb.rules          = pickle.loads(kb_state["rules"]) if isinstance(kb_state["rules"], bytes) else kb_state["rules"]
    kb._n_rules       = kb_state["n_rules"]
    kb._n_proposed    = kb_state["n_proposed"]
    kb._n_verified    = kb_state["n_verified"]
    kb._records       = (pickle.loads(kb_state["records"])
                         if isinstance(kb_state["records"], bytes)
                         else kb_state.get("records", {}))
    kb._rule_hash_set = kb_state.get("rule_hash_set", {hash(r) for r in kb.rules})
    # Інвалідуємо кеш фактів
    kb._facts_cache   = None
    kb._facts_cache_n = -1
    print(f"    [restore] KB: {n_f} facts, {kb_state['n_rules']} rules")


def _save_ckpt(model, optimizer, epoch, metrics, save_dir):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    path = str(Path(save_dir) / f"omen_epoch{epoch:04d}.pt")

    # ── KB: повна серіалізація (Bug 1 fix) ─────────────────────────────────────
    kb_state = _serialize_kb(model.prover.kb)

    # ── Memory cache: episodic recall (Bug 3 fix) ───────────────────────────────
    mem_cache = [(s.cpu().clone(), v.cpu().clone())
                 for s, v in model.memory.cache]

    # ── EMC: мета-статистика (Actor/Critic/StoppingUtility weights — у model.state_dict())
    # EfficientMetaController є nn.Module зареєстрованим як self.emc → його ваги
    # (Actor, Critic, StoppingUtility.task_estimator, EMCStateEncoder) АВТОМАТИЧНО
    # зберігаються через model.state_dict(). Зберігаємо також мета-статистику для
    # діагностики та перевірки коректності відновлення.
    emc_meta = {}
    if getattr(model, 'emc_enabled', False) and hasattr(model, 'emc'):
        emc = model.emc
        emc_meta = {
            "max_steps":    emc.max_steps,
            "gamma":        emc.gamma,
            "lambda_time":  emc.lambda_time,
            "lambda_gap":   emc.lambda_gap,
            "lambda_mdl":   emc.lambda_mdl,
            "eta_int":      emc.eta_int,
            "entropy_beta": emc.entropy_beta,
            "use_gae":      emc.use_gae,
            # Підтверджуємо що ваги зберігаються через model.state_dict()
            "actor_param_count":  sum(p.numel() for p in emc.actor.parameters()),
            "critic_param_count": sum(p.numel() for p in emc.critic.parameters()),
            "stop_param_count":   sum(p.numel() for p in emc.stopping_utility.parameters()),
        }

    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else {},
        "metrics":   metrics,
        "net_vocab": model.net.quantizer.current_size.item() if model.net_enabled else 0,
        "net_tau":   float(model.net.quantizer.tau) if model.net_enabled else None,
        "kb_facts":  kb_state["n_facts"],    # для швидкого перегляду
        "kb_state":  kb_state,               # повний стан KB
        "mem_cache": mem_cache,              # episodic recall cache
        "emc_meta":  emc_meta,              # EMC гіперпараметри + param counts (weights у model)
        # OSF: зберігаємо running CE estimate, щоб мета-контролер відновлював
        # реалістичну оцінку якості замість хардкоду 5.0 після завантаження.
        "osf_running_ce": getattr(model, '_osf_running_ce', 5.0),
    }, path)
    emc_info = ""
    if emc_meta:
        emc_info = (f"  EMC actor={emc_meta.get('actor_param_count',0)}p"
                    f" critic={emc_meta.get('critic_param_count',0)}p"
                    f" stoputil={emc_meta.get('stop_param_count',0)}p")
    print(f"    [ckpt] {path}  (KB={kb_state['n_facts']}f/{kb_state['n_rules']}r"
          f"  cache={len(mem_cache)}){emc_info}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def net_diagnostics(model: OMENScale,
                    dataset: List[Tuple[torch.Tensor, torch.Tensor]],
                    n_samples: int = 32) -> None:
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
                          lr=1e-4, max_batches_per_epoch=8,
                          use_amp=False, grad_accum=1)

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
    # ── Оптимізація швидкості ──────────────────────────────────────────────
    p.add_argument("--amp", action="store_true", default=False,
                   help="Увімкнути AMP (FP16 autocast). "
                        "Дає 40-100%% прискорення на CUDA GPU. "
                        "Автоматично вимикається якщо немає CUDA.")
    p.add_argument("--grad_accum", type=int, default=1,
                   help="Gradient accumulation steps. "
                        "effective_batch = batch_size × grad_accum. "
                        "Рекомендовано: 2-4 (зменшує к-сть opt.step() викликів).")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers (0 = main thread). "
                        "На CUDA можна спробувати 2-4 для prefetch.")
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

    # Підказка: GPU-оптимізація
    if DEVICE.type == "cuda" and not args.amp:
        print("\n  💡 ПІДКАЗКА: ви на GPU але --amp не задано.")
        print("     Рекомендовано: --amp --grad_accum 2")
        print("     Очікуване прискорення: ~2x forward/backward (FP16 tensor cores)\n")

    if args.real_text:
        dataset = load_text_corpus(args.real_text, cfg.seq_len)
    else:
        dataset = make_synthetic_dataset(cfg, n=512)

    split    = int(0.9 * len(dataset))
    train_ds = dataset[:split]

    model = OMENScale(cfg).to(DEVICE)
    _resume_opt_state = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        # Bug 4 fix: відновлюємо KB та memory cache
        if "kb_state" in ckpt:
            _restore_kb(model, ckpt["kb_state"], DEVICE)
        if "mem_cache" in ckpt:
            model.memory.cache.extend(
                [(s.to(DEVICE), v.to(DEVICE)) for s, v in ckpt["mem_cache"]]
            )
        # Bug fix: відновлюємо net_tau (Python float — не входить у state_dict)
        if "net_tau" in ckpt and ckpt["net_tau"] is not None and model.net_enabled:
            model.net.quantizer.tau = float(ckpt["net_tau"])
        # Bug fix: відновлюємо optimizer state для відновлення у joint_train
        _resume_opt_state = ckpt.get("optimizer") or None
        # OSF: відновлюємо running CE estimate (не у state_dict)
        if "osf_running_ce" in ckpt and getattr(model, 'osf_enabled', False):
            model._osf_running_ce = float(ckpt["osf_running_ce"])

        # ── Перевірка EMC state відновлення ──────────────────────────────────
        # EfficientMetaController (Actor + Critic + StoppingUtility + StateEncoder)
        # зберігається через model.state_dict() → відновлюється через load_state_dict().
        # Тут перевіряємо що EMC param counts збігаються з очікуваними (meta-info).
        emc_meta_ckpt = ckpt.get("emc_meta", {})
        if emc_meta_ckpt and getattr(model, 'emc_enabled', False) and hasattr(model, 'emc'):
            emc = model.emc
            ok_actor  = sum(p.numel() for p in emc.actor.parameters()) == emc_meta_ckpt.get("actor_param_count", -1)
            ok_critic = sum(p.numel() for p in emc.critic.parameters()) == emc_meta_ckpt.get("critic_param_count", -1)
            ok_stop   = sum(p.numel() for p in emc.stopping_utility.parameters()) == emc_meta_ckpt.get("stop_param_count", -1)
            status = "✓" if (ok_actor and ok_critic and ok_stop) else "⚠ mismatch"
            print(f"  [resume] EMC weights restored {status} "
                  f"(actor={ok_actor} critic={ok_critic} stoputil={ok_stop})")

        print(f"  [resume] Завантажено epoch={ckpt.get('epoch', '?')}  "
              f"KB={ckpt.get('kb_facts', '?')} facts  "
              f"cache={len(ckpt.get('mem_cache', []))}  "
              f"tau={ckpt.get('net_tau', '?')}")

    # Stage 1: пропускаємо якщо resume (NET відновлено з checkpoint)
    if args.resume:
        print("  [resume] Stage 1 пропущено — NET відновлено з checkpoint")
    else:
        pretrain_net(model, train_ds, n_steps=args.stage1_steps,
                     batch_size=args.batch_size)
    if cfg.net_enabled:
        net_diagnostics(model, train_ds)

    joint_train(model, train_ds, n_epochs=args.stage2_epochs,
                batch_size=args.batch_size, lr=args.lr,
                max_batches_per_epoch=args.max_batches,
                checkpoint_dir=args.checkpoint_dir,
                use_amp=args.amp,
                grad_accum=args.grad_accum,
                num_workers=args.num_workers,
                opt_state=_resume_opt_state)
    print(model.memory_report())


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        smoke_test()