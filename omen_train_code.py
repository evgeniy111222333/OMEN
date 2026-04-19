"""
omen_train_code.py — Full OMEN-Scale + NET training loop
========================================================

Two training stages (as described in the NET spec):

  Stage 1 — NET Pretraining (byte-level compression)
    · Optimize L_code + L_rec (without S-Core)
    · Build the NET vocabulary from scratch via EMA + dead-code restart
    · Goal: a stable codebook before symbolic fine-tuning

  Stage 2 — OMEN-Scale Full Training (joint)
    · J = Perplexity + beta * L_proof + gamma * L_world - alpha * I(Z;M)
          + L_scale + lambda_rule * Complexity(Gamma) + eta_tok * L_NET
    · S-Core consumes tokens and derives Horn rules
    · The NET vocabulary keeps growing through abduction

Usage:
  python omen_train_code.py                     # smoke test (demo config)
  python omen_train_code.py --config demo       # CPU (fast)
  python omen_train_code.py --config mid        # 1x A100
  python omen_train_code.py --real_text FILE    # real corpus (UTF-8)

  # Recommended GPU run (RTX 3080/4080):
  python omen_train_code.py \\
      --real_text codesearchnet_python.txt \\
      --config strong --stage1_steps 1000 \\
      --stage2_epochs 20 --batch_size 8 \\
      --amp --grad_accum 2 \\
      --checkpoint_dir ./checkpoints

  Stage 2 optimizations (without quality loss):
    --amp            : FP16 autocast on GPU  -> ~2x faster matrix ops
    --grad_accum N   : gradient accumulation -> fewer opt.step calls (recommended: 2-4)
    --num_workers N  : DataLoader workers for parallel data prefetch (2-4 on GPU)
    --max_batches N  : batch-per-epoch cap for shorter iterations
"""

from __future__ import annotations
import argparse
import math
import random
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader, Dataset, Subset

import contextlib
import pickle

from omen import OMEN, build_omen, canonical_architecture
from omen_scale_config import OMENScaleConfig
from omen_data import make_counting, make_python, make_rule_transfer, collate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# AMP is only available on CUDA (on CPU it becomes a no-op via enabled=False)
_AMP_AVAILABLE = torch.cuda.is_available()

if DEVICE.type == "cuda":
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def _configure_stdio() -> None:
    """Avoid console-encoding crashes on non-UTF8 terminals."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError, OSError):
            continue


_configure_stdio()


# ══════════════════════════════════════════════════════════════════════════════
# 0. STACKED DATASET  (zero-copy batching)
# ══════════════════════════════════════════════════════════════════════════════

class StackedDataset(Dataset):
    """
    Convert List[Tuple[Tensor,Tensor]] into two large tensors (N,T).
    This is done one time; after that, each batch is just a slice (zero-copy).

    Compared with the old approach (torch.stack on every batch):
      · Old: N_batches × torch.stack(8 tensors) = N_batches × ~0.1ms = accumulates
      · New: stack once during initialization, then __getitem__ = O(1) indirection

    On GPU with pin_memory=True, DataLoader transfers CPU->GPU data asynchronously
    while the GPU computes the previous batch (real overlap).
    """
    def __init__(self, data: List[Tuple[torch.Tensor, torch.Tensor]]):
        t0 = time.perf_counter()
        # Support two formats (same as collate):
        #   List[Tuple[src, tgt]] - real text (load_text_corpus)
        #   List[Tensor]          - synthetic (make_counting, make_python, ...)
        if isinstance(data[0], (tuple, list)):
            self.src = torch.stack([x[0] for x in data])   # (N, T)
            self.tgt = torch.stack([x[1] for x in data])   # (N, T)
        else:
            stacked  = torch.stack(data)                    # (N, T+1)
            self.src = stacked[:, :-1]                      # (N, T)
            self.tgt = stacked[:, 1:]                       # (N, T)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  [StackedDataset] {len(self.src):,} samples x T={self.src.shape[1]} "
              f"stacked in {elapsed:.0f}ms  "
              f"({self.src.numel()*2*4/1e6:.1f} MB RAM)")

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.src[i], self.tgt[i]


class StreamingTextDataset(Dataset):
    """
    File-backed UTF-8 byte dataset.

    Does not keep the full corpus in RAM: each __getitem__ reads only the needed
    byte window from disk. It is not an IterableDataset in the narrow sense, but
    for training it solves the main problem: the corpus scales without full preload.
    """
    def __init__(self, path: Union[str, Path], seq_len: int, max_samples: int = 50_000):
        self.path = Path(path)
        self.seq_len = int(seq_len)
        self.stride = max(1, self.seq_len // 2)
        file_size = self.path.stat().st_size
        usable = max(0, file_size - self.seq_len - 1)
        n_samples = 0 if usable <= 0 else (usable // self.stride) + 1
        self.n_samples = min(int(max_samples), int(n_samples))
        self.file_size = int(file_size)
        print(f"  [StreamingTextDataset] {self.path}: {self.n_samples:,} samples x {self.seq_len} bytes "
              f"(file={self.file_size/1e6:.1f} MB, stride={self.stride})")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if i < 0 or i >= self.n_samples:
            raise IndexError(i)
        start = i * self.stride
        with self.path.open("rb") as fh:
            fh.seek(start)
            chunk = fh.read(self.seq_len + 1)
        if len(chunk) < self.seq_len + 1:
            raise IndexError(i)
        src = torch.tensor(list(chunk[:-1]), dtype=torch.long)
        tgt = torch.tensor(list(chunk[1:]), dtype=torch.long)
        return src, tgt


def sample_examples(dataset: Union[Sequence, Dataset],
                    n_samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Select a few examples from a list/Subset/file-backed Dataset without preload."""
    total = len(dataset)
    if total == 0:
        return []
    take = min(n_samples, total)
    indices = random.sample(range(total), take)
    return [dataset[i] for i in indices]


def build_loader(dataset_obj: Dataset,
                 batch_size: int,
                 num_workers: int = 0,
                 shuffle: bool = True,
                 drop_last: bool = True) -> DataLoader:
    """
    Build a DataLoader with pin_memory (when CUDA is available) and prefetch_factor.
    num_workers=0 is safe on any GPU/CPU; use >0 only when CUDA is stable.
    """
    loader_kwargs = {
        "dataset": dataset_obj,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "pin_memory": (DEVICE.type == "cuda"),
        "num_workers": num_workers,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = True
    return DataLoader(
        **loader_kwargs,
    )


def _finite_metric(value: object, default: float = 0.0) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return float(default)
    return scalar if math.isfinite(scalar) else float(default)


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_text_corpus(path: str, seq_len: int,
                     max_samples: int = 50_000) -> StreamingTextDataset:
    """
    Load real text as byte sequences.

    UTF-8 text -> int bytes [0..255] -> (src, tgt) pairs.
    Stride = seq_len//2 for 50% overlap between neighboring examples.
    """
    return StreamingTextDataset(path, seq_len=seq_len, max_samples=max_samples)


def make_synthetic_dataset(cfg: OMENScaleConfig,
                           n: int = 512) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Synthetic dataset: counting + python + rule_transfer."""
    ds: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for gen in [make_counting, make_python, make_rule_transfer]:
        ds.extend(gen(n // 3, cfg.seq_len))
    random.shuffle(ds)
    return ds


# ══════════════════════════════════════════════════════════════════════════════
# 2. STAGE 1 — NET PRETRAINING
# ══════════════════════════════════════════════════════════════════════════════

def pretrain_net(model: OMEN,
                 dataset: Union[Sequence[Tuple[torch.Tensor, torch.Tensor]], Dataset],
                 n_steps: int = 500,
                 batch_size: int = 8,
                 lr: float = 3e-4,
                 log_every: int = 50,
                 use_amp: bool = False,
                 num_workers: int = 0) -> Dict:
    """
    Stage 1: NET pretraining without the symbolic stack.
    Optimizes only: L_NET = L_code + L_rec + L_vocab + lambda_vq * L_vq.
    All remaining parameters are frozen.
    """
    if not model.net_enabled:
        print("  [Stage 1] NET disabled - skipping")
        return {}
    if len(dataset) == 0:
        print("  [Stage 1] empty dataset — skipping")
        return {}

    print("\n" + "═" * 68)
    print("  STAGE 1: NET Pretraining  (byte-level compression)")
    print("═" * 68)

    # Freeze everything except NET
    for name, p in model.named_parameters():
        p.requires_grad = "net." in name

    net_params = [p for p in model.parameters() if p.requires_grad]
    print(f"  NET params: {sum(p.numel() for p in net_params):,}")

    opt   = AdamW(net_params, lr=lr, weight_decay=1e-5)
    sched = CosineAnnealingLR(opt, T_max=max(n_steps, 1), eta_min=lr * 0.01)
    history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=log_every))
    use_amp = use_amp and _AMP_AVAILABLE
    _device_type = "cuda" if DEVICE.type == "cuda" else "cpu"

    def _amp_ctx():
        return torch.autocast(
            device_type=_device_type,
            dtype=torch.float16 if use_amp else torch.float32,
            enabled=use_amp,
        )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
        init_scale=2.0 ** 10,
    )
    loader = build_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=False,
    )
    loader_iter = iter(loader)
    amp_overflow_steps = 0.0

    model.net.train()
    t0 = time.perf_counter()

    for step in range(1, n_steps + 1):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            src, tgt = batch
        else:
            src = batch[:, :-1]
            tgt = batch[:, 1:]
        non_blocking = DEVICE.type == "cuda"
        src = src.to(DEVICE, non_blocking=non_blocking)
        tgt = tgt.to(DEVICE, non_blocking=non_blocking)

        opt.zero_grad(set_to_none=True)

        with _amp_ctx():
            h_q, _, vq_info = model.net.encode(src)
            z_dummy = torch.zeros(
                src.size(0),
                model.cfg.d_latent,
                device=DEVICE,
                dtype=h_q.dtype,
            )
            _, l_rec = model.net.decode(
                tgt,
                z_dummy,
                h_q,
                return_logits=False,
                return_recon_loss=True,
            )

            # Non-autoregressive reconstruction loss (positional, without causal bypass).
            # Without it, the decoder can learn to predict tgt[i+1] from tgt[0..i]
            # without using h_q, which weakens the gradient through h_q to the encoder.
            # stage1_rec_loss forces h_q[i] to encode src[i] directly.
            l_nonauto = model.net.stage1_rec_loss(h_q, src)

            loss_dict = model.net.compute_loss(vq_info, l_rec)
            loss = loss_dict["net_total"] + 0.5 * l_nonauto

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        prev_scale = float(scaler.get_scale()) if use_amp else 1.0
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net_params, 1.0)
        scaler.step(opt)
        scaler.update()
        next_scale = float(scaler.get_scale()) if use_amp else prev_scale
        if use_amp and next_scale < prev_scale:
            amp_overflow_steps += 1.0
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

    # Unfreeze all parameters
    for p in model.parameters():
        p.requires_grad = True

    q = model.net.quantizer
    print(f"\n  Stage 1 done. vocab={q.current_size.item()}/{q.max_vocab}  "
          f"new_tokens={q.n_new_tokens}  "
          f"kb_facts={model.prover.kb.n_facts()}")
    return {
        "final_vocab": q.current_size.item(),
        "new_tokens": q.n_new_tokens,
        "amp_overflow_steps": amp_overflow_steps,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. STAGE 2 — JOINT TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def joint_train(model: OMEN,
                dataset: Union[Sequence[Tuple[torch.Tensor, torch.Tensor]], Dataset],
                n_epochs: int = 10,
                batch_size: int = 8,
                lr: float = 1e-4,
                weight_decay: float = 1e-2,
                max_batches_per_epoch: Optional[int] = None,
                checkpoint_dir: Optional[str] = None,
                use_amp: bool = False,
                grad_accum: int = 1,
                num_workers: int = 0,
                resume_state: Optional[dict] = None,
                start_epoch: int = 0) -> List[Dict]:
    """
    Stage 2: joint OMEN-Scale training.
    NET uses a smaller LR (0.3x) so it does not destroy the Stage-1 codebook.

    Optimizations (without quality loss):
      · StackedDataset + DataLoader -> zero-copy batching + pin_memory async transfer
      · AMP (autocast + GradScaler) -> FP16 matmul on GPU (40-100% faster)
      · grad_accum > 1 -> fewer optimizer.step() calls (effective_batch = batch * accum)
      · Throughput report: samples/sec, tokens/sec, ms/batch

    max_batches_per_epoch=None means the full dataset each epoch.
    Set it to a number to cap the run (for example 256 = 2048 samples / epoch).
    """
    use_amp = use_amp and _AMP_AVAILABLE  # AMP only on CUDA
    _device_type = "cuda" if DEVICE.type == "cuda" else "cpu"

    # torch.autocast is the unified API (PyTorch >= 2.0).
    # enabled=False acts like a nullcontext and is safe on CPU/GPU.
    def _amp_ctx():
        return torch.autocast(device_type=_device_type,
                              dtype=torch.float16 if use_amp else torch.float32,
                              enabled=use_amp)

    print("\n" + "═" * 68)
    print("  STAGE 2: Joint Training  (OMEN-Scale + NET)")
    print("═" * 68)

    # ── StackedDataset + DataLoader ───────────────────────────────────────────
    dataset_obj = dataset if isinstance(dataset, Dataset) else StackedDataset(dataset)
    loader  = build_loader(dataset_obj, batch_size, num_workers=num_workers, shuffle=True)

    # Compute the effective batch limit
    n_total_batches = len(loader)
    _max_bat = max_batches_per_epoch if max_batches_per_epoch is not None else n_total_batches
    _max_bat = min(_max_bat, n_total_batches)
    start_epoch = max(int(start_epoch), 0)
    target_epoch = max(int(n_epochs), start_epoch)

    effective_batch = batch_size * grad_accum
    print(f"  batches/epoch : {_max_bat}  ({_max_bat * batch_size} samples)")
    print(f"  effective_batch: {effective_batch}  (batch_size={batch_size} × grad_accum={grad_accum})")
    print(f"  AMP: {'✓ FP16' if use_amp else '✗ FP32'}  "
          f"pin_memory: {'✓' if DEVICE.type=='cuda' else '✗'}  "
          f"workers: {num_workers}")
    if start_epoch > 0:
        print(f"  [resume] continuing Stage 2 from epoch {start_epoch} to epoch {target_epoch}")
    if target_epoch <= start_epoch:
        print("  [resume] requested target epoch already reached; skipping Stage 2")
        return []

    net_params   = [p for n, p in model.named_parameters() if "net." in n]
    other_params = [p for n, p in model.named_parameters() if "net." not in n]

    opt = AdamW([
        {"params": net_params,   "lr": lr * 0.3, "weight_decay": 1e-5},
        {"params": other_params, "lr": lr,        "weight_decay": weight_decay},
    ])
    # Bug fix: restore optimizer state (Adam m1/m2) after resume
    if resume_state and resume_state.get("optimizer"):
        try:
            opt.load_state_dict(resume_state["optimizer"])
            print("  [resume] optimizer state restored")
        except Exception as e:
            print(f"  [resume] optimizer state restore failed: {e} — starting with a fresh optimizer")

    # Keep the scheduler aligned with the total Stage-2 epoch count.
    _warmup_ep = min(2, target_epoch)
    _decay_ep  = max(target_epoch - _warmup_ep, 1)
    warmup_sched = LinearLR(opt, start_factor=0.1, end_factor=1.0,
                            total_iters=_warmup_ep)
    decay_sched  = CosineAnnealingLR(opt, T_max=_decay_ep, eta_min=lr * 0.01)
    sched = SequentialLR(opt, schedulers=[warmup_sched, decay_sched],
                         milestones=[_warmup_ep])

    # AMP GradScaler (no-op when use_amp=False)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
        init_scale=2.0 ** 10,
        growth_interval=512,
    )
    if resume_state and resume_state.get("scheduler"):
        try:
            sched.load_state_dict(resume_state["scheduler"])
            print("  [resume] scheduler state restored")
        except Exception as e:
            print(f"  [resume] scheduler state restore failed: {e} — using a fresh scheduler")
    if resume_state and resume_state.get("scaler"):
        try:
            scaler.load_state_dict(resume_state["scaler"])
            print("  [resume] AMP scaler state restored")
        except Exception as e:
            print(f"  [resume] AMP scaler state restore failed: {e} — using a fresh scaler")

    hdr = (f"{'Ep':>4} {'CE':>7} {'World':>7} {'LScale':>7} "
           f"{'NetL':>7} {'Voc':>4} {'KBf':>4} {'PPL':>8} {'ms/b':>6} {'tok/s':>7}")
    print(f"\n  {hdr}")
    print("  " + "─" * len(hdr))

    def _optimizer_step() -> Dict[str, float]:
        scaler.unscale_(opt)
        gnorm_net = _finite_metric(torch.nn.utils.clip_grad_norm_(net_params, 0.5))
        gnorm_other = _finite_metric(torch.nn.utils.clip_grad_norm_(other_params, 1.0))
        prev_scale = float(scaler.get_scale()) if use_amp else 1.0
        scaler.step(opt)
        scaler.update()
        next_scale = float(scaler.get_scale()) if use_amp else prev_scale
        opt.zero_grad(set_to_none=True)
        model.memory.maybe_flush()
        overflow = 1.0 if use_amp and next_scale < prev_scale else 0.0
        return {
            "gnorm_net": gnorm_net,
            "gnorm_other": gnorm_other,
            "amp_overflow_steps": overflow,
            "optimizer_steps": 1.0,
        }

    results: List[Dict] = []

    for epoch in range(start_epoch + 1, target_epoch + 1):
        model.train()
        t0_epoch = time.perf_counter()
        agg: Dict[str, float] = defaultdict(float)
        n_bat = 0
        total_tokens = 0

        # timing buckets for throughput analysis
        t_data, t_fwd, t_bwd, t_opt = 0.0, 0.0, 0.0, 0.0

        opt.zero_grad(set_to_none=True)

        t_iter_start = time.perf_counter()
        for src, tgt in loader:
            t_data += time.perf_counter() - t_iter_start  # data prefetch time

            src = src.to(DEVICE, non_blocking=True)
            tgt = tgt.to(DEVICE, non_blocking=True)

            # ── Forward (with AMP when use_amp) ───────────────────────────────
            t1 = time.perf_counter()
            with _amp_ctx():
                out = model(
                    src,
                    tgt,
                    metric_profile="train_fast" if DEVICE.type == "cuda" else "full",
                )

            # ── NaN/Inf guard with per-component diagnostics ──────────────────
            if torch.isnan(out["total"]) or torch.isinf(out["total"]):
                # Find the problematic component
                bad = [k for k, v in out.items()
                       if k not in ("logits", "z", "emc_stop")
                       and isinstance(v, float) and (math.isnan(v) or math.isinf(v))]
                print(f"\n  WARNING: NaN/Inf in total (batch {n_bat+1}), skipping. "
                      f"Problematic keys: {bad if bad else 'unknown'}")
                t_iter_start = time.perf_counter()
                continue
            t_fwd += time.perf_counter() - t1

            # ── Backward (scaled for AMP) ─────────────────────────────────────
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

            # ── Optimizer step every grad_accum steps ─────────────────────────
            if n_bat % grad_accum == 0:
                t3 = time.perf_counter()
                step_stats = _optimizer_step()
                for key, value in step_stats.items():
                    agg[key] += value
                t_opt += time.perf_counter() - t3

            # ── Progress counter ───────────────────────────────────────────────
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

        # Final flush if accumulated gradients remain
        if n_bat % grad_accum != 0:
            step_stats = _optimizer_step()
            for key, value in step_stats.items():
                agg[key] += value

        sched.step()

        if n_bat == 0:
            continue

        epoch_time = time.perf_counter() - t0_epoch
        n_opt_steps = max(int(agg.get("optimizer_steps", 0.0)), 1)
        avg = {k: v / n_bat for k, v in agg.items()}
        # Grad norms averaged by the number of optimizer steps
        avg["gnorm_net"]   = agg.get("gnorm_net",   0.0) / n_opt_steps
        avg["gnorm_other"] = agg.get("gnorm_other", 0.0) / n_opt_steps
        avg["amp_overflow_steps"] = agg.get("amp_overflow_steps", 0.0)
        avg["ppl"]       = math.exp(min(avg.get("ce", 10), 10))
        avg["ms"]        = (epoch_time / n_bat) * 1000
        avg["tok_s"]     = total_tokens / epoch_time
        avg["epoch"]     = float(epoch)
        avg["net_vocab"] = model.net.quantizer.current_size.item() if model.net_enabled else 0
        avg["kb_facts"]  = model.prover.kb.n_facts()
        avg["lr"]        = opt.param_groups[-1]["lr"]  # main-group LR

        # Throughput breakdown (epoch 1 only)
        if epoch == 1:
            print(f"\n  ── Epoch 1 throughput breakdown ──")
            print(f"     data prefetch : {t_data*1000/n_bat:5.1f} ms/batch")
            print(f"     forward       : {t_fwd*1000/n_bat:5.1f} ms/batch")
            print(f"     backward      : {t_bwd*1000/n_bat:5.1f} ms/batch")
            print(f"     opt+step      : {t_opt*1000/n_bat:5.1f} ms/batch  "
                  f"(every {grad_accum} batches)")
            print(f"     total/batch   : {avg['ms']:5.1f} ms/batch")
            print(f"     throughput    : {avg['tok_s']:,.0f} tok/s\n")
            print(f"  {hdr}")
            print("  " + "─" * len(hdr))

        # ── PPL spike detector ────────────────────────────────────────────────
        _prev_ppl = results[-1]["ppl"] if results else avg["ppl"]
        _spike = avg["ppl"] > _prev_ppl * 3.0 and avg["ppl"] > 20.0
        if _spike:
            print(f"\n  PPL SPIKE: {_prev_ppl:.1f} -> {avg['ppl']:.1f}  Loss components:")
            diag_keys = ["ce", "world", "l_scale", "sym_ground", "ltm_pen",
                         "ltm_pen_raw", "net_loss", "meta_loss", "traj_reward",
                         "curiosity", "recall", "novelty", "vem_pen",
                         "gnorm_net", "gnorm_other", "lr",
                         # OSF: J_OSF components for spike diagnostics
                         "osf_l_plan", "osf_l_sim", "osf_l_refl", "osf_l_meta",
                         "osf_l_intent", "osf_struct", "osf_plan_rl"]
            for dk in diag_keys:
                v = avg.get(dk, 0.0)
                flag = " ⚠️" if abs(v) > 5.0 else ""
                print(f"     {dk:<18}: {v:+10.5f}{flag}")

        # ── Compact epoch line ────────────────────────────────────────────────
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

        # ── Detailed loss-component line ──────────────────────────────────────
        print(f"       "
              f"sym={avg.get('sym_ground',0):.3f} "
              f"ltm={avg.get('ltm_pen',0):.3f}({avg.get('ltm_pen_raw',0):.1f}) "
              f"meta={avg.get('meta_loss',0):.3f} "
              f"traj_r={avg.get('traj_reward',0):.3f} "
              f"vem={avg.get('vem_pen',0):.4f} "
              f"cur={avg.get('curiosity',0):.3f} "
              f"|∇net|={avg.get('gnorm_net',0):.3f} "
              f"|∇oth|={avg.get('gnorm_other',0):.3f} "
              f"lr={avg.get('lr',0):.2e} "
              f"amp_of={avg.get('amp_overflow_steps',0):.0f}")

        # ── EMC line (when enabled) ───────────────────────────────────────────
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

        # ── OSF line (when enabled) ───────────────────────────────────────────
        # Displays all J_OSF components plus the strategic meta-controller state.
        # sigma: 0=Fast, 1=Careful, 2=Exploratory - epoch average.
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

        if checkpoint_dir and (epoch % 2 == 0 or epoch == target_epoch):
            _save_ckpt(model, opt, sched, scaler, epoch, avg, checkpoint_dir)

    print("  " + "─" * len(hdr))
    if results:
        print(f"  Best PPL: {min(r['ppl'] for r in results):.2f}")
    return results


def _serialize_kb(kb) -> dict:
    """Full TensorKnowledgeBase serialization: facts, rules, records, counters."""
    return {
        "fact_buf":   kb._fact_buf[:kb._n_facts].cpu().clone(),   # (n_facts, 3) int64
        "rules":      pickle.dumps(list(kb.rules)),               # bytes → safe for weights_only=True
        "n_facts":    kb._n_facts,
        "n_rules":    kb._n_rules,
        "n_proposed": kb._n_proposed,
        "n_verified": kb._n_verified,
        "records":    pickle.dumps(dict(kb._records)),             # bytes -> safe (RuleRecord contains HornClause)
        "fact_set":   set(kb._fact_set),
        "rule_hash_set": set(kb._rule_hash_set),
    }


def _restore_structural_rule_set(kb_state: dict, rules: Sequence) -> set:
    """
    Normalize a legacy checkpoint state into a structural rule set.

    Older checkpoints could store only int hash values or omit the rule set
    entirely. For TensorKnowledgeBase this breaks dedupe after restore, so
    we always rebuild the set from real HornClause objects here.
    """
    raw_rule_set = kb_state.get("rule_hash_set")
    if raw_rule_set is None:
        raw_rule_set = kb_state.get("rule_set")
    if not raw_rule_set:
        return set(rules)
    structural_rules = {rule for rule in rules if rule in raw_rule_set}
    return structural_rules if len(structural_rules) == len(rules) else set(rules)


def _restore_kb(model, kb_state: dict, device) -> None:
    """Restore TensorKnowledgeBase from serialized state."""
    kb = model.prover.kb
    # Restore facts
    n_f = kb_state["n_facts"]
    if n_f > 0:
        kb._fact_buf[:n_f] = kb_state["fact_buf"].to(device)
    kb._n_facts       = n_f
    kb._fact_set      = kb_state.get("fact_set", set())
    # Restore rules
    kb.rules          = pickle.loads(kb_state["rules"]) if isinstance(kb_state["rules"], bytes) else kb_state["rules"]
    kb._n_rules       = kb_state["n_rules"]
    kb._n_proposed    = kb_state["n_proposed"]
    kb._n_verified    = kb_state["n_verified"]
    kb._records       = (pickle.loads(kb_state["records"])
                         if isinstance(kb_state["records"], bytes)
                         else kb_state.get("records", {}))
    kb._rule_hash_set = _restore_structural_rule_set(kb_state, kb.rules)
    # Invalidate the fact cache
    kb._facts_cache   = None
    kb._facts_cache_n = -1
    print(f"    [restore] KB: {n_f} facts, {kb_state['n_rules']} rules")


def _save_ckpt(model, optimizer, scheduler, scaler, epoch, metrics, save_dir):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    path = str(Path(save_dir) / f"omen_epoch{epoch:04d}.pt")

    # ── KB: full serialization (Bug 1 fix) ────────────────────────────────────
    kb_state = _serialize_kb(model.prover.kb)

    # ── Memory cache: episodic recall (Bug 3 fix) ───────────────────────────────
    runtime_state = model.export_runtime_state()
    mem_cache = list(runtime_state.get("memory", {}).get("cache", ()))

    # ── EMC: meta statistics (Actor/Critic/StoppingUtility weights live in model.state_dict())
    # EfficientMetaController is registered as nn.Module via self.emc, so its weights
    # (Actor, Critic, StoppingUtility.task_estimator, EMCStateEncoder) are saved
    # automatically through model.state_dict(). We also store meta statistics for
    # diagnostics and restore-integrity checks.
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
            # Confirm that weights are saved through model.state_dict()
            "actor_param_count":  sum(p.numel() for p in emc.actor.parameters()),
            "critic_param_count": sum(p.numel() for p in emc.critic.parameters()),
            "stop_param_count":   sum(p.numel() for p in emc.stopping_utility.parameters()),
        }

    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else {},
        "scheduler": scheduler.state_dict() if scheduler else {},
        "scaler":    scaler.state_dict() if scaler else {},
        "metrics":   metrics,
        "net_vocab": model.net.quantizer.current_size.item() if model.net_enabled else 0,
        "net_tau":   float(model.net.quantizer.tau) if model.net_enabled else None,
        "kb_facts":  kb_state["n_facts"],    # quick glance value
        "kb_state":  kb_state,               # full KB state
        "runtime_state": runtime_state,
        "mem_cache": mem_cache,              # episodic recall cache
        "emc_meta":  emc_meta,              # EMC hyperparameters + param counts (weights live in model)
        # OSF: save the running CE estimate so the meta-controller restores
        # a realistic quality estimate instead of hardcoded 5.0 after loading.
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
def net_diagnostics(model: OMEN,
                    dataset: Union[Sequence[Tuple[torch.Tensor, torch.Tensor]], Dataset],
                    n_samples: int = 32) -> None:
    if not model.net_enabled:
        return
    model.eval()
    q   = model.net.quantizer
    V   = q.current_size.item()
    batch = sample_examples(dataset, n_samples)
    src, _ = collate(batch)
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
    model   = build_omen(cfg, device=DEVICE)
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
    # Check that best PPL beats the first step (not necessarily the last)
    assert best_ppl < ppls[0] * 0.95 or best_ppl < ppls[0], \
        f"PPL did not improve: best={best_ppl:.1f} vs initial={ppls[0]:.1f}"
    print(f"\n  PPL: {ppls[0]:.1f} → min={best_ppl:.1f} (ep{ppls.index(best_ppl)+1})  ✓")

    vocab_final = model.net.quantizer.current_size.item()
    assert vocab_final > cfg.net_init_vocab, "vocab did not grow"
    print(f"  NET vocab: {cfg.net_init_vocab} → {vocab_final}  ✓")

    assert model.prover.kb.n_facts() > 0, "KB is empty"
    print(f"  KB facts: {model.prover.kb.n_facts()}  ✓")

    print(f"\n{model.memory_report()}")
    print("\n  Smoke test passed")


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
    # FIX 4: max_batches used to be hardcoded to 32 -> 256 samples / 45,000 = 0.57%.
    # None = full dataset (one full epoch). A numeric value = batch limit.
    # Recommendation for RTX 3080 + strong config:
    #   --max_batches 256  (~2048 samples / epoch, ~2 min / epoch)
    #   --max_batches 500  (~4000 samples / epoch, ~4 min / epoch)
    #   no argument        = full dataset, but slow
    p.add_argument("--max_batches", type=int, default=None,
                   help="Batch limit per epoch (None = full dataset). "
                        "Recommended: 256-500 for fast iterations.")
    # ── Speed optimization ────────────────────────────────────────────────────
    p.add_argument("--amp", action="store_true", default=False,
                   help="Enable AMP (FP16 autocast). "
                        "Gives 40-100%% speedup on CUDA GPUs. "
                        "Automatically disables itself when CUDA is unavailable.")
    p.add_argument("--grad_accum", type=int, default=1,
                   help="Gradient accumulation steps. "
                        "effective_batch = batch_size × grad_accum. "
                        "Recommended: 2-4 (reduces the number of opt.step() calls).")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers (0 = main thread). "
                        "On CUDA, you can try 2-4 for prefetch.")
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

    canon = canonical_architecture()
    print(
        f"Config={args.config}  device={DEVICE}  NET={'on' if cfg.net_enabled else 'off'}  "
        f"canonical={canon.stack_id}"
    )

    # Hint: GPU optimization
    if DEVICE.type == "cuda" and not args.amp:
        print("\n  HINT: you are on GPU but --amp is not enabled.")
        print("     Recommended: --amp --grad_accum 2")
        print("     Expected speedup: ~2x forward/backward (FP16 tensor cores)\n")

    if args.real_text:
        dataset = load_text_corpus(args.real_text, cfg.seq_len)
    else:
        dataset = make_synthetic_dataset(cfg, n=512)

    split = int(0.9 * len(dataset))
    if isinstance(dataset, Dataset):
        train_ds = Subset(dataset, range(split))
    else:
        train_ds = dataset[:split]

    model = build_omen(cfg, device=DEVICE)
    _resume_training_state = None
    _resume_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        _resume_epoch = int(ckpt.get("epoch", 0) or 0)
        # Bug 4 fix: restore KB and memory cache
        if "kb_state" in ckpt:
            _restore_kb(model, ckpt["kb_state"], DEVICE)
        if "runtime_state" in ckpt:
            model.load_runtime_state(ckpt["runtime_state"], device=DEVICE)
        elif "mem_cache" in ckpt:
            model.memory.cache.extend(
                [(s.to(DEVICE), v.to(DEVICE)) for s, v in ckpt["mem_cache"]]
            )
        # Bug fix: restore net_tau (Python float - not part of state_dict)
        if "net_tau" in ckpt and ckpt["net_tau"] is not None and model.net_enabled:
            model.net.quantizer.tau = float(ckpt["net_tau"])
        _resume_training_state = {
            "optimizer": ckpt.get("optimizer") or None,
            "scheduler": ckpt.get("scheduler") or None,
            "scaler": ckpt.get("scaler") or None,
        }
        # Backward compatibility for checkpoints that predate runtime_state.
        if "runtime_state" not in ckpt and "osf_running_ce" in ckpt and getattr(model, 'osf_enabled', False):
            model._osf_running_ce = float(ckpt["osf_running_ce"])

        # ── EMC state restore check ───────────────────────────────────────────
        # EfficientMetaController (Actor + Critic + StoppingUtility + StateEncoder)
        # saved through model.state_dict() -> restored through load_state_dict().
        # Here we verify that EMC param counts match the expected meta-info.
        emc_meta_ckpt = ckpt.get("emc_meta", {})
        if emc_meta_ckpt and getattr(model, 'emc_enabled', False) and hasattr(model, 'emc'):
            emc = model.emc
            ok_actor  = sum(p.numel() for p in emc.actor.parameters()) == emc_meta_ckpt.get("actor_param_count", -1)
            ok_critic = sum(p.numel() for p in emc.critic.parameters()) == emc_meta_ckpt.get("critic_param_count", -1)
            ok_stop   = sum(p.numel() for p in emc.stopping_utility.parameters()) == emc_meta_ckpt.get("stop_param_count", -1)
            status = "✓" if (ok_actor and ok_critic and ok_stop) else "⚠ mismatch"
            print(f"  [resume] EMC weights restored {status} "
                  f"(actor={ok_actor} critic={ok_critic} stoputil={ok_stop})")

        runtime_cache = 0
        if "runtime_state" in ckpt:
            runtime_cache = len(ckpt["runtime_state"].get("memory", {}).get("cache", ()))
        else:
            runtime_cache = len(ckpt.get("mem_cache", []))
        print(f"  [resume] loaded epoch={ckpt.get('epoch', '?')}  "
              f"KB={ckpt.get('kb_facts', '?')} facts  "
              f"cache={runtime_cache}  "
              f"tau={ckpt.get('net_tau', '?')}")

    # Stage 1: skip when resuming (NET restored from checkpoint)
    if args.resume:
        print("  [resume] Stage 1 skipped because NET state was restored from the checkpoint")
    else:
        pretrain_net(model, train_ds, n_steps=args.stage1_steps,
                     batch_size=args.batch_size,
                     use_amp=args.amp,
                     num_workers=args.num_workers)
    if cfg.net_enabled:
        net_diagnostics(model, train_ds)

    joint_train(model, train_ds, n_epochs=args.stage2_epochs,
                batch_size=args.batch_size, lr=args.lr,
                max_batches_per_epoch=args.max_batches,
                checkpoint_dir=args.checkpoint_dir,
                use_amp=args.amp,
                grad_accum=args.grad_accum,
                num_workers=args.num_workers,
                resume_state=_resume_training_state,
                start_epoch=_resume_epoch)
    print(model.memory_report())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        smoke_test()
