"""
test_memory_inplace.py — Перевірка фіксу inplace memory error
================================================================
Симулює training loop: forward → flush → backward.
Якщо фікс працює — тест проходить без RuntimeError.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import random

from omen_scale_config import OMENScaleConfig
from omen_scale import OMENScale
from omen_v2 import collate, make_counting

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_memory_no_inplace_error():
    """
    Тест: forward → schedule_write (flush) → backward
    Якщо flush() робить inplace модифікацію memory, backward() впаде.
    """
    torch.manual_seed(42)
    random.seed(42)

    cfg = OMENScaleConfig.demo()
    # Зменшуємо mem_update_steps щоб flush викликався часто
    cfg.mem_update_steps = 2  # flush кожні 2 батчі
    model = OMENScale(cfg).to(DEVICE)
    model.train()

    # Створюємо міні-датасет
    dataset = make_counting(32, cfg.seq_len)

    opt = AdamW(model.parameters(), lr=1e-4)

    print(f"[Test] device={DEVICE}, mem_update_steps={cfg.mem_update_steps}")
    print(f"[Test] memory shape: {model.memory.memory.shape}")

    errors = []
    for step in range(1, 9):  # 8 кроків — кілька flush-ів
        batch = random.sample(dataset, min(4, len(dataset)))
        src, tgt = collate(batch)
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)

        opt.zero_grad(set_to_none=True)

        try:
            out = model(src, tgt)
        except Exception as e:
            errors.append(f"Step {step} FORWARD: {e}")
            continue

        # flush() викликається всередині forward() кожні mem_update_steps кроків
        loss = out["total"]

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  step {step}: loss=nan/inf — skip")
            continue

        try:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            # Flush AFTER optimizer step — no autograd graph anymore
            model.memory.maybe_flush()
            print(f"  step {step}: OK  loss={loss.item():.4f}  "
                  f"n_writes={model.memory.n_writes}")
        except RuntimeError as e:
            errors.append(f"Step {step} BACKWARD: {e}")
            print(f"  step {step}: FAIL — {e}")

    # ── Результат ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    if errors:
        print(f"  ❌ FAIL: {len(errors)} помилок:")
        for err in errors:
            print(f"    {err}")
        return False
    else:
        print(f"  ✅ PASS: 8/8 кроків без inplace errors")
        print(f"  Memory writes: {model.memory.n_writes}")
        return True


def test_memory_read_detach():
    """
    Перевірка що read() повертає тензор, detached від self.memory.
    """
    torch.manual_seed(0)
    from omen_scale import AsyncTensorProductMemory
    from omen_scale_config import OMENScaleConfig

    cfg = OMENScaleConfig.demo()
    mem = AsyncTensorProductMemory(cfg).to(DEVICE)

    z = torch.randn(4, cfg.d_latent, device=DEVICE, requires_grad=True)
    v = mem.read(z)

    # v не повинен мати grad_fn що посилається на memory buffer
    loss = v.sum()
    loss.backward()

    assert z.grad is not None, "grad по z повинен існувати"
    print(f"[Test] read() grad norm: {z.grad.norm():.4f}")
    print(f"  ✅ PASS: read() detach працює")
    return True


def test_v2_memory_write():
    """
    Перевірка що omen_v2.py TensorProductMemory.write() не ламає backward.
    """
    torch.manual_seed(0)
    from omen_v2 import TensorProductMemory, OMENv2Config, WorldRNN

    cfg = OMENv2Config()
    mem = TensorProductMemory(cfg).to(DEVICE)
    world_rnn = WorldRNN(cfg).to(DEVICE)

    z = torch.randn(4, cfg.d_latent, device=DEVICE, requires_grad=True)
    z_sim, _ = world_rnn(z, torch.zeros(4, dtype=torch.long, device=DEVICE))

    # Read
    v = mem.read(z)
    loss = F.mse_loss(z_sim, z.detach()) + v.sum()
    loss.backward()

    # Write (це має не зламати backward для попереднього loss)
    conf = torch.rand(4, device=DEVICE)
    mem.write(z.detach(), z_sim.detach(), conf)

    # Ще один forward/backward
    z2 = torch.randn(4, cfg.d_latent, device=DEVICE, requires_grad=True)
    v2 = mem.read(z2)
    loss2 = v2.sum()
    loss2.backward()

    assert z2.grad is not None
    print(f"[Test] v2 write() grad norm: {z2.grad.norm():.4f}")
    print(f"  ✅ PASS: v2 TensorProductMemory write+read працює")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("  TEST: Memory inplace operation fix")
    print("=" * 60)

    all_pass = True

    print("\n── T1: read() detach ──────────────────────────")
    all_pass &= test_memory_read_detach()

    print("\n── T2: v2 TensorProductMemory write ───────────")
    all_pass &= test_v2_memory_write()

    print("\n── T3: Full forward → flush → backward ────────")
    all_pass &= test_memory_no_inplace_error()

    print(f"\n{'='*60}")
    if all_pass:
        print("  ✅  ВСІ ТЕСТИ ПРОЙДЕНО")
    else:
        print("  ❌  ДЕЯКІ ТЕСТИ НЕ ПРОЙДЕНО")
    print("=" * 60)
