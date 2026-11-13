#!/usr/bin/env python3
"""对比同一份 mid200 v2 ckpt 在 MPS / CPU 上续训 1 步的 loss 与梯度范数。

只读 checkpoint，不写、不覆盖任何权重。用法：

  .venv/bin/python -u scripts/diag_resume_one_step.py
  .venv/bin/python -u scripts/diag_resume_one_step.py --devices mps --micros 8
  .venv/bin/python -u scripts/diag_resume_one_step.py --devices cpu --micros 1
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import load_memmap_datasets  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402


def _param_grad_norms(model: torch.nn.Module) -> OrderedDict:
    layers: OrderedDict[str, float] = OrderedDict()
    buckets: dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        sq = float(g.norm(2).item() ** 2)
        if name.startswith("blocks."):
            key = ".".join(name.split(".")[:2])  # blocks.N
        elif name.startswith("wte") or name.startswith("transformer.wte"):
            key = "wte"
        elif name.startswith("wpe") or name.startswith("transformer.wpe"):
            key = "wpe"
        elif "lm_head" in name:
            key = "lm_head"
        elif name.startswith("ln_f") or name.endswith("ln_f.weight") or "ln_f" in name:
            key = "ln_f"
        else:
            key = name.split(".")[0]
        buckets[key] = buckets.get(key, 0.0) + sq
    for k, sq in buckets.items():
        layers[k] = sq ** 0.5
    return layers


def _global_grad_stats(model: torch.nn.Module) -> dict:
    total_sq = 0.0
    max_abs = 0.0
    n_nan = 0
    n_inf = 0
    n_elem = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        n_elem += g.numel()
        n_nan += int(torch.isnan(g).sum().item())
        n_inf += int(torch.isinf(g).sum().item())
        finite = g[torch.isfinite(g)]
        if finite.numel():
            max_abs = max(max_abs, float(finite.abs().max().item()))
            total_sq += float(finite.norm(2).item() ** 2)
    return {
        "grad_norm": total_sq ** 0.5,
        "max_abs": max_abs,
        "n_nan": n_nan,
        "n_inf": n_inf,
        "n_elem": n_elem,
    }


def run_one(device: str, ckpt: dict, args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    if device == "cpu" and args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
        try:
            torch.set_num_interop_threads(max(1, min(args.cpu_threads, 2)))
        except RuntimeError:
            pass

    mc = ckpt.get("model_config") or {}
    gpt_cfg = GPTConfig(
        vocab_size=int(mc.get("vocab_size", 151665)),
        block_size=int(mc.get("block_size", 512)),
        n_layer=int(mc.get("n_layer", 12)),
        n_head=int(mc.get("n_head", 12)),
        n_embd=int(mc.get("n_embd", 768)),
        dropout=float(mc.get("dropout", 0.0)),
        bias=bool(mc.get("bias", True)),
    )

    t0 = time.time()
    train_ds, _val_ds = load_memmap_datasets(args.train_bin, args.val_bin, gpt_cfg.block_size)
    model = GPT(gpt_cfg).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
    # load_state_dict returns IncompatibleKeys; empty is expected
    model.train()
    load_s = time.time() - t0

    micros = max(1, int(args.micros))
    batch = 1
    first_x = None
    loss_accum = 0.0
    t1 = time.time()
    model.zero_grad(set_to_none=True)
    for i in range(micros):
        xb, yb = train_ds.get_batch(batch, device)
        if i == 0:
            first_x = xb[0, :8].detach().to("cpu").tolist()
        _logits, loss = model(xb, yb)
        (loss / micros).backward()
        loss_accum += float(loss.item())
        print(
            f"  [{device}] micro {i+1}/{micros} loss={float(loss.item()):.4f}",
            flush=True,
        )
    step_s = time.time() - t1
    loss_mean = loss_accum / micros
    stats = _global_grad_stats(model)
    layers = _param_grad_norms(model)

    print(
        f"[{device}] iter={ckpt.get('iter')}  micros={micros}  "
        f"loss={loss_mean:.4f}  grad_norm={stats['grad_norm']:.4f}  "
        f"max_abs={stats['max_abs']:.4e}  nan={stats['n_nan']} inf={stats['n_inf']}  "
        f"first_x[0,:8]={first_x}  load={load_s:.1f}s step={step_s:.1f}s",
        flush=True,
    )
    print(f"[{device}] 分层 grad_norm:", flush=True)
    for k, v in layers.items():
        print(f"  {k:16s} {v:.4f}", flush=True)

    # 释放显存 / 内存，给下一设备腾地方
    del model, train_ds, _val_ds
    gc.collect()
    if device == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    return {
        "device": device,
        "loss": loss_mean,
        "grad_norm": stats["grad_norm"],
        "max_abs": stats["max_abs"],
        "n_nan": stats["n_nan"],
        "n_inf": stats["n_inf"],
        "first_x": first_x,
        "layers": dict(layers),
        "load_s": load_s,
        "step_s": step_s,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/ckpt_hf_mid200_v2.pt")
    p.add_argument("--train-bin", default="data/seed_v2_zh_hq_v2_train.bin")
    p.add_argument("--val-bin", default="data/seed_v2_zh_hq_v2_val.bin")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--micros", type=int, default=8, help="grad accum 微步数，默认与训练相同为 8")
    p.add_argument("--devices", default="mps,cpu")
    p.add_argument("--cpu-threads", type=int, default=8)
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        print(f"[错误] 找不到 ckpt: {ckpt_path}", file=sys.stderr)
        return 1

    print(
        f"只读诊断：ckpt={ckpt_path}  seed={args.seed}  micros={args.micros}  "
        f"devices={args.devices}  （不写 checkpoint）",
        flush=True,
    )
    t0 = time.time()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(
        f"已加载 ckpt：iter={ckpt.get('iter')}  "
        f"metrics={ckpt.get('metrics')}  用时 {time.time()-t0:.1f}s",
        flush=True,
    )

    results = []
    for raw in args.devices.split(","):
        device = raw.strip()
        if not device:
            continue
        if device == "mps":
            if not torch.backends.mps.is_available():
                print("[错误] 本机没有 MPS", file=sys.stderr)
                return 1
        elif device != "cpu":
            print(f"[错误] 不支持的设备: {device}", file=sys.stderr)
            return 1
        print(f"\n======== {device} ========", flush=True)
        results.append(run_one(device, ckpt, args))

    if len(results) >= 2:
        a, b = results[0], results[1]
        print("\n======== 对比 ========", flush=True)
        print(
            f"loss:      {a['device']}={a['loss']:.4f}  {b['device']}={b['loss']:.4f}  "
            f"差={a['loss']-b['loss']:+.4f}",
            flush=True,
        )
        ratio = (
            a["grad_norm"] / b["grad_norm"] if b["grad_norm"] else float("inf")
        )
        print(
            f"grad_norm: {a['device']}={a['grad_norm']:.4f}  {b['device']}={b['grad_norm']:.4f}  "
            f"比值 {a['device']}/{b['device']}={ratio:.3f}",
            flush=True,
        )
        print(f"first_x 相同: {a['first_x'] == b['first_x']}", flush=True)
        keys = list(dict.fromkeys(list(a["layers"]) + list(b["layers"])))
        print(f"{'layer':16s} {a['device']:>12s} {b['device']:>12s} {'比值':>8s}", flush=True)
        for k in keys:
            va = a["layers"].get(k, 0.0)
            vb = b["layers"].get(k, 0.0)
            r = va / vb if vb else float("inf")
            print(f"{k:16s} {va:12.4f} {vb:12.4f} {r:8.3f}", flush=True)

    print("\n诊断结束：没有写入任何 checkpoint。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
