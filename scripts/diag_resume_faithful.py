#!/usr/bin/env python3
"""更贴近 train.py 的 1 步续训诊断：是否加载优化器、clip_grad_norm_、参数设备。只读，不写 ckpt。"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import load_memmap_datasets  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402


def layer_key(name: str) -> str:
    parts = name.split(".")
    # transformer.h.N....  / transformer.wte  / transformer.wpe  / transformer.ln_f  / lm_head
    if len(parts) >= 3 and parts[0] == "transformer" and parts[1] == "h":
        return f"h.{parts[2]}"
    if name.startswith("transformer.wte") or name.startswith("lm_head"):
        return "wte/lm_head(tied)"
    if name.startswith("transformer.wpe"):
        return "wpe"
    if name.startswith("transformer.ln_f"):
        return "ln_f"
    return parts[0]


def opt_state_devices(optimizer: torch.optim.Optimizer) -> str:
    counts: dict[str, int] = {}
    for st in optimizer.state.values():
        for v in st.values():
            if torch.is_tensor(v):
                key = str(v.device)
                counts[key] = counts.get(key, 0) + 1
    if not counts:
        return "(empty)"
    return ", ".join(f"{k}×{n}" for k, n in sorted(counts.items()))


def run(device: str, ckpt: dict, load_optim: bool, args: argparse.Namespace) -> dict:
    tag = f"{device} opt={'yes' if load_optim else 'no'}"
    torch.manual_seed(args.seed)
    if device == "cpu" and args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
        try:
            torch.set_num_interop_threads(max(1, min(args.cpu_threads, 2)))
        except RuntimeError:
            pass

    mc = ckpt["model_config"]
    gpt_cfg = GPTConfig(
        vocab_size=int(mc["vocab_size"]),
        block_size=int(mc["block_size"]),
        n_layer=int(mc["n_layer"]),
        n_head=int(mc["n_head"]),
        n_embd=int(mc["n_embd"]),
        dropout=float(mc.get("dropout", 0.0)),
        bias=bool(mc.get("bias", True)),
    )
    train_ds, _ = load_memmap_datasets(args.train_bin, args.val_bin, gpt_cfg.block_size)
    model = GPT(gpt_cfg).to(device)
    incompat = model.load_state_dict(ckpt["model"], strict=True)
    print(
        f"[{tag}] params={model.num_params():,}  "
        f"device={next(model.parameters()).device}  "
        f"missing={list(incompat.missing_keys)}  unexpected={list(incompat.unexpected_keys)}",
        flush=True,
    )

    decay, no_decay = [], []
    for _n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.1},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=1e-4,
        eps=1e-6,
    )
    if load_optim:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"[{tag}] optimizer state devices: {opt_state_devices(optimizer)}", flush=True)

    model.train()
    micros = 8
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0
    first_x = None
    first_dev = None
    t0 = time.time()
    for i in range(micros):
        xb, yb = train_ds.get_batch(1, device)
        if i == 0:
            first_x = xb[0, :8].detach().to("cpu").tolist()
            first_dev = str(xb.device)
        _logits, loss = model(xb, yb)
        (loss / micros).backward()
        loss_accum += float(loss.item())
        print(f"  [{tag}] micro {i+1}/8 loss={float(loss.item()):.4f}", flush=True)
    loss_mean = loss_accum / micros
    step_s = time.time() - t0

    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_params = sum(1 for _ in model.parameters())
    # 与 train.py 完全相同的 API
    clip_ret = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))

    buckets: dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        sq = float(p.grad.detach().float().norm(2).item() ** 2)
        k = layer_key(name)
        buckets[k] = buckets.get(k, 0.0) + sq
    layers = OrderedDict((k, buckets[k] ** 0.5) for k in buckets)

    print(
        f"[{tag}] loss={loss_mean:.4f}  clip_grad_norm_={clip_ret:.4f}  "
        f"params_with_grad={n_with_grad}/{n_params}  "
        f"xb.device={first_dev}  first_x={first_x}  step={step_s:.1f}s",
        flush=True,
    )
    for k, v in layers.items():
        print(f"  {k:20s} {v:.4f}", flush=True)

    del model, optimizer, train_ds
    gc.collect()
    if device == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    return {
        "tag": tag,
        "device": device,
        "load_optim": load_optim,
        "loss": loss_mean,
        "grad_norm": clip_ret,
        "first_x": first_x,
        "layers": dict(layers),
        "xb_device": first_dev,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/ckpt_hf_mid200_v2.pt")
    p.add_argument("--train-bin", default="data/seed_v2_zh_hq_v2_train.bin")
    p.add_argument("--val-bin", default="data/seed_v2_zh_hq_v2_val.bin")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--cpu-threads", type=int, default=8)
    p.add_argument(
        "--cases",
        default="mps:no,mps:yes,cpu:no,cpu:yes",
        help="逗号分隔，设备:yes/no（是否加载优化器）",
    )
    args = p.parse_args()

    print(f"只读诊断（贴近 train.py）：ckpt={args.ckpt}  不写权重", flush=True)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"ckpt iter={ckpt.get('iter')} metrics={ckpt.get('metrics')}", flush=True)

    results = []
    for raw in args.cases.split(","):
        raw = raw.strip()
        if not raw:
            continue
        dev, flag = raw.split(":")
        load_optim = flag.strip().lower() in ("yes", "1", "true", "on")
        print(f"\n======== {dev} load_optim={load_optim} ========", flush=True)
        results.append(run(dev.strip(), ckpt, load_optim, args))

    print("\n======== 汇总 ========", flush=True)
    print(f"{'case':22s} {'loss':>8s} {'grad_norm':>10s}  xb.device", flush=True)
    for r in results:
        print(
            f"{r['tag']:22s} {r['loss']:8.4f} {r['grad_norm']:10.4f}  {r['xb_device']}",
            flush=True,
        )
    print("诊断结束：没有写入任何 checkpoint。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
