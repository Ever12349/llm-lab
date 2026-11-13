#!/usr/bin/env python3
"""从 checkpoint 采样生成：默认 HF 词表；python generate.py [--prompt ...] [--bpe|--char]

默认流式输出：每采样一个 token 立刻 decode 并 flush 打印。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from config import MODEL_TIERS, TrainConfig, apply_model_tier, pick_device
from tokenizer import BPETokenizer, CharTokenizer, HFTokenizer, load_tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="用训练好的微型 GPT 生成文本")
    p.add_argument("--checkpoint", default=None, help="checkpoint 路径")
    p.add_argument("--tokenizer", default=None, help="本地 tokenizer 路径（仅 --bpe/--char）")
    p.add_argument(
        "--hf-tokenizer",
        default=None,
        help="HF tokenizer id 或本地目录（默认读 checkpoint / config）",
    )
    p.add_argument(
        "--hf-cache-dir",
        default=None,
        help="HF 缓存目录（默认 checkpoints/hf）",
    )
    p.add_argument("--prompt", default="\n", help="起始提示文本（默认换行）")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50, help="设为 0 关闭 top-k")
    p.add_argument("--device", default=None, help="cpu / cuda / mps")
    p.add_argument(
        "--hf",
        action="store_true",
        help="显式使用 HF 词表（当前默认，可省略）",
    )
    p.add_argument(
        "--bpe",
        action="store_true",
        help="加载自训 BPE：ckpt_bpe.pt + tokenizer_bpe.json",
    )
    p.add_argument(
        "--char",
        action="store_true",
        help="加载字符版对照：ckpt_char.pt + tokenizer.json",
    )
    p.add_argument(
        "--tier",
        default=None,
        choices=sorted(MODEL_TIERS.keys()),
        help="模型档位：mid100 → checkpoints/ckpt_hf_mid100.pt",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = TrainConfig()
    n_modes = sum(bool(x) for x in (args.char, args.bpe, args.hf))
    if n_modes > 1:
        print("[错误] --char / --bpe / --hf 只能选一个", file=sys.stderr)
        return 1

    if args.char:
        cfg.checkpoint_name = cfg.char_checkpoint_name
        cfg.tokenizer_name = cfg.char_tokenizer_name
        backend = "char"
    elif args.bpe:
        cfg.checkpoint_name = cfg.bpe_checkpoint_name
        cfg.tokenizer_name = cfg.bpe_tokenizer_name
        backend = "bpe"
    else:
        cfg.checkpoint_name = "ckpt_hf.pt"
        backend = "hf"
        if args.hf_tokenizer:
            cfg.hf_tokenizer_id = args.hf_tokenizer
        if args.hf_cache_dir:
            cfg.hf_cache_dir = args.hf_cache_dir
        if args.tier:
            apply_model_tier(cfg, args.tier)

    ckpt_path = Path(args.checkpoint or cfg.checkpoint_path())
    tok_path = Path(args.tokenizer or cfg.tokenizer_path())
    device = pick_device(args.device)

    if not ckpt_path.is_file():
        print(
            f"[错误] 找不到 checkpoint: {ckpt_path.resolve()}\n"
            "请先准备语料并运行: python train.py\n"
            "BPE 对照: python generate.py --bpe\n"
            "字符版对照: python generate.py --char",
            file=sys.stderr,
        )
        return 1

    print(
        f"加载 checkpoint: {ckpt_path.resolve()} (device={device})",
        file=sys.stderr,
        flush=True,
    )
    if device == "mps":
        print(
            "提示: 首次在 MPS 上初始化可能较慢，请稍候…",
            file=sys.stderr,
            flush=True,
        )

    try:
        import torch
    except ImportError:
        print(
            "[错误] 未安装 PyTorch。请先: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    from model import GPT, GPTConfig

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    saved_tc = ckpt.get("train_config") or {}
    # checkpoint 优先决定后端
    backend = ckpt.get("tokenizer_backend") or saved_tc.get("tokenizer_backend") or backend

    if backend == "hf":
        hf_id = (
            args.hf_tokenizer
            or ckpt.get("hf_tokenizer_id")
            or saved_tc.get("hf_tokenizer_id")
            or cfg.hf_tokenizer_id
        )
        cache = args.hf_cache_dir or saved_tc.get("hf_cache_dir") or cfg.hf_cache_dir
        # 优先本地已保存目录
        local_dir = Path(cache) / "Qwen2.5-1.5B-tokenizer"
        load_from = str(local_dir) if local_dir.is_dir() else hf_id
        print(f"加载 HF tokenizer: {load_from}", file=sys.stderr, flush=True)
        tokenizer = HFTokenizer.load(load_from, cache_dir=cache)
    else:
        if not tok_path.is_file():
            print(f"[错误] 找不到 tokenizer: {tok_path.resolve()}", file=sys.stderr)
            return 1
        if backend == "char":
            tokenizer = CharTokenizer.load(tok_path)
        else:
            tokenizer = load_tokenizer(tok_path)

    model_cfg = GPTConfig(**ckpt["model_config"])
    model = GPT(model_cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    print("开始生成…", file=sys.stderr, flush=True)

    prompt = args.prompt
    try:
        ids = tokenizer.encode(prompt)
    except KeyError as e:
        print(f"[错误] prompt 含未知字符: {e}", file=sys.stderr)
        return 1

    if len(ids) == 0:
        ids = [0]

    all_ids: List[int] = list(ids)
    prev_text = tokenizer.decode(all_ids)
    print(prev_text, end="", flush=True)

    def on_token(tid: int) -> None:
        nonlocal prev_text
        all_ids.append(tid)
        text = tokenizer.decode(all_ids)
        if text.startswith(prev_text):
            chunk = text[len(prev_text) :]
        else:
            chunk = tokenizer.decode([tid])
        prev_text = text
        if chunk:
            print(chunk, end="", flush=True)

    idx = torch.tensor([ids], dtype=torch.long, device=device)
    top_k = args.top_k if args.top_k > 0 else None
    with torch.no_grad():
        model.generate(
            idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=top_k,
            token_callback=on_token,
        )
    print(flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
