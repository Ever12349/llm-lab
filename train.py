#!/usr/bin/env python3
"""训练入口：默认 HF 词表 + memmap；python train.py [--resume] [--bpe] [--char]

退出码：
  0  正常跑完本轮 max_iters（或已达到 --until-iter）
  1  用法 / 文件 / 依赖等错误；续训后梯度范数异常也走这里（不要自动续训）
  2  MPS/数值熔断（loss 为 NaN/Inf 或接近 0）——可以自动续训
  3  高损失熔断（loss 持续高于阈值，默认 >20）——不要自动续训
熔断时都不会覆盖主 checkpoint。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

# 给看门狗 / 脚本用的稳定退出码（不要改数字，scripts/watch_train.py 依赖它们）
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_FUSE_NUMERIC = 2
EXIT_FUSE_HIGH = 3

from config import MODEL_TIERS, TrainConfig, apply_model_tier, get_lr, pick_device
from dataset import build_datasets, load_memmap_datasets, load_text_corpus
from tokenizer import BPETokenizer, CharTokenizer, HFTokenizer


def apply_cpu_thread_limit(n: int) -> None:
    """限制 CPU 并行度；须在 import torch 之前设置环境变量更稳妥。"""
    if n <= 0:
        return
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)



def _loss_is_bad(loss_val: float, eps: float) -> bool:
    """True if loss is NaN/Inf or |loss| < eps (MPS hang often yields ~0)."""
    if not math.isfinite(loss_val):
        return True
    return abs(loss_val) < eps


def _rotate_checkpoint(main_path: Path, keep: int, it: int) -> None:
    """Copy main ckpt to <stem>_rot_{iter}.pt; keep only the newest `keep` rot files."""
    if keep <= 0:
        return
    main_path = Path(main_path)
    rot_path = main_path.with_name(f"{main_path.stem}_rot_{it}.pt")
    shutil.copy2(main_path, rot_path)
    print(f"旋转备份 → {rot_path}", flush=True)
    # Never touch *_backup_* milestone files; only <stem>_rot_*.pt
    rots = sorted(
        main_path.parent.glob(f"{main_path.stem}_rot_*.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    while len(rots) > keep:
        old = rots.pop(0)
        try:
            old.unlink()
            print(f"已删除旧旋转备份 → {old}", flush=True)
        except OSError as e:
            print(f"[警告] 无法删除旧旋转备份 {old}: {e}", flush=True)



def _make_adamw(*args, **kwargs):
    """创建 AdamW：关掉 fused/foreach，避开 MPS 上不稳的融合核。"""
    import torch

    kwargs.setdefault("foreach", False)
    kwargs.setdefault("fused", False)
    try:
        return torch.optim.AdamW(*args, **kwargs)
    except TypeError:
        kwargs.pop("fused", None)
        try:
            return torch.optim.AdamW(*args, **kwargs)
        except TypeError:
            kwargs.pop("foreach", None)
            return torch.optim.AdamW(*args, **kwargs)


def _optimizer_state_to_device(optimizer, device) -> int:
    """把优化器里的张量搬到和模型同一设备。返回搬动的张量个数。"""
    import torch

    moved = 0
    target = torch.device(device)
    for state in optimizer.state.values():
        for key, val in list(state.items()):
            if torch.is_tensor(val) and val.device != target:
                state[key] = val.to(device)
                moved += 1
    return moved


def _sync_device(device: str) -> None:
    import torch

    if device == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.synchronize()
        except Exception:
            pass
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _disable_mps_reduced_precision() -> None:
    import torch

    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return
    for name in (
        "allow_fp16_reduced_precision_reduction",
        "allow_bf16_reduced_precision_reduction",
    ):
        if hasattr(backend, name):
            try:
                setattr(backend, name, False)
            except Exception:
                pass


def _clip_grad_norm(params, max_norm: float) -> float:
    """裁剪前总范数；foreach=False，避免 MPS 融合实现算错范数。"""
    import torch

    try:
        return float(
            torch.nn.utils.clip_grad_norm_(params, max_norm, foreach=False)
        )
    except TypeError:
        return float(torch.nn.utils.clip_grad_norm_(params, max_norm))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="训练微型 GPT（默认 Hugging Face 词表）")
    p.add_argument(
        "--device",
        default=None,
        choices=["cpu", "mps", "cuda"],
        help="覆盖自动设备选择（默认：cuda > mps > cpu）",
    )
    p.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        metavar="N",
        help="限制 CPU 线程数（N>0 时设置 torch/OMP/MKL；0=不限制，默认）",
    )
    p.add_argument("--data", default=None, help="语料文件或目录（仅 --char 时使用）")
    p.add_argument(
        "--corpus-dir",
        default=None,
        help="多文件语料目录（仅 --char；默认也可用 data/corpus/）",
    )
    p.add_argument("--max-iters", type=int, default=None, help="本次再训练的迭代次数")
    p.add_argument(
        "--until-iter",
        type=int,
        default=None,
        metavar="N",
        help="目标全局 iter（续训时本轮步数=N-已完成iter；用于「训到 5 万步」这类目标。"
        "不传则仍按 --max-iters 表示「再跑多少步」）",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="从 checkpoint 继续训练",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="权重路径，覆盖默认",
    )
    p.add_argument(
        "--hf",
        action="store_true",
        help="显式使用 HF 词表（当前默认，可省略）",
    )
    p.add_argument(
        "--hf-tokenizer",
        default=None,
        help="Hugging Face tokenizer id（默认 Qwen/Qwen2.5-1.5B）",
    )
    p.add_argument(
        "--hf-cache-dir",
        default=None,
        help="HF 缓存目录（默认 checkpoints/hf）",
    )
    p.add_argument(
        "--bpe",
        action="store_true",
        help="自训 BPE 对照：tokenizer_bpe.json → ckpt_bpe.pt",
    )
    p.add_argument(
        "--char",
        action="store_true",
        help="字符级对照：读整份文本 + tokenizer.json → ckpt_char.pt",
    )
    p.add_argument(
        "--train-bin",
        default=None,
        help="memmap train，默认 data/train.bin",
    )
    p.add_argument(
        "--val-bin",
        default=None,
        help="memmap val，默认 data/val.bin",
    )
    p.add_argument(
        "--tokenizer",
        default=None,
        help="本地词表路径（仅 --bpe / --char）",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="覆盖 config.batch_size（减小可降低峰值内存）",
    )
    p.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="覆盖 config.block_size",
    )
    p.add_argument(
        "--tier",
        default=None,
        choices=sorted(MODEL_TIERS.keys()),
        help="模型档位：seed20(~20M) / mid100(~100–120M) / mid200(~200M)；"
        "写入对应 ckpt_hf_*.pt，不覆盖其他档位权重",
    )
    p.add_argument(
        "--step-sleep",
        type=float,
        default=0.0,
        metavar="SEC",
        help="每步结束后随机休眠 [0, SEC] 秒（SEC 为上界；0=不休眠，默认）。用于压低平均占用",
    )
    p.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        metavar="N",
        help="梯度累积步数（micro-batch×N=有效 batch；默认 1=不累积）",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="峰值学习率（覆盖 config / 档位默认；mid100 默认 3e-4，mid200 默认 1e-4）",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=None,
        metavar="N",
        help="线性 warmup 步数（0=不预热；本轮短于 N 时压缩到本轮长度）",
    )
    p.add_argument(
        "--min-lr",
        type=float,
        default=None,
        metavar="LR",
        help="余弦衰减下限（默认跟档位走；不传则 mid100 不衰减，mid200 为 1e-5）",
    )
    p.add_argument(
        "--loss-fuse-eps",
        type=float,
        default=1e-6,
        metavar="EPS",
        help="loss≈0 熔断阈值：|loss|<EPS 或 NaN/Inf 计为坏步（默认 1e-6）",
    )
    p.add_argument(
        "--loss-fuse-patience",
        type=int,
        default=3,
        metavar="N",
        help="连续 N 个坏步后中止且不写主 ckpt（默认 3；0=关闭熔断）",
    )
    p.add_argument(
        "--loss-fuse-high",
        type=float,
        default=20.0,
        metavar="LOSS",
        help="明显高于随机基线时的熔断阈值（默认 20；0=关闭此项）",
    )
    p.add_argument(
        "--loss-fuse-high-patience",
        type=int,
        default=3,
        metavar="N",
        help="步进 loss（按日志间隔）或评估连续 N 次高于阈值则中止（默认 3）",
    )
    p.add_argument(
        "--ckpt-rotate",
        type=int,
        default=6,
        metavar="K",
        help="每次成功保存主 ckpt 后保留最近 K 份 <stem>_rot_{iter}.pt（默认 6；0=不旋转）",
    )
    p.add_argument(
        "--probe",
        action="store_true",
        help="探路模式：照常训练但绝不写入主 ckpt / best / 旋转备份 / last_summary",
    )
    p.add_argument(
        "--grad-fuse-norm",
        type=float,
        default=50.0,
        metavar="N",
        help="续训后前若干步，若裁剪前梯度范数 >N 则中止且不写主 ckpt（默认 50；0=关闭）",
    )
    p.add_argument(
        "--grad-fuse-steps",
        type=int,
        default=20,
        metavar="K",
        help="梯度范数熔断只检查续训后的前 K 步（默认 20；0=关闭）",
    )
    return p.parse_args()


def _load_char_text(cfg) -> str:
    """字符版读语料：优先 data/corpus/*.txt，否则 data_path 单文件。"""
    from corpus import resolve_corpus_files, read_corpus_text

    corpus_dir = Path(cfg.corpus_dir)
    data_path = Path(cfg.data_path)
    if data_path.is_dir():
        files = resolve_corpus_files(corpus_dir=data_path)
        return read_corpus_text(files)
    if corpus_dir.is_dir() and any(corpus_dir.rglob("*.txt")):
        files = resolve_corpus_files(corpus_dir=corpus_dir)
        print(f"语料目录: {corpus_dir}（{len(files)} 个文件）")
        return read_corpus_text(files)
    return load_text_corpus(cfg.data_path)


def _apply_backend_defaults(cfg: TrainConfig, args: argparse.Namespace) -> str:
    """根据 --char / --bpe / 默认 hf 设置路径，返回 backend 名。"""
    n_modes = sum(bool(x) for x in (args.char, args.bpe, args.hf))
    if n_modes > 1:
        raise SystemExit("[错误] --char / --bpe / --hf 只能选一个")
    if args.char:
        cfg.tokenizer_backend = "char"
        cfg.checkpoint_name = cfg.char_checkpoint_name
        cfg.tokenizer_name = cfg.char_tokenizer_name
        return "char"
    if args.bpe:
        cfg.tokenizer_backend = "bpe"
        cfg.checkpoint_name = cfg.bpe_checkpoint_name
        cfg.tokenizer_name = cfg.bpe_tokenizer_name
        return "bpe"
    # 默认 / --hf
    cfg.tokenizer_backend = "hf"
    cfg.checkpoint_name = "ckpt_hf.pt"
    if args.hf_tokenizer:
        cfg.hf_tokenizer_id = args.hf_tokenizer
    if args.hf_cache_dir:
        cfg.hf_cache_dir = args.hf_cache_dir
    return "hf"


def main() -> int:
    args = parse_args()
    apply_cpu_thread_limit(args.cpu_threads)
    cfg = TrainConfig()
    if args.device:
        cfg.device = pick_device(args.device)
    else:
        cfg.device = pick_device(cfg.device if cfg.device else None)
    if args.data:
        cfg.data_path = args.data
    if args.corpus_dir:
        cfg.corpus_dir = args.corpus_dir
    if args.max_iters is not None:
        cfg.max_iters = args.max_iters
    if args.train_bin:
        cfg.train_bin = args.train_bin
    if args.val_bin:
        cfg.val_bin = args.val_bin

    try:
        backend = _apply_backend_defaults(cfg, args)
    except SystemExit as e:
        print(e, file=sys.stderr)
        return 1

    if args.tier:
        try:
            apply_model_tier(cfg, args.tier)
        except ValueError as e:
            print(f"[错误] {e}", file=sys.stderr)
            return 1
        print(
            f"档位: {args.tier} — {MODEL_TIERS[args.tier]['summary']} "
            f"(L={cfg.n_layer}, H={cfg.n_head}, d={cfg.n_embd}, "
            f"block={cfg.block_size}, batch={cfg.batch_size})",
            flush=True,
        )

    # CLI 覆盖放在档位之后，这样 --batch-size / --block-size / --lr 才能限性能或改档位默认
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.block_size is not None:
        cfg.block_size = args.block_size
    if args.lr is not None:
        cfg.learning_rate = float(args.lr)
    if args.warmup is not None:
        cfg.warmup_iters = int(args.warmup)
    if args.min_lr is not None:
        cfg.min_lr = float(args.min_lr)

    if args.checkpoint:
        cp = Path(args.checkpoint)
        if cp.parent != Path("."):
            cfg.checkpoint_dir = str(cp.parent)
            cfg.checkpoint_name = cp.name
        else:
            cfg.checkpoint_name = cp.name
    if args.tokenizer and backend in ("bpe", "char"):
        tp = Path(args.tokenizer)
        if tp.parent != Path("."):
            cfg.checkpoint_dir = str(tp.parent) if not args.checkpoint else cfg.checkpoint_dir
            cfg.tokenizer_name = tp.name
        else:
            cfg.tokenizer_name = tp.name

    try:
        import torch
    except ImportError:
        print(
            "[错误] 未安装 PyTorch。请先: pip install -r requirements.txt\n"
            "Mac 上若需 MPS，请参考 https://pytorch.org 选择对应安装命令。",
            file=sys.stderr,
        )
        return 1

    if cfg.device == "mps":
        _disable_mps_reduced_precision()

    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
        try:
            torch.set_num_interop_threads(max(1, min(args.cpu_threads, 2)))
        except RuntimeError:
            pass
        print(f"CPU 线程限制: {args.cpu_threads}")

    from model import GPT, GPTConfig

    torch.manual_seed(cfg.seed)

    print(f"设备: {cfg.device}")
    print(f"模式: {backend}")
    grad_accum = max(1, int(getattr(args, "grad_accum", 1) or 1))
    args.grad_accum = grad_accum
    print(
        f"有效: batch={cfg.batch_size}, grad_accum={grad_accum}, "
        f"eff_batch={cfg.batch_size * grad_accum}, block={cfg.block_size}, "
        f"max_iters={cfg.max_iters}, ckpt={cfg.checkpoint_path()}, "
        f"step_sleep={args.step_sleep}, "
        f"lr={cfg.learning_rate:g}, warmup={cfg.warmup_iters}, "
        f"min_lr={cfg.learning_rate if cfg.min_lr is None else cfg.min_lr:g}, "
        f"dropout={cfg.dropout}, adam_eps={cfg.adam_eps:g}, "
        f"decay_only_2d={cfg.decay_only_2d}, "
        f"loss_fuse_eps={args.loss_fuse_eps}, "
        f"loss_fuse_patience={args.loss_fuse_patience}, "
        f"loss_fuse_high={args.loss_fuse_high}, "
        f"loss_fuse_high_patience={args.loss_fuse_high_patience}, "
        f"ckpt_rotate={args.ckpt_rotate}, "
        f"probe={bool(args.probe)}, "
        f"adamw_foreach=False, "
        f"grad_fuse_norm={args.grad_fuse_norm}, "
        f"grad_fuse_steps={args.grad_fuse_steps}",
        flush=True,
    )
    if args.probe:
        print("探路模式：本轮不会写入任何 checkpoint 或 last_summary", flush=True)
    if "_rot_" in Path(cfg.checkpoint_path()).stem or Path(cfg.checkpoint_path()).name.endswith("_rot.pt"):
        print(
            "[警告] --checkpoint 看起来像旋转备份名；请用主 ckpt 作写入目标，"
            "rot 仅作回滚读入。",
            flush=True,
        )

    start_iter = 0
    ckpt = None
    ckpt_path = Path(cfg.checkpoint_path())
    tok_path = Path(args.tokenizer or cfg.tokenizer_path())

    if args.resume:
        if not ckpt_path.is_file():
            print(f"[错误] 找不到 checkpoint: {ckpt_path}", file=sys.stderr)
            return 1
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # 优先用 checkpoint 里记录的后端 / hf id
        saved_tc = ckpt.get("train_config") or {}
        backend = saved_tc.get("tokenizer_backend", backend)
        if backend == "hf":
            hf_id = saved_tc.get("hf_tokenizer_id", cfg.hf_tokenizer_id)
            cache = saved_tc.get("hf_cache_dir", cfg.hf_cache_dir)
            tokenizer = HFTokenizer.load(hf_id, cache_dir=cache)
            cfg.hf_tokenizer_id = hf_id
        elif backend == "char":
            if not tok_path.is_file():
                print(f"[错误] 找不到 tokenizer: {tok_path}", file=sys.stderr)
                return 1
            tokenizer = CharTokenizer.load(str(tok_path))
        else:
            if not tok_path.is_file():
                print(f"[错误] 找不到 tokenizer: {tok_path}", file=sys.stderr)
                return 1
            tokenizer = BPETokenizer.load(str(tok_path))
        cfg.vocab_size = tokenizer.vocab_size
        saved_mc = ckpt.get("model_config") or {}
        for k in ("block_size", "n_layer", "n_head", "n_embd", "dropout", "bias"):
            if k in saved_mc:
                setattr(cfg, k, saved_mc[k])
        start_iter = int(ckpt.get("iter", 0))
        print(f"续训：加载 {ckpt_path}（已完成 iter={start_iter}）")
    else:
        if backend == "char":
            try:
                text = _load_char_text(cfg)
            except (FileNotFoundError, ValueError) as e:
                print(f"[错误] {e}", file=sys.stderr)
                return 1
            print(f"语料: {cfg.data_path}（{len(text)} 字符）")
            tokenizer = CharTokenizer.build_from_text(text)
            cfg.vocab_size = tokenizer.vocab_size
        elif backend == "bpe":
            if not tok_path.is_file():
                print(
                    f"[错误] 找不到 BPE 词表: {tok_path}\n"
                    "请先: python scripts/train_bpe.py",
                    file=sys.stderr,
                )
                return 1
            tokenizer = BPETokenizer.load(str(tok_path))
            cfg.vocab_size = tokenizer.vocab_size
            print(f"词表: {tok_path}")
        else:
            print(f"加载 HF tokenizer: {cfg.hf_tokenizer_id} …", flush=True)
            tokenizer = HFTokenizer.load(cfg.hf_tokenizer_id, cache_dir=cfg.hf_cache_dir)
            cfg.vocab_size = tokenizer.vocab_size
            print(f"词表: {cfg.hf_tokenizer_id}")

    print(f"词表大小: {cfg.vocab_size}")

    try:
        if backend == "char":
            text = _load_char_text(cfg)
            print(f"语料: {cfg.data_path}（{len(text)} 字符）")
            train_ds, val_ds = build_datasets(
                text, tokenizer, cfg.block_size, train_ratio=cfg.train_ratio
            )
        else:
            train_ds, val_ds = load_memmap_datasets(
                cfg.train_bin, cfg.val_bin, cfg.block_size
            )
            print(
                f"memmap: train={cfg.train_bin} ({len(train_ds._mmap):,} tok, dtype={train_ds.dtype}) | "
                f"val={cfg.val_bin} ({len(val_ds._mmap):,} tok)"
            )
    except (FileNotFoundError, ValueError) as e:
        print(f"[错误] {e}", file=sys.stderr)
        if backend != "char":
            print(
                "请先: python scripts/encode_corpus.py"
                + (" --bpe" if backend == "bpe" else ""),
                file=sys.stderr,
            )
        return 1

    gpt_cfg = GPTConfig(
        vocab_size=cfg.vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
        bias=cfg.bias,
    )
    model = GPT(gpt_cfg).to(cfg.device)
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        print("已加载模型权重")
    print(f"参数量: {model.num_params():,}")

    if cfg.decay_only_2d:
        # mid200：只对 2D 权重做 weight decay；bias / LayerNorm / 其它 1D 参数不衰减。
        # mid100 默认 decay_only_2d=False，仍走下方单组 AdamW，保证旧 ckpt 能续训。
        decay, no_decay = [], []
        for _name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        param_groups = [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        print(
            f"AdamW: decay {sum(p.numel() for p in decay):,} 个 2D 参数 "
            f"(wd={cfg.weight_decay}), no_decay {sum(p.numel() for p in no_decay):,} "
            f"个 1D 参数, eps={cfg.adam_eps:g}",
            flush=True,
        )
        optimizer = _make_adamw(
            param_groups,
            lr=cfg.learning_rate,
            eps=cfg.adam_eps,
        )
    else:
        optimizer = _make_adamw(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            eps=cfg.adam_eps,
        )
    if ckpt is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        moved = _optimizer_state_to_device(optimizer, cfg.device)
        _sync_device(cfg.device)
        print(
            f"已加载优化器状态，并将 {moved} 个张量搬到 {cfg.device}",
            flush=True,
        )

    @torch.no_grad()
    def estimate_loss() -> dict:
        model.eval()
        out = {}
        for split, ds in (("train", train_ds), ("val", val_ds)):
            losses = torch.zeros(cfg.eval_iters)
            for i in range(cfg.eval_iters):
                xb, yb = ds.get_batch(cfg.batch_size, cfg.device)
                _, loss = model(xb, yb)
                losses[i] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    if backend == "char":
        tokenizer.save(cfg.tokenizer_path())
    elif backend == "bpe" and not Path(cfg.tokenizer_path()).is_file():
        tokenizer.save(cfg.tokenizer_path())
    elif backend == "hf":
        # 本地再存一份，方便离线 generate
        local_tok_dir = Path(cfg.hf_cache_dir) / "Qwen2.5-1.5B-tokenizer"
        tokenizer.save(local_tok_dir)

    summary_name = {
        "hf": "last_summary_hf.json",
        "bpe": "last_summary_bpe.json",
        "char": "last_summary_char.json",
    }[backend]

    loss_fuse_eps = float(args.loss_fuse_eps)
    loss_fuse_patience = max(0, int(args.loss_fuse_patience))
    loss_fuse_high = float(args.loss_fuse_high)
    loss_fuse_high_patience = max(0, int(args.loss_fuse_high_patience))
    ckpt_rotate = max(0, int(args.ckpt_rotate))
    bad_loss_streak = 0
    high_loss_step_streak = 0
    high_loss_eval_streak = 0
    if getattr(args, "until_iter", None) is not None:
        target = int(args.until_iter)
        remaining = target - start_iter
        if remaining <= 0:
            print(
                f"已达到或超过目标 iter={target}（当前 {start_iter}），无需再训。",
                flush=True,
            )
            return EXIT_OK
        cfg.max_iters = remaining
        print(
            f"目标 iter={target}：已完成 {start_iter}，本轮再跑 {remaining} 步",
            flush=True,
        )

    schedule_end = start_iter + cfg.max_iters
    best_val = float("inf")
    if ckpt is not None and isinstance(ckpt.get("metrics"), dict):
        try:
            best_val = float(ckpt["metrics"].get("val", "inf"))
        except (TypeError, ValueError):
            best_val = float("inf")

    model.train()
    for step in range(1, cfg.max_iters + 1):
        it = start_iter + step
        lr = get_lr(
            it,
            max_iters=schedule_end,
            learning_rate=cfg.learning_rate,
            warmup_iters=cfg.warmup_iters,
            min_lr=cfg.min_lr,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _micro in range(grad_accum):
            xb, yb = train_ds.get_batch(cfg.batch_size, cfg.device)
            _, loss = model(xb, yb)
            (loss / grad_accum).backward()
            loss_accum += float(loss.item())
        if cfg.grad_clip > 0:
            # 返回值是裁剪前的总范数；foreach=False，避开 MPS 融合核
            grad_norm = _clip_grad_norm(model.parameters(), cfg.grad_clip)
        else:
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += float(p.grad.detach().float().norm(2).item() ** 2)
            grad_norm = grad_norm ** 0.5
        # 续训后若梯度范数异常，先不要 step，以免把内存里的好权重冲掉
        grad_fuse_norm = float(args.grad_fuse_norm)
        grad_fuse_steps = max(0, int(args.grad_fuse_steps))
        if (
            ckpt is not None
            and grad_fuse_steps > 0
            and grad_fuse_norm > 0
            and step <= grad_fuse_steps
            and (not math.isfinite(grad_norm) or grad_norm > grad_fuse_norm)
        ):
            print(
                f"[熔断] 续训后 iter {it} 的梯度范数 {grad_norm} "
                f"{'>' if math.isfinite(grad_norm) else '非有限，阈值'} {grad_fuse_norm:g}。"
                f"中止且不覆盖主 checkpoint: {cfg.checkpoint_path()}"
                f"（退出码 {EXIT_ERROR}）",
                file=sys.stderr,
                flush=True,
            )
            return EXIT_ERROR
        optimizer.step()
        loss_val = loss_accum / grad_accum

        # MPS hang / corrupted compute often shows up as NaN/Inf or ~0 loss.
        if loss_fuse_patience > 0 and _loss_is_bad(loss_val, loss_fuse_eps):
            bad_loss_streak += 1
            print(
                f"[熔断预警] iter {it} | loss={loss_val!r} "
                f"(坏步 {bad_loss_streak}/{loss_fuse_patience})",
                flush=True,
            )
            if bad_loss_streak >= loss_fuse_patience:
                print(
                    f"[熔断] 连续 {bad_loss_streak} 步 loss 为 NaN/Inf 或 |loss|<{loss_fuse_eps}。"
                    f"疑似 MPS/GPU 已坏，中止且不覆盖主 checkpoint: {cfg.checkpoint_path()}"
                    f"（退出码 {EXIT_FUSE_NUMERIC}）",
                    file=sys.stderr,
                    flush=True,
                )
                return EXIT_FUSE_NUMERIC
        else:
            bad_loss_streak = 0

        if getattr(args, "step_sleep", 0) and args.step_sleep > 0:
            # step_sleep 是上界：每步 uniform(0, step_sleep)；0 表示不睡
            time.sleep(random.uniform(0.0, float(args.step_sleep)))

        log_this = step % 50 == 0 or step == 1 or bool(args.probe)
        if log_this:
            print(
                f"iter {it:5d} | loss {loss_val:.4f} | lr {lr:.2e} | "
                f"grad_norm {grad_norm:.4f} (clip {cfg.grad_clip:g})",
                flush=True,
            )
            # 高损失熔断按「日志点」计数，避免单步毛刺立刻停训；约 3 次日志 ≈ 150 步。
            if loss_fuse_high > 0 and loss_fuse_high_patience > 0 and loss_val > loss_fuse_high:
                high_loss_step_streak += 1
                print(
                    f"[熔断预警] iter {it} | loss={loss_val:.4f} > {loss_fuse_high:g} "
                    f"(高损失步进 {high_loss_step_streak}/{loss_fuse_high_patience})",
                    flush=True,
                )
                if high_loss_step_streak >= loss_fuse_high_patience:
                    print(
                        f"[熔断] 连续 {high_loss_step_streak} 次日志 loss > {loss_fuse_high:g}。"
                        f"中止且不覆盖主 checkpoint: {cfg.checkpoint_path()}"
                        f"（退出码 {EXIT_FUSE_HIGH}）",
                        file=sys.stderr,
                        flush=True,
                    )
                    return EXIT_FUSE_HIGH
            else:
                high_loss_step_streak = 0

        if step % cfg.eval_interval == 0 or step == cfg.max_iters:
            metrics = estimate_loss()
            print(
                f"iter {it:5d} | train {metrics['train']:.4f} | val {metrics['val']:.4f} | "
                f"lr {lr:.2e}",
                flush=True,
            )
            if args.probe:
                print("探路模式：评估已完成，不写入任何 checkpoint", flush=True)
                continue
            # If eval also looks dead, abort before writing main ckpt.
            eval_bad = any(
                _loss_is_bad(float(metrics[k]), loss_fuse_eps) for k in ("train", "val")
            )
            if loss_fuse_patience > 0 and eval_bad:
                print(
                    f"[熔断] eval 指标异常 train={metrics['train']!r} val={metrics['val']!r}；"
                    f"跳过保存，不覆盖主 checkpoint: {cfg.checkpoint_path()}"
                    f"（退出码 {EXIT_FUSE_NUMERIC}）",
                    file=sys.stderr,
                    flush=True,
                )
                return EXIT_FUSE_NUMERIC
            eval_high = (
                loss_fuse_high > 0
                and loss_fuse_high_patience > 0
                and (
                    float(metrics["train"]) > loss_fuse_high
                    or float(metrics["val"]) > loss_fuse_high
                )
            )
            if eval_high:
                high_loss_eval_streak += 1
                print(
                    f"[熔断预警] iter {it} | eval train={metrics['train']:.4f} "
                    f"val={metrics['val']:.4f} > {loss_fuse_high:g} "
                    f"(高损失评估 {high_loss_eval_streak}/{loss_fuse_high_patience})",
                    flush=True,
                )
                if high_loss_eval_streak >= loss_fuse_high_patience:
                    print(
                        f"[熔断] 连续 {high_loss_eval_streak} 次评估高于 {loss_fuse_high:g}；"
                        f"跳过保存，不覆盖主 checkpoint: {cfg.checkpoint_path()}"
                        f"（退出码 {EXIT_FUSE_HIGH}）",
                        file=sys.stderr,
                        flush=True,
                    )
                    return EXIT_FUSE_HIGH
                # 未达耐心：跳过本次主 ckpt，以免好权重被高损失覆盖
                print(
                    f"[熔断预警] 本次不写入主 checkpoint: {cfg.checkpoint_path()}",
                    flush=True,
                )
                continue
            high_loss_eval_streak = 0
            payload = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_config": gpt_cfg.__dict__,
                "train_config": cfg.to_dict(),
                "iter": it,
                "metrics": metrics,
                "tokenizer_name": cfg.tokenizer_name,
                "tokenizer_backend": backend,
                "hf_tokenizer_id": cfg.hf_tokenizer_id if backend == "hf" else None,
            }
            torch.save(payload, cfg.checkpoint_path())
            summary = {
                "iter": it,
                "metrics": metrics,
                "vocab_size": cfg.vocab_size,
                "block_size": cfg.block_size,
                "n_layer": cfg.n_layer,
                "n_head": cfg.n_head,
                "n_embd": cfg.n_embd,
                "device": cfg.device,
                "checkpoint": cfg.checkpoint_path(),
                "tokenizer": cfg.hf_tokenizer_id if backend == "hf" else cfg.tokenizer_path(),
                "mode": backend,
            }
            Path(cfg.checkpoint_dir, summary_name).write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            Path(cfg.checkpoint_dir, "last_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"已保存 checkpoint → {cfg.checkpoint_path()}", flush=True)
            _rotate_checkpoint(Path(cfg.checkpoint_path()), ckpt_rotate, it)
            if float(metrics["val"]) < best_val:
                best_val = float(metrics["val"])
                best_path = cfg.best_checkpoint_path()
                torch.save(payload, best_path)
                print(
                    f"已保存 best-val checkpoint → {best_path} (val={best_val:.4f})",
                    flush=True,
                )

    print("训练结束。可用: python generate.py")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
