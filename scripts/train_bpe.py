#!/usr/bin/env python3
"""从语料（单文件或多文件）训练 BPE 词表（不训练语言模型）。

大语料走 tokenizers 的 files= 流式训练，避免整文件读入内存。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def _limit_threads(n: int = 2) -> None:
    """限制训练占用的并行度（CPU 友好；慢一点没关系）。

    必须在 import tokenizers / 启动 Rayon 之前调用。
    """
    n = max(1, int(n))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        os.environ[key] = str(n)


# 默认先压到 2 线程，再解析 CLI（可被 --threads 覆盖后重新 import 前已生效）
# CLI 里若改 threads，需要在 import tokenizers 前再次设置。
def _early_threads_from_argv() -> int:
    n = 2
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--threads" and i + 1 < len(argv):
            try:
                n = int(argv[i + 1])
            except ValueError:
                pass
        elif a.startswith("--threads="):
            try:
                n = int(a.split("=", 1)[1])
            except ValueError:
                pass
    return n


_limit_threads(_early_threads_from_argv())

from tokenizers import Regex, Tokenizer, decoders, pre_tokenizers  # noqa: E402
from tokenizers.models import BPE  # noqa: E402
from tokenizers.trainers import BpeTrainer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from corpus import (  # noqa: E402
    add_corpus_cli_args,
    format_corpus_summary,
    resolve_from_args,
)

SPECIALS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]


def train_bpe(
    files: list[Path],
    out_path: Path,
    vocab_size: int = 32000,
) -> Tokenizer:
    """空白切分预分词 + BPE：中文连续串内部可合并成多字 token。"""
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    # 按空白切开，空白本身保留为 token；非空白串（中文/英文）内部做 BPE 合并
    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex(r"\s+"), behavior="isolated")
    # 直接拼接 token 字符串，避免 decode 时在汉字间插空格
    tokenizer.decoder = decoders.Fuse()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIALS,
        show_progress=True,
        min_frequency=2,
    )
    # files= 流式读多份语料，适合数百 MB～GB 级
    tokenizer.train(files=[str(p) for p in files], trainer=trainer)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))
    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BPE tokenizer from corpus")
    add_corpus_cli_args(parser)
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="限制 CPU 线程数（OMP/MKL/RAYON），默认 2",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "checkpoints" / "tokenizer_bpe.json",
        help="输出词表路径（勿覆盖字符版 tokenizer.json）",
    )
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="只打印将要使用的语料文件，不训练",
    )
    args = parser.parse_args()
    # 再次写入环境变量（若 Rayon 已启动则主要靠启动前的 early 设置）
    _limit_threads(args.threads)
    print(f"thread_limit: {args.threads}")
    print(
        "env:",
        {k: os.environ.get(k) for k in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS")},
    )

    try:
        files = resolve_from_args(args)
    except FileNotFoundError as e:
        raise SystemExit(f"语料错误: {e}") from e

    print(format_corpus_summary(files))
    if args.list_only:
        return

    print(f"vocab_size target: {args.vocab_size}")
    print(f"specials: {SPECIALS}")
    t0 = time.perf_counter()
    tok = train_bpe(files, args.out, vocab_size=args.vocab_size)
    elapsed = time.perf_counter() - t0
    print(f"saved: {args.out} ({args.out.stat().st_size} bytes)")
    print(f"actual vocab_size: {tok.get_vocab_size()}")
    print(f"train_seconds: {elapsed:.1f}")


if __name__ == "__main__":
    main()
