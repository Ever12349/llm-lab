#!/usr/bin/env python3
"""把语料（单文件或多文件）编码成 memmap：data/train.bin + data/val.bin。

默认用 Hugging Face 词表（Qwen/Qwen2.5-1.5B）；也可用 --bpe 走本地自训 BPE。
流式分块编码，避免把整份大语料一次性读进 Python 字符串再 encode。
多文件时按排序后的顺序依次写入同一条 token 流。
大词表（id ≥ 65535）自动用 uint32；旧 BPE（≤32k）仍用 uint16。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import TrainConfig  # noqa: E402
from corpus import (  # noqa: E402
    add_corpus_cli_args,
    format_corpus_summary,
    resolve_from_args,
)
from tokenizer import BPETokenizer, HFTokenizer  # noqa: E402


def pick_dtype(vocab_size: int) -> np.dtype:
    """词表过大时 uint16 装不下 token id。"""
    if vocab_size > 65535:
        return np.dtype(np.uint32)
    return np.dtype(np.uint16)


def encode_files_streaming(
    files: list[Path],
    tokenizer,
    out_all: Path,
    *,
    dtype: np.dtype,
    chunk_chars: int = 2_000_000,
) -> int:
    """按文件顺序、字符块流式 encode，写入 all.bin，返回 token 总数。"""
    tmp = out_all.with_suffix(".tmp.bin")
    if tmp.exists():
        tmp.unlink()

    n_tokens = 0
    n_chars = 0
    t0 = time.perf_counter()
    buf: list[np.ndarray] = []
    buf_tokens = 0
    flush_every = 5_000_000  # 每攒够约 500 万 token 刷盘
    max_id_allowed = int(np.iinfo(dtype).max)

    def flush() -> None:
        nonlocal buf, buf_tokens
        if not buf:
            return
        arr = np.concatenate(buf)
        with open(tmp, "ab") as f:
            arr.tofile(f)
        buf = []
        buf_tokens = 0

    total_files = len(files)
    for fi, corpus in enumerate(files, 1):
        file_size = corpus.stat().st_size
        try:
            rel = corpus.relative_to(ROOT)
        except ValueError:
            rel = corpus
        print(f"[{fi}/{total_files}] {rel} ({file_size:,} bytes)")
        file_tokens = 0
        file_chars = 0
        with corpus.open("r", encoding="utf-8", errors="strict") as f:
            while True:
                chunk = f.read(chunk_chars)
                if not chunk:
                    break
                file_chars += len(chunk)
                n_chars += len(chunk)
                ids = tokenizer.encode(chunk)
                if ids:
                    arr = np.asarray(ids, dtype=dtype)
                    if int(arr.max()) > max_id_allowed:
                        raise ValueError(
                            f"token id 超出 {dtype} 上限 {max_id_allowed}，请检查词表。"
                        )
                    buf.append(arr)
                    buf_tokens += len(arr)
                    n_tokens += len(arr)
                    file_tokens += len(arr)
                if buf_tokens >= flush_every:
                    flush()
                pos = f.tell()
                pct = 100.0 * pos / file_size if file_size else 100.0
                print(
                    f"\r    encode {pct:5.1f}% | file chars {file_chars:,} | "
                    f"file tok {file_tokens:,} | all tok {n_tokens:,}",
                    end="",
                    flush=True,
                )
        flush()
        print()  # 结束本文件的 \r 行
        elapsed = time.perf_counter() - t0
        print(
            f"    done: chars {file_chars:,} → tokens {file_tokens:,} | "
            f"累计 {n_tokens:,} tok | {elapsed:.0f}s"
        )

    if not tmp.exists():
        tmp.write_bytes(b"")
    tmp.rename(out_all)
    print(f"all tokens: {n_tokens:,} | chars: {n_chars:,} | dtype={dtype}")
    return n_tokens


def split_bins(
    all_bin: Path,
    train_bin: Path,
    val_bin: Path,
    train_ratio: float,
    dtype: np.dtype,
) -> tuple[int, int]:
    """把 all.bin 按比例切成 train / val（token 流切分，不打乱）。"""
    data = np.memmap(all_bin, dtype=dtype, mode="r")
    n = len(data)
    if n == 0:
        raise ValueError("编码结果为空，请检查语料是否非空。")
    n_train = int(n * train_ratio)
    n_train = max(n_train, 1)
    n_train = min(n_train, n - 1) if n > 1 else n
    train_bin.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(data[:n_train]).tofile(train_bin)
    np.asarray(data[n_train:]).tofile(val_bin)
    return n_train, n - n_train


def main() -> int:
    cfg = TrainConfig()
    p = argparse.ArgumentParser(
        description="Encode corpus (one or many .txt) to train.bin / val.bin"
    )
    add_corpus_cli_args(p)
    p.add_argument(
        "--hf-tokenizer",
        default=None,
        help=f"Hugging Face tokenizer id（默认 {cfg.hf_tokenizer_id}）",
    )
    p.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=None,
        help=f"HF 缓存目录（默认 {cfg.hf_cache_dir}）",
    )
    p.add_argument(
        "--bpe",
        action="store_true",
        help="使用本地自训 BPE（checkpoints/tokenizer_bpe.json），非默认",
    )
    p.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="本地 BPE JSON 路径（仅 --bpe；默认 checkpoints/tokenizer_bpe.json）",
    )
    p.add_argument("--train-bin", type=Path, default=ROOT / "data" / "train.bin")
    p.add_argument("--val-bin", type=Path, default=ROOT / "data" / "val.bin")
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument("--chunk-chars", type=int, default=2_000_000)
    p.add_argument(
        "--keep-all",
        action="store_true",
        help="保留 data/tokens_all.bin（默认切分后删除）",
    )
    p.add_argument(
        "--list-only",
        action="store_true",
        help="只打印将要使用的语料文件，不编码",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="只对前 1 个语料文件、前 chunk 做快速编码冒烟（不覆盖正式 train/val）",
    )
    args = p.parse_args()

    try:
        files = resolve_from_args(args)
    except FileNotFoundError as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1

    print(format_corpus_summary(files))
    if args.list_only:
        return 0

    if args.bpe:
        tok_path = args.tokenizer or (ROOT / "checkpoints" / "tokenizer_bpe.json")
        if not Path(tok_path).is_file():
            print(f"[错误] BPE 词表不存在: {tok_path}", file=sys.stderr)
            return 1
        tok = BPETokenizer.load(tok_path)
        tok_label = str(tok_path)
        backend = "bpe"
    else:
        hf_id = args.hf_tokenizer or cfg.hf_tokenizer_id
        cache = args.hf_cache_dir or (ROOT / cfg.hf_cache_dir)
        print(f"加载 HF tokenizer: {hf_id}（cache={cache}）…", flush=True)
        tok = HFTokenizer.load(hf_id, cache_dir=cache)
        # 额外落到项目目录，方便离线
        local_dir = Path(cache) / "Qwen2.5-1.5B-tokenizer"
        if "Qwen2.5-1.5B" in hf_id.replace("/", "-") or hf_id.endswith("Qwen2.5-1.5B"):
            tok.save(local_dir)
        else:
            safe = hf_id.replace("/", "__")
            tok.save(Path(cache) / f"{safe}-tokenizer")
        tok_label = hf_id
        backend = "hf"

    dtype = pick_dtype(tok.vocab_size)
    print(f"tokenizer ({backend}): {tok_label} | vocab_size={tok.vocab_size} | dtype={dtype}")

    if args.smoke:
        smoke_files = files[:1]
        all_bin = ROOT / "data" / "tokens_smoke.bin"
        n_tokens = encode_files_streaming(
            smoke_files,
            tok,
            all_bin,
            dtype=dtype,
            chunk_chars=min(args.chunk_chars, 50_000),
        )
        print(f"smoke OK: {n_tokens:,} tokens → {all_bin}")
        if all_bin.exists():
            all_bin.unlink()
        return 0

    all_bin = ROOT / "data" / "tokens_all.bin"
    t0 = time.perf_counter()
    n_tokens = encode_files_streaming(
        files, tok, all_bin, dtype=dtype, chunk_chars=args.chunk_chars
    )
    n_train, n_val = split_bins(
        all_bin, args.train_bin, args.val_bin, args.train_ratio, dtype
    )
    elapsed = time.perf_counter() - t0

    meta = {
        "corpus_files": [str(f) for f in files],
        "n_files": len(files),
        "tokenizer": tok_label,
        "tokenizer_backend": backend,
        "vocab_size": tok.vocab_size,
        "n_tokens": n_tokens,
        "n_train": n_train,
        "n_val": n_val,
        "train_ratio": args.train_ratio,
        "dtype": str(dtype),
        "train_bin": str(args.train_bin),
        "val_bin": str(args.val_bin),
        "encode_seconds": round(elapsed, 1),
    }
    meta_path = ROOT / "data" / "tokens_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"train.bin: {args.train_bin} ({n_train:,} tokens, {args.train_bin.stat().st_size:,} bytes)")
    print(f"val.bin:   {args.val_bin} ({n_val:,} tokens, {args.val_bin.stat().st_size:,} bytes)")
    print(f"meta: {meta_path}")
    print(f"elapsed: {elapsed:.1f}s")

    if not args.keep_all and all_bin.exists():
        all_bin.unlink()
        print("removed tokens_all.bin")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
