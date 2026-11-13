#!/usr/bin/env python3
"""用 SentencePiece 在 spm_corpus_v1 上训练 4 份词表：32k/50k × BPE/Unigram。

先把过长行按字节切成不超过 8192 的句子（SentencePiece 默认会丢弃超过
max_sentence_length 的行；本语料中文约有 1.9 万行超过 4192 字节），再训练。
只训词表，不训练语言模型。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_DIR = ROOT / "data" / "spm_corpus_v1"
DEFAULT_OUT_DIR = ROOT / "data" / "spm_vocabs_v1"
MAX_SENTENCE_BYTES = 8192
VOCAB_SPECS = (
    (32000, "bpe"),
    (32000, "unigram"),
    (50000, "bpe"),
    (50000, "unigram"),
)
SAMPLE_TEXTS = {
    "zh": "数学是研究数量、结构以及空间等概念及其变化的一门学科。",
    "en": "Anarchism is a political philosophy and movement that is skeptical of authority.",
    "ja": "アンパサンドは、並立助詞「…と…」を意味する記号である。",
}


def _utf8_cut(buf: bytes, limit: int) -> int:
    """返回不超过 limit 的合法 UTF-8 切分点。"""
    if limit >= len(buf):
        return len(buf)
    i = limit
    while i > 0 and (buf[i] & 0xC0) == 0x80:
        i -= 1
    return i if i > 0 else limit


def _preferred_cut(buf: bytes, limit: int) -> int:
    end = _utf8_cut(buf, limit)
    if end >= len(buf):
        return end
    window = buf[max(0, end - 1024) : end]
    for sep in (b"\n", "。".encode(), "！".encode(), "？".encode(), b". ", b"? ", b"! ",
                "．".encode(), "、".encode(), b"; ", b", ", "，".encode(), b" "):
        pos = window.rfind(sep)
        if pos >= 0:
            cut = max(0, end - 1024) + pos + len(sep)
            cut = _utf8_cut(buf, cut)
            if cut > 0:
                return cut
    return end


def iter_wrapped_lines(path: Path, limit: int = MAX_SENTENCE_BYTES):
    with path.open("rb") as fh:
        for raw in fh:
            line = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n").strip()
            if not line:
                continue
            while line:
                if len(line) <= limit:
                    yield line
                    break
                cut = _preferred_cut(line, limit)
                if cut <= 0:
                    cut = _utf8_cut(line, limit)
                chunk = line[:cut].strip()
                if chunk:
                    yield chunk
                line = line[cut:].lstrip()


def build_train_lines(inputs: list[Path], dest: Path, limit: int = MAX_SENTENCE_BYTES) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    meta_path = dest.with_suffix(dest.suffix + ".meta.json")
    src_stat = {str(p): {"size": p.stat().st_size, "mtime": int(p.stat().st_mtime)} for p in inputs}
    if dest.exists() and meta_path.exists():
        try:
            old = json.loads(meta_path.read_text(encoding="utf-8"))
            if old.get("sources") == src_stat and old.get("max_sentence_bytes") == limit and old.get("status") == "ok":
                print(f"复用已有训练行文件: {dest} ({dest.stat().st_size:,} bytes, lines={old.get('lines'):,})", flush=True)
                return old
        except (OSError, json.JSONDecodeError):
            pass

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    n_lines = 0
    n_src = 0
    n_long = 0
    t0 = time.perf_counter()
    with tmp.open("wb") as out:
        for src in inputs:
            print(f"  切分 {src.name} …", flush=True)
            with src.open("rb") as fh:
                for raw in fh:
                    line = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n").strip()
                    if not line:
                        continue
                    n_src += 1
                    if len(line) > limit:
                        n_long += 1
                    remain = line
                    while remain:
                        if len(remain) <= limit:
                            out.write(remain)
                            out.write(b"\n")
                            n_lines += 1
                            break
                        cut = _preferred_cut(remain, limit)
                        if cut <= 0:
                            cut = _utf8_cut(remain, limit)
                        chunk = remain[:cut].strip()
                        if chunk:
                            out.write(chunk)
                            out.write(b"\n")
                            n_lines += 1
                        remain = remain[cut:].lstrip()
    tmp.replace(dest)
    n_wraps = n_lines - n_src
    meta = {
        "status": "ok",
        "path": str(dest),
        "bytes": dest.stat().st_size,
        "lines": n_lines,
        "source_nonblank_lines": n_src,
        "source_long_lines": n_long,
        "extra_wrapped_lines": n_wraps,
        "max_sentence_bytes": limit,
        "sources": src_stat,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"训练行文件: {dest}  lines={n_lines:,}  long_src={n_long:,}  "
        f"extra={n_wraps:,}  {meta['seconds']}s",
        flush=True,
    )
    return meta


def _probe(model_path: Path) -> dict:
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    samples = {}
    for lang, text in SAMPLE_TEXTS.items():
        pieces = sp.encode(text, out_type=str)
        ids = sp.encode(text, out_type=int)
        decoded = sp.decode(ids)
        samples[lang] = {
            "text": text,
            "n_pieces": len(pieces),
            "pieces": pieces,
            "ids": ids,
            "decode_ok": decoded.replace(" ", "") == text.replace(" ", "") or decoded == text,
            "decoded": decoded,
        }
    return {
        "vocab_size": int(sp.vocab_size()),
        "unk_id": int(sp.unk_id()),
        "bos_id": int(sp.bos_id()),
        "eos_id": int(sp.eos_id()),
        "pad_id": int(sp.pad_id()),
        "samples": samples,
    }


def train_one(
    train_file: Path,
    out_dir: Path,
    vocab_size: int,
    model_type: str,
    threads: int,
    character_coverage: float,
) -> dict:
    import sentencepiece as spm

    prefix = out_dir / f"spm_v1_{vocab_size // 1000}k_{model_type}"
    model_path = Path(str(prefix) + ".model")
    vocab_path = Path(str(prefix) + ".vocab")
    print(f"\n=== 训练 {prefix.name}  vocab_size={vocab_size} type={model_type} ===", flush=True)
    t0 = time.perf_counter()
    kwargs = dict(
        input=str(train_file),
        model_prefix=str(prefix),
        vocab_size=int(vocab_size),
        model_type=model_type,
        character_coverage=float(character_coverage),
        byte_fallback=True,
        split_digits=True,
        train_extremely_large_corpus=True,
        shuffle_input_sentence=True,
        input_sentence_size=0,
        max_sentence_length=MAX_SENTENCE_BYTES,
        num_threads=int(threads),
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        pad_piece="<pad>",
        num_sub_iterations=2,
    )
    try:
        spm.SentencePieceTrainer.train(**kwargs)
    except RuntimeError as exc:
        # 32k 在 CJK + byte_fallback 下可能触达 hard_vocab_limit
        if "Vocabulary size" in str(exc) or "hard_vocab_limit" in str(exc).lower():
            print(f"  首次失败（{exc}），改用 hard_vocab_limit=false 重试", flush=True)
            kwargs["hard_vocab_limit"] = False
            spm.SentencePieceTrainer.train(**kwargs)
        else:
            raise
    elapsed = round(time.perf_counter() - t0, 1)
    if not model_path.exists():
        raise RuntimeError(f"训练结束但未找到 {model_path}")
    probe = _probe(model_path)
    print(
        f"完成 {prefix.name}: vocab={probe['vocab_size']}  {elapsed}s  "
        f"sample pieces zh/en/ja="
        f"{probe['samples']['zh']['n_pieces']}/"
        f"{probe['samples']['en']['n_pieces']}/"
        f"{probe['samples']['ja']['n_pieces']}",
        flush=True,
    )
    return {
        "name": prefix.name,
        "model_type": model_type,
        "requested_vocab_size": vocab_size,
        "model_path": str(model_path),
        "vocab_path": str(vocab_path),
        "bytes_model": model_path.stat().st_size,
        "bytes_vocab": vocab_path.stat().st_size if vocab_path.exists() else 0,
        "seconds": elapsed,
        "hard_vocab_limit": kwargs.get("hard_vocab_limit", True),
        "character_coverage": character_coverage,
        **probe,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="训练 4 份 SentencePiece 词表")
    p.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--threads", type=int, default=8, help="SentencePiece 训练线程数")
    p.add_argument("--character-coverage", type=float, default=0.9995)
    p.add_argument("--only", type=str, default="", help="只训指定项，逗号分隔，如 32k_bpe,50k_unigram")
    p.add_argument("--skip-lines", action="store_true", help="假定训练行文件已存在")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    corpus_dir = args.corpus_dir
    inputs = [corpus_dir / "zh.txt", corpus_dir / "en.txt", corpus_dir / "ja.txt"]
    missing = [p for p in inputs if not p.exists()]
    if missing:
        print("缺少语料: " + ", ".join(str(p) for p in missing), file=sys.stderr)
        return 2
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=== train_spm_vocabs_v1 ===", flush=True)
    print(f"corpus={corpus_dir}", flush=True)
    print(f"out_dir={out_dir}", flush=True)
    print(f"threads={args.threads} coverage={args.character_coverage}", flush=True)

    train_lines = corpus_dir / "train_lines.txt"
    t0 = time.perf_counter()
    if args.skip_lines:
        if not train_lines.exists():
            print(f"缺少 {train_lines}", file=sys.stderr)
            return 2
        lines_meta = {"path": str(train_lines), "bytes": train_lines.stat().st_size, "status": "skipped_build"}
    else:
        lines_meta = build_train_lines(inputs, train_lines)

    wanted = {(vs, mt) for vs, mt in VOCAB_SPECS}
    if args.only.strip():
        wanted = set()
        for item in args.only.split(","):
            item = item.strip().lower()
            if not item:
                continue
            size_s, typ = item.split("_", 1)
            wanted.add((int(size_s.replace("k", "000")), typ))

    results = []
    errors = []
    for vocab_size, model_type in VOCAB_SPECS:
        if (vocab_size, model_type) not in wanted:
            continue
        try:
            results.append(
                train_one(
                    train_lines,
                    out_dir,
                    vocab_size,
                    model_type,
                    threads=args.threads,
                    character_coverage=args.character_coverage,
                )
            )
        except Exception as exc:
            print(f"[错误] {vocab_size} {model_type}: {exc}", file=sys.stderr, flush=True)
            errors.append({"vocab_size": vocab_size, "model_type": model_type, "error": str(exc)})

    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "corpus_dir": str(corpus_dir),
        "out_dir": str(out_dir),
        "train_lines": lines_meta,
        "settings": {
            "character_coverage": args.character_coverage,
            "byte_fallback": True,
            "split_digits": True,
            "train_extremely_large_corpus": True,
            "max_sentence_length": MAX_SENTENCE_BYTES,
            "threads": args.threads,
            "special_tokens": {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3},
        },
        "models": results,
        "errors": errors,
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
        "note": "仅词表；尚未接入模型训练。",
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== 汇总 ===", flush=True)
    for item in results:
        print(
            f"{item['name']}: vocab={item['vocab_size']}  {item['seconds']}s  → {item['model_path']}",
            flush=True,
        )
    print(f"manifest: {manifest_path}", flush=True)
    print(f"elapsed: {manifest['elapsed_seconds']}s", flush=True)
    return 0 if not errors else 2


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(code or 0))
