#!/usr/bin/env python3
"""准备 SentencePiece 用的中英日各约 1GiB 高质量语料。

只写 data/spm_corpus_v1/，不训练词表，不碰 mid200 / HQ v2 bin。
各语言拆成 parts/，达到预算后可跳过，便于续跑。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3
MIB = 1024**2

DEFAULT_OUT = ROOT / "data" / "spm_corpus_v1"
ZH_WIKI_FILES = [
    ROOT / "data" / "seed_v2_zh_hq_v2" / "wiki_pretrain_zh.txt",
    ROOT / "data" / "seed_v2_zh_hq_v2" / "wiki_pretrain_zh_more.txt",
]
# 繁体近重复，不并入（与 wiki_pretrain_zh 同源）
ZH_WIKI_SKIP = [ROOT / "data" / "seed_v2_zh_hq_v2" / "wiki_zh_clean_base.txt"]

# 1GiB 内的来源预算（UTF-8 字节，不含分隔空行也可，按写入文件大小计）
BUDGETS = {
    "zh_wiki": 500 * MIB,
    "zh_fineweb_edu": GIB - 500 * MIB,
    "en_wiki": 300 * MIB,
    "en_fineweb_edu": 700 * MIB,
    "ja_wiki": 300 * MIB,
    "ja_aozora": 100 * MIB,
    "ja_fineweb_edu": GIB - 400 * MIB,
}

URL_PAT = re.compile(r"https?://|www\.[a-z0-9\-]+\.[a-z]{2,}", re.I)
HIRAGANA = re.compile(r"[\u3040-\u309f]")
KATAKANA = re.compile(r"[\u30a0-\u30ff]")
CJK = re.compile(r"[\u4e00-\u9fff]")
LATIN = re.compile(r"[A-Za-z]")


def _load_clean_zh():
    path = ROOT / "scripts" / "clean_zh_corpus.py"
    spec = importlib.util.spec_from_file_location("clean_zh_corpus", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _zh_to_simplified(text: str) -> str:
    from zhconv import convert

    return convert(text, "zh-cn")


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""
    lines = [ln.rstrip() for ln in text.splitlines()]
    out: list[str] = []
    blank = 0
    for ln in lines:
        if not ln.strip():
            blank += 1
            if blank <= 1:
                out.append("")
            continue
        blank = 0
        out.append(ln)
    return "\n".join(out).strip()


def _iter_local_docs(path: Path) -> Iterable[str]:
    buf: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip() == "" and buf:
                doc = _clean_text("".join(buf))
                buf = []
                if doc:
                    yield doc
            else:
                buf.append(line)
        if buf:
            doc = _clean_text("".join(buf))
            if doc:
                yield doc


def _doc_key(text: str) -> str:
    norm = " ".join(text.split())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def _byte_len(text: str) -> int:
    return len(text.encode("utf-8"))


def _load_stream(hf_id: str, config: Optional[str], split: str):
    from datasets import load_dataset

    kwargs = {"split": split, "streaming": True}
    tries = [
        dict(kwargs),
        dict(kwargs, trust_remote_code=True),
    ]
    last = None
    for kw in tries:
        try:
            if config:
                return load_dataset(hf_id, config, **kw)
            return load_dataset(hf_id, **kw)
        except Exception as e:
            last = e
            continue
    raise last  # type: ignore[misc]


def _looks_english(text: str, min_chars: int) -> bool:
    n = len(text)
    if n < min_chars:
        return False
    latin = len(LATIN.findall(text))
    if latin / n < 0.4:
        return False
    if len(URL_PAT.findall(text)) >= 6:
        return False
    return True


def _looks_japanese(text: str, min_chars: int) -> bool:
    n = len(text)
    if n < min_chars:
        return False
    h = len(HIRAGANA.findall(text))
    k = len(KATAKANA.findall(text))
    c = len(CJK.findall(text))
    if (h + k + c) / n < 0.3:
        return False
    # 纯汉字短串可能是中文；要求有假名，或足够长的日文维基（可几乎全是汉字专名页）
    if h + k == 0 and n < 400:
        return False
    if len(URL_PAT.findall(text)) >= 6:
        return False
    return True


def _write_docs(
    out_path: Path,
    docs: Iterable[str],
    byte_budget: int,
    *,
    seen: set,
    min_chars: int,
    accept: Optional[Callable[[str], bool]] = None,
    label: str = "",
) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".partial")
    started = time.perf_counter()
    docs_kept = 0
    docs_seen = 0
    docs_dup = 0
    docs_drop = 0
    nbytes = 0
    with tmp.open("w", encoding="utf-8") as f:
        for raw in docs:
            docs_seen += 1
            text = _clean_text(raw)
            if len(text) < min_chars:
                docs_drop += 1
                continue
            if accept is not None and not accept(text):
                docs_drop += 1
                continue
            key = _doc_key(text)
            if key in seen:
                docs_dup += 1
                continue
            extra = _byte_len(text) + 2
            if nbytes + extra > byte_budget and docs_kept > 0:
                break
            if extra > byte_budget and docs_kept == 0:
                # 单篇过大：截到预算附近的换行处
                raw_bytes = text.encode("utf-8")[: byte_budget]
                text = raw_bytes.decode("utf-8", errors="ignore").rsplit("\n", 1)[0]
                if len(text) < min_chars:
                    break
                extra = _byte_len(text) + 2
            seen.add(key)
            f.write(text)
            f.write("\n\n")
            nbytes += extra
            docs_kept += 1
            if docs_kept % 500 == 0 or nbytes >= byte_budget:
                elapsed = time.perf_counter() - started
                rate = nbytes / elapsed if elapsed else 0
                print(
                    f"\r   {label}: kept={docs_kept:,} bytes={nbytes:,}/"
                    f"{byte_budget:,} ({100 * nbytes / byte_budget:5.1f}%) "
                    f"drop={docs_drop:,} dup={docs_dup:,} {rate/MIB:,.2f} MiB/s",
                    end="",
                    flush=True,
                )
            if nbytes >= byte_budget:
                break
    print(flush=True)
    tmp.replace(out_path)
    return {
        "path": str(out_path),
        "docs_seen": docs_seen,
        "docs_kept": docs_kept,
        "docs_dup": docs_dup,
        "docs_drop": docs_drop,
        "bytes": out_path.stat().st_size,
        "seconds": round(time.perf_counter() - started, 1),
        "status": "ok",
    }


def _part_complete(path: Path, budget: int) -> bool:
    return path.is_file() and path.stat().st_size >= int(budget * 0.98)


def _iter_hf_text(
    hf_id: str,
    config: Optional[str],
    split: str,
    text_field: str,
    *,
    score_field: Optional[str] = None,
    min_score: Optional[float] = None,
    extra_ok: Optional[Callable[[dict], bool]] = None,
) -> Iterable[str]:
    ds = _load_stream(hf_id, config, split)
    for ex in ds:
        if extra_ok is not None and not extra_ok(ex):
            continue
        if score_field and min_score is not None:
            try:
                score = float(ex.get(score_field))
            except (TypeError, ValueError):
                continue
            if score < min_score:
                continue
        raw = ex.get(text_field) or ""
        if isinstance(raw, str) and raw.strip():
            yield raw


def prepare_zh_wiki(out_path: Path, budget: int, seen: set, min_chars: int) -> dict:
    missing = [p for p in ZH_WIKI_FILES if not p.is_file()]
    if missing:
        return {
            "path": str(out_path),
            "status": "error",
            "error": "missing " + ", ".join(str(p) for p in missing),
            "bytes": 0,
        }

    def docs():
        for p in ZH_WIKI_FILES:
            print(f"   读取本地维基 {p.name} ({p.stat().st_size / MIB:.1f} MiB)", flush=True)
            for doc in _iter_local_docs(p):
                yield _zh_to_simplified(doc)

    return _write_docs(
        out_path, docs(), budget, seen=seen, min_chars=min_chars, label="zh_wiki"
    )


def prepare_zh_fineweb(
    out_path: Path, budget: int, seen: set, min_chars: int, min_score: float
) -> dict:
    clean = _load_clean_zh()

    def accept(text: str) -> bool:
        text = _zh_to_simplified(text)
        return clean.classify(text, min_chars=min_chars, min_cjk=40) == "keep"

    def docs():
        for doc in _iter_hf_text(
            "opencsg/chinese-fineweb-edu-v2",
            None,
            "train",
            "text",
            score_field="score",
            min_score=min_score,
        ):
            yield _zh_to_simplified(doc)

    print(
        f"   流式下载 opencsg/chinese-fineweb-edu-v2 (score>={min_score})",
        flush=True,
    )
    return _write_docs(
        out_path,
        docs(),
        budget,
        seen=seen,
        min_chars=min_chars,
        accept=accept,
        label="zh_fineweb_edu",
    )


def prepare_en_wiki(out_path: Path, budget: int, seen: set, min_chars: int) -> dict:
    print("   流式下载 wikimedia/wikipedia 20231101.en", flush=True)
    return _write_docs(
        out_path,
        _iter_hf_text("wikimedia/wikipedia", "20231101.en", "train", "text"),
        budget,
        seen=seen,
        min_chars=min_chars,
        accept=lambda t: _looks_english(t, min_chars),
        label="en_wiki",
    )


def prepare_en_fineweb(out_path: Path, budget: int, seen: set, min_chars: int) -> dict:
    print("   流式下载 HuggingFaceFW/fineweb-edu sample-10BT", flush=True)
    return _write_docs(
        out_path,
        _iter_hf_text("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text"),
        budget,
        seen=seen,
        min_chars=min_chars,
        accept=lambda t: _looks_english(t, min_chars),
        label="en_fineweb_edu",
    )


def prepare_ja_wiki(out_path: Path, budget: int, seen: set, min_chars: int) -> dict:
    print("   流式下载 wikimedia/wikipedia 20231101.ja", flush=True)
    return _write_docs(
        out_path,
        _iter_hf_text("wikimedia/wikipedia", "20231101.ja", "train", "text"),
        budget,
        seen=seen,
        min_chars=min_chars,
        accept=lambda t: _looks_japanese(t, min_chars),
        label="ja_wiki",
    )


def _aozora_ok(ex: dict) -> bool:
    # 若带有版权标记，只保留公版
    for key in ("作品著作権フラグ", "copyright_flag", "copyright"):
        if key in ex and ex[key] not in (None, "", "なし", "none", "None", False, 0, "0"):
            # 値「なし」表示无版权；其它非空可能有版权
            val = str(ex[key]).strip()
            if val and val not in {"なし", "none", "None", "false", "False", "0"}:
                return False
    meta = ex.get("meta")
    if isinstance(meta, dict):
        flag = str(meta.get("作品著作権フラグ") or meta.get("copyright_flag") or "").strip()
        if flag and flag not in {"なし", "none", "None"}:
            return False
    return True


def prepare_ja_aozora(out_path: Path, budget: int, seen: set, min_chars: int) -> dict:
    candidates = [
        ("globis-university/aozorabunko-clean", None, "train", "text"),
        ("Podtech/llm-jp-corpus-v4-ja_aozorabunko", None, "train", "text"),
    ]
    last_err = ""
    for hf_id, config, split, field in candidates:
        print(f"   尝试青空文库 {hf_id}", flush=True)
        try:
            docs = _iter_hf_text(
                hf_id, config, split, field, extra_ok=_aozora_ok
            )
            st = _write_docs(
                out_path,
                docs,
                budget,
                seen=seen,
                min_chars=min_chars,
                accept=lambda t: _looks_japanese(t, min_chars),
                label="ja_aozora",
            )
            st["hf_id"] = hf_id
            if st["docs_kept"] == 0:
                last_err = f"{hf_id}: 0 docs"
                if out_path.exists():
                    out_path.unlink()
                continue
            return st
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            print(f"   [跳过] {hf_id}: {last_err}", flush=True)
            continue
    return {
        "path": str(out_path),
        "status": "skipped",
        "error": last_err or "no aozora source",
        "bytes": 0,
        "docs_kept": 0,
    }


def prepare_ja_fineweb(out_path: Path, budget: int, seen: set, min_chars: int) -> dict:
    print("   流式下载 hotchpotch/fineweb-2-edu-japanese sample_10BT", flush=True)
    # 该集已是教育向过滤；score 下限 2.5，再略提高一点
    return _write_docs(
        out_path,
        _iter_hf_text(
            "hotchpotch/fineweb-2-edu-japanese",
            "sample_10BT",
            "train",
            "text",
            score_field="score",
            min_score=2.6,
        ),
        budget,
        seen=seen,
        min_chars=min_chars,
        accept=lambda t: _looks_japanese(t, min_chars),
        label="ja_fineweb_edu",
    )


def _concat(parts: list[Path], dest: Path) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    total = 0
    used = []
    with tmp.open("wb") as out:
        for p in parts:
            if not p.is_file() or p.stat().st_size == 0:
                continue
            data = p.read_bytes()
            out.write(data)
            if not data.endswith(b"\n"):
                out.write(b"\n")
            total += p.stat().st_size
            used.append(str(p))
    tmp.replace(dest)
    return {
        "path": str(dest),
        "bytes": dest.stat().st_size,
        "parts": used,
    }


def _scale_budgets(frac: float) -> dict:
    if frac >= 0.999:
        return dict(BUDGETS)
    return {k: max(64 * 1024, int(v * frac)) for k, v in BUDGETS.items()}


def main() -> int:
    p = argparse.ArgumentParser(description="准备中英日各约 1GiB SentencePiece 语料")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--min-chars", type=int, default=80)
    p.add_argument(
        "--zh-min-score",
        type=float,
        default=0.80,
        help="中文 Fineweb-Edu V2 的 score 下限（默认 0.80）",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="冒烟：按 1/200 预算试通各源（约每语言 5MiB）",
    )
    p.add_argument("--force", action="store_true", help="忽略已有 parts，重新下载")
    p.add_argument(
        "--only",
        default="",
        help="只跑这些 part key，逗号分隔，如 zh_wiki,en_wiki",
    )
    args = p.parse_args()

    budgets = _scale_budgets(1 / 200 if args.smoke else 1.0)
    out_dir = args.out_dir
    if args.smoke and args.out_dir == DEFAULT_OUT:
        out_dir = DEFAULT_OUT.parent / (DEFAULT_OUT.name + "_smoke")
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    print("=== prepare_spm_corpus_v1 ===", flush=True)
    print(f"out_dir={out_dir}", flush=True)
    print(f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT', '(default hub)')}", flush=True)
    print(f"smoke={args.smoke} force={args.force}", flush=True)
    print(
        "skip traditional wiki: "
        + ", ".join(p.name for p in ZH_WIKI_SKIP),
        flush=True,
    )

    wanted = [x.strip() for x in args.only.split(",") if x.strip()]
    jobs = [
        ("zh_wiki", lambda path, bud, seen: prepare_zh_wiki(path, bud, seen, args.min_chars)),
        (
            "zh_fineweb_edu",
            lambda path, bud, seen: prepare_zh_fineweb(
                path, bud, seen, args.min_chars, args.zh_min_score
            ),
        ),
        ("en_wiki", lambda path, bud, seen: prepare_en_wiki(path, bud, seen, args.min_chars)),
        (
            "en_fineweb_edu",
            lambda path, bud, seen: prepare_en_fineweb(path, bud, seen, args.min_chars),
        ),
        ("ja_wiki", lambda path, bud, seen: prepare_ja_wiki(path, bud, seen, args.min_chars)),
        ("ja_aozora", lambda path, bud, seen: prepare_ja_aozora(path, bud, seen, args.min_chars)),
        (
            "ja_fineweb_edu",
            lambda path, bud, seen: prepare_ja_fineweb(path, bud, seen, args.min_chars),
        ),
    ]
    if wanted:
        jobs = [j for j in jobs if j[0] in wanted]

    # 每种语言共享一份去重集合，避免维基与网页重复
    seen_by_lang = {"zh": set(), "en": set(), "ja": set()}
    stats = {}
    t0 = time.perf_counter()
    errors = []

    def _lang_bytes(lang: str) -> int:
        total = 0
        for k, st in stats.items():
            if k.startswith(lang + "_") and st.get("status") in {"ok", "skipped_existing", "skipped"}:
                total += int(st.get("bytes") or 0)
        return total

    def _fill_budget(key: str, default: int) -> int:
        target = GIB if not args.smoke else int(GIB / 200)
        if key == "zh_fineweb_edu":
            return max(0, target - _lang_bytes("zh"))
        if key == "en_fineweb_edu":
            return max(0, target - _lang_bytes("en"))
        if key == "ja_fineweb_edu":
            return max(0, target - _lang_bytes("ja"))
        return default

    for key, fn in jobs:
        lang = key.split("_", 1)[0]
        budget = _fill_budget(key, budgets[key])
        path = parts_dir / f"{key}.txt"
        if budget <= 0:
            print(f"→ {key}: 预算已满，跳过", flush=True)
            stats[key] = {"path": str(path), "bytes": 0, "status": "skipped_budget"}
            continue
        print(f"→ {key} budget={budget:,} bytes → {path}", flush=True)
        if not args.force and _part_complete(path, budget):
            print(
                f"   已有 {path.stat().st_size:,} bytes，跳过（--force 可重做）",
                flush=True,
            )
            # 把已有文档哈希加入 seen，后续源可去重
            for doc in _iter_local_docs(path):
                seen_by_lang[lang].add(_doc_key(doc))
            stats[key] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "status": "skipped_existing",
            }
            continue
        try:
            st = fn(path, budget, seen_by_lang[lang])
        except Exception as e:
            st = {
                "path": str(path),
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "bytes": path.stat().st_size if path.exists() else 0,
            }
            print(f"   [错误] {key}: {st['error']}", file=sys.stderr, flush=True)
        stats[key] = st
        if st.get("status") == "error":
            errors.append(key)

    # 组装语言大文件：某语言若核心源失败则不写最终文件
    assemblies = {
        "zh": ["zh_wiki", "zh_fineweb_edu"],
        "en": ["en_wiki", "en_fineweb_edu"],
        "ja": ["ja_wiki", "ja_aozora", "ja_fineweb_edu"],
    }
    assembled = {}
    for lang, keys in assemblies.items():
        if wanted and not any(k in wanted for k in keys):
            continue
        core = [k for k in keys if k != "ja_aozora"]
        if any(stats.get(k, {}).get("status") == "error" for k in core):
            print(f"   跳过组装 {lang}.txt（有核心源失败）", flush=True)
            continue
        paths = [parts_dir / f"{k}.txt" for k in keys]
        assembled[lang] = _concat(paths, out_dir / f"{lang}.txt")
        print(
            f"   组装 {lang}.txt: {assembled[lang]['bytes']:,} bytes "
            f"({assembled[lang]['bytes'] / GIB:.3f} GiB)",
            flush=True,
        )

    elapsed = round(time.perf_counter() - t0, 1)
    meta = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "out_dir": str(out_dir),
        "target_bytes_per_lang": GIB if not args.smoke else int(GIB / 200),
        "unit": "UTF-8 bytes, 1 GiB = 1073741824",
        "smoke": bool(args.smoke),
        "hf_endpoint": os.environ.get("HF_ENDPOINT", ""),
        "zh_min_score": args.zh_min_score,
        "min_chars": args.min_chars,
        "zh_wiki_sources": [str(p) for p in ZH_WIKI_FILES],
        "zh_wiki_skipped": [str(p) for p in ZH_WIKI_SKIP],
        "budgets": budgets,
        "parts": stats,
        "assembled": assembled,
        "elapsed_seconds": elapsed,
        "note": "语料已准备；词表需确认后再训。",
    }
    meta_path = out_dir / "manifest.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== 汇总 ===", flush=True)
    for lang, info in assembled.items():
        print(
            f"{lang}: {info['bytes']:,} bytes ({info['bytes'] / GIB:.3f} GiB) → {info['path']}",
            flush=True,
        )
    print(f"manifest: {meta_path}", flush=True)
    print(f"elapsed: {elapsed}s", flush=True)
    if errors:
        print("失败的源: " + ", ".join(errors), file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    code = main()
    # datasets/pyarrow 流式下载后可能留下非守护线程，阻止正常退出
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(code or 0))
