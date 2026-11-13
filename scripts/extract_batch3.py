#!/usr/bin/env python3
"""Extract batch3: ~100 EPUBs from root series into data/raw_epub/batch3_NNN.txt."""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow importing sibling module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_epub import extract_epub  # noqa: E402

ROOT = Path("/Users/qianxiao/worker/llm-lab")
OUT_DIR = ROOT / "data" / "raw_epub"
CANDIDATES = Path("/tmp/batch3_candidates.txt")
MANIFEST_PATH = OUT_DIR / "manifest_batch3.json"


def main() -> int:
    sources = [Path(line.strip()) for line in CANDIDATES.read_text().splitlines() if line.strip()]
    if len(sources) != 100:
        print(f"WARN: expected 100 candidates, got {len(sources)}", file=sys.stderr)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    books: list[dict] = []
    total = 0
    ok = 0
    fail = 0

    for i, src in enumerate(sources, start=1):
        out_name = f"batch3_{i:03d}.txt"
        out = OUT_DIR / out_name
        entry: dict = {
            "index": i,
            "title": src.stem,
            "source": str(src),
            "output": str(out),
            "series": src.parent.name,
        }
        print(f"[{i}/{len(sources)}] {src.name} ...", flush=True)
        if not src.is_file():
            entry["status"] = "error"
            entry["error"] = "source missing"
            entry["char_count"] = 0
            books.append(entry)
            fail += 1
            print(f"  MISSING", flush=True)
            continue
        try:
            text, log = extract_epub(src)
            out.write_text(text, encoding="utf-8")
            n = len(text)
            total += n
            ok += 1
            entry["status"] = "ok"
            entry["char_count"] = n
            entry["chapters_kept"] = sum(1 for c in log if c.get("status") == "kept")
            entry["chapters_skipped"] = sum(1 for c in log if c.get("status") == "skipped")
            # omit full chapter_log to keep manifest smaller
            print(
                f"  OK chars={n} kept={entry['chapters_kept']} skipped={entry['chapters_skipped']}",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            entry["status"] = "error"
            entry["error"] = f"{type(e).__name__}: {e}"
            entry["char_count"] = 0
            fail += 1
            print(f"  FAIL {e}", flush=True)
        books.append(entry)

    summary = {
        "batch": 3,
        "requested": len(sources),
        "ok_count": ok,
        "fail_count": fail,
        "total_chars": total,
        "cleaning": {
            "spine_order": True,
            "skip_front_back": True,
            "zhconv": "zh-cn",
        },
        "books": books,
    }
    MANIFEST_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nWrote {MANIFEST_PATH}")
    print(f"ok={ok} fail={fail} total_chars={total}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
