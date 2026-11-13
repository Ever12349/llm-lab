#!/usr/bin/env python3
"""Extract batch5: ALL remaining leftover EPUBs with resumable progress checkpoints."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_epub import extract_epub  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw_epub"
SOURCES = OUT_DIR / "manifest_batch5_sources.txt"
PROGRESS = OUT_DIR / "batch5_progress.json"
MANIFEST = OUT_DIR / "manifest_batch5.json"
CHECKPOINT_EVERY = 10


def load_progress() -> dict:
    if PROGRESS.is_file():
        return json.loads(PROGRESS.read_text(encoding="utf-8"))
    return {
        "batch": 4,
        "done_count": 0,
        "ok_count": 0,
        "fail_count": 0,
        "total_chars": 0,
        "books": [],
        "failures": [],
        "last_index": 0,
    }


def save_progress(prog: dict) -> None:
    PROGRESS.write_text(
        json.dumps(prog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    sources = [
        Path(line.strip())
        for line in SOURCES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"sources: {len(sources)} from {SOURCES}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prog = load_progress()
    start_idx = prog.get("last_index", 0)  # 1-based last completed; 0 = none
    # Rebuild index map from existing books if resuming
    done_by_index = {b["index"]: b for b in prog.get("books", [])}

    t0 = time.time()
    for i, src in enumerate(sources, start=1):
        if i <= start_idx and i in done_by_index:
            continue
        out_name = f"batch5_{i:04d}.txt"
        out = OUT_DIR / out_name
        entry: dict = {
            "index": i,
            "title": src.stem,
            "source": str(src),
            "output": str(out),
            "series": src.parent.name,
        }
        print(f"[{i}/{len(sources)}] {src.name} ...", flush=True)
        if out.is_file() and out.stat().st_size > 0:
            n = out.stat().st_size  # approximate; re-count chars properly
            try:
                n = len(out.read_text(encoding="utf-8"))
            except Exception:
                pass
            entry["status"] = "ok"
            entry["char_count"] = n
            entry["skipped_existing"] = True
            prog["ok_count"] = prog.get("ok_count", 0) + 1
            prog["total_chars"] = prog.get("total_chars", 0) + n
            print(f"  SKIP existing chars={n}", flush=True)
        elif not src.is_file():
            entry["status"] = "error"
            entry["error"] = "source missing"
            entry["char_count"] = 0
            prog["fail_count"] = prog.get("fail_count", 0) + 1
            prog.setdefault("failures", []).append(
                {"index": i, "source": str(src), "error": "source missing"}
            )
            print("  MISSING", flush=True)
        else:
            try:
                text, log = extract_epub(src)
                out.write_text(text, encoding="utf-8")
                n = len(text)
                entry["status"] = "ok"
                entry["char_count"] = n
                entry["chapters_kept"] = sum(1 for c in log if c.get("status") == "kept")
                entry["chapters_skipped"] = sum(
                    1 for c in log if c.get("status") == "skipped"
                )
                prog["ok_count"] = prog.get("ok_count", 0) + 1
                prog["total_chars"] = prog.get("total_chars", 0) + n
                print(
                    f"  OK chars={n} kept={entry['chapters_kept']} "
                    f"skipped={entry['chapters_skipped']}",
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001
                entry["status"] = "error"
                entry["error"] = f"{type(e).__name__}: {e}"
                entry["char_count"] = 0
                prog["fail_count"] = prog.get("fail_count", 0) + 1
                prog.setdefault("failures", []).append(
                    {"index": i, "source": str(src), "error": entry["error"]}
                )
                print(f"  FAIL {e}", flush=True)

        # Update books list: append or replace
        if i in done_by_index:
            for j, b in enumerate(prog["books"]):
                if b["index"] == i:
                    prog["books"][j] = entry
                    break
        else:
            prog["books"].append(entry)
            done_by_index[i] = entry

        prog["last_index"] = i
        prog["done_count"] = i
        prog["requested"] = len(sources)
        prog["elapsed_sec"] = round(time.time() - t0, 1)

        if i % CHECKPOINT_EVERY == 0 or i == len(sources):
            save_progress(prog)
            print(
                f"  checkpoint i={i} ok={prog['ok_count']} fail={prog['fail_count']} "
                f"chars={prog['total_chars']}",
                flush=True,
            )

    save_progress(prog)

    summary = {
        "batch": 4,
        "requested": len(sources),
        "ok_count": prog["ok_count"],
        "fail_count": prog["fail_count"],
        "total_chars": prog["total_chars"],
        "elapsed_sec": prog.get("elapsed_sec"),
        "cleaning": {
            "spine_order": True,
            "skip_front_back": True,
            "zhconv": "zh-cn",
        },
        "failures": prog.get("failures", []),
        "books": prog["books"],
    }
    MANIFEST.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nWrote {MANIFEST}", flush=True)
    print(
        f"ok={summary['ok_count']} fail={summary['fail_count']} "
        f"total_chars={summary['total_chars']}",
        flush=True,
    )
    return 0 if summary["ok_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
