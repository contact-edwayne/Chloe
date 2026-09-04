"""bench_ollama.py - measure Ollama latency the way Chloe experiences it.

Hits localhost:11434 directly (no jarvis.py / WS imports), so it's safe to
run at any time and isolates the model's behavior from Chloe's plumbing.
Reports time-to-first-token + total time for both streaming and
non-streaming, plus a cold-vs-warm second pass.

Usage:
    python bench_ollama.py
    python bench_ollama.py "what is the kelly criterion"
    OLLAMA_MODEL=qwen2.5:14b python bench_ollama.py

Exit 0 always (this is a measurement tool, not a test).
"""
import json
import os
import sys
import time

import requests

URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:32b").strip()
PROMPT = sys.argv[1] if len(sys.argv) > 1 else "Briefly: what is the Kelly criterion?"
MAX_TOKENS = int(os.environ.get("BENCH_MAX_TOKENS", "120"))


def _payload(stream):
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": stream,
        "keep_alive": os.environ.get("OLLAMA_KEEP_ALIVE", "30m"),
        "options": {"temperature": 0.7, "num_predict": MAX_TOKENS},
    }


def time_streaming():
    """Returns (first_token_s, total_s, char_count, tokens_per_sec)."""
    t0 = time.perf_counter()
    first_at = None
    chars = 0
    with requests.post(f"{URL}/api/chat", json=_payload(True),
                       timeout=300, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except Exception:
                continue
            piece = (chunk.get("message", {}) or {}).get("content") or ""
            if piece:
                if first_at is None:
                    first_at = time.perf_counter() - t0
                chars += len(piece)
            if chunk.get("done"):
                break
    total = time.perf_counter() - t0
    gen_time = max(total - (first_at or 0), 0.001)
    return first_at, total, chars, chars / 4 / gen_time  # ~4 chars/tok


def time_nonstreaming():
    """Returns (total_s, char_count)."""
    t0 = time.perf_counter()
    r = requests.post(f"{URL}/api/chat", json=_payload(False), timeout=300)
    r.raise_for_status()
    total = time.perf_counter() - t0
    content = (r.json().get("message", {}) or {}).get("content") or ""
    return total, len(content)


def banner(s):
    print()
    print("=" * 64)
    print(" " + s)
    print("=" * 64)


def main():
    print(f"URL:    {URL}")
    print(f"MODEL:  {MODEL}")
    print(f"PROMPT: {PROMPT!r}")
    print(f"MAX_TOKENS: {MAX_TOKENS}")

    # Quick reachability check
    try:
        r = requests.get(f"{URL}/api/tags", timeout=3)
        r.raise_for_status()
        names = [m.get("name", "") for m in r.json().get("models", [])]
        print(f"Ollama up. Models pulled: {len(names)}")
        if not any(MODEL in n or n.startswith(MODEL.split(':')[0]) for n in names):
            print(f"WARNING: target model '{MODEL}' not in {names}")
    except Exception as e:
        print(f"FATAL: Ollama not reachable at {URL}: {e}")
        return

    banner("Pass 1 (cold or warm — depends on current Ollama state)")
    try:
        f1, t1, c1, tps1 = time_streaming()
        print(f"  STREAMING     first token: {f1:.2f}s   total: {t1:.2f}s   "
              f"chars: {c1}   ~tok/s: {tps1:.1f}")
    except Exception as e:
        print(f"  STREAMING     FAILED: {e}")

    try:
        t1n, c1n = time_nonstreaming()
        print(f"  NON-STREAMING total: {t1n:.2f}s   chars: {c1n}   "
              f"(user sees nothing until total elapses)")
    except Exception as e:
        print(f"  NON-STREAMING FAILED: {e}")

    banner("Pass 2 (model now warm in VRAM — this is the steady-state cost)")
    try:
        f2, t2, c2, tps2 = time_streaming()
        print(f"  STREAMING     first token: {f2:.2f}s   total: {t2:.2f}s   "
              f"chars: {c2}   ~tok/s: {tps2:.1f}")
    except Exception as e:
        print(f"  STREAMING     FAILED: {e}")

    try:
        t2n, c2n = time_nonstreaming()
        print(f"  NON-STREAMING total: {t2n:.2f}s   chars: {c2n}")
    except Exception as e:
        print(f"  NON-STREAMING FAILED: {e}")

    banner("Interpretation")
    print("  - Pass 1 'first token' high (>10s) and pass 2 'first token' low (<3s)")
    print("    => the gap is cold-load. The keep_alive + boot warm-up fixes that.")
    print("  - 'NON-STREAMING total' is what Chloe currently makes you wait")
    print("    before showing ANY word. STREAMING 'first token' is what the new")
    print("    code will deliver. The bigger the gap, the bigger the perceived win.")


if __name__ == "__main__":
    main()
