"""
download_kokoro.py — One-time helper to fetch the Kokoro TTS model and
voice files. Drops them in `kokoro_models/` next to jarvis.py.

Run once:
    (venv) C:\\Users\\eleew\\Documents\\jarvis> python download_kokoro.py

Total download: ~330 MB. After it finishes, set USE_KOKORO=1 in your
_env file and restart Chloe.
"""

import sys
import urllib.request
from pathlib import Path

KOKORO_DIR = Path(__file__).resolve().parent / "kokoro_models"
KOKORO_DIR.mkdir(parents=True, exist_ok=True)

# Pinned to the v1.0 release of the kokoro-onnx model files. If these
# URLs go stale (rare for GitHub release artifacts but possible), check
# https://github.com/thewh1teagle/kokoro-onnx/releases for current URLs.
FILES = {
    "kokoro-v1.0.onnx":
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    "voices-v1.0.bin":
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
}


def _progress_callback(blocks_done: int, block_size: int, total_size: int):
    """Inline progress bar for urllib.request.urlretrieve."""
    done = blocks_done * block_size
    if total_size > 0:
        pct = min(100, 100 * done / total_size)
        sys.stdout.write(
            f"\r[kokoro]   {pct:5.1f}%  "
            f"({done / (1024 * 1024):.1f} MB / "
            f"{total_size / (1024 * 1024):.1f} MB)"
        )
    else:
        sys.stdout.write(f"\r[kokoro]   {done / (1024 * 1024):.1f} MB")
    sys.stdout.flush()


def download_one(filename: str, url: str) -> bool:
    target = KOKORO_DIR / filename
    if target.exists() and target.stat().st_size > 0:
        size_mb = target.stat().st_size / (1024 * 1024)
        print(f"[kokoro] {filename} already exists ({size_mb:.1f} MB) — skipping")
        return True
    print(f"[kokoro] downloading {filename}")
    print(f"[kokoro]   from {url}")
    print(f"[kokoro]   to   {target}")
    try:
        urllib.request.urlretrieve(url, target, _progress_callback)
        print()  # newline after the inline progress bar
        size_mb = target.stat().st_size / (1024 * 1024)
        print(f"[kokoro] downloaded {filename} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print()
        print(f"[kokoro] FAILED to download {filename}: {type(e).__name__}: {e}")
        # Clean up the partial file if any
        try:
            if target.exists():
                target.unlink()
        except OSError:
            pass
        return False


def main():
    print(f"[kokoro] target directory: {KOKORO_DIR}")
    all_ok = True
    for fname, url in FILES.items():
        if not download_one(fname, url):
            all_ok = False

    print()
    if all_ok:
        print(f"[kokoro] all files ready in {KOKORO_DIR}")
        print(f"[kokoro] next steps:")
        print(f"[kokoro]   1. pip install kokoro-onnx soundfile")
        print(f"[kokoro]   2. add to _env:  USE_KOKORO=1")
        print(f"[kokoro]   3. restart Chloe")
    else:
        print(f"[kokoro] one or more downloads failed.")
        print(f"[kokoro] you can also download manually from:")
        print(f"[kokoro]   https://github.com/thewh1teagle/kokoro-onnx/releases")
        print(f"[kokoro] put both files in: {KOKORO_DIR}")
        sys.exit(1)


if __name__ == "__main__":
    main()
