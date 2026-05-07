"""
make_chloe_icon.py — Converts a source PNG into a multi-resolution Windows
.ico file, cropping out dark/empty borders and padding to a clean square so
the icon doesn't end up stretched or surrounded by background.

Run after dropping `chloe_icon_source.png` into the project root:

    (venv) C:\\Users\\eleew\\Documents\\jarvis> python make_chloe_icon.py

Output:
    chloe_icon.ico   (multi-res, 16/32/48/64/128/256, transparent corners)

Pipeline:
  1. Detect the bounding box of meaningful content. If the source has a real
     alpha channel, use that. Otherwise fall back to a brightness threshold
     (assumes a dark background, which AI-generated icons typically have).
  2. Crop to that bbox plus a small breathing margin.
  3. Pad to a square on a transparent canvas (preserves aspect ratio — no
     stretching at the cost of transparent corners, which Windows handles
     fine and looks correct on any desktop wallpaper).
  4. Save as a multi-resolution .ico.
"""

import sys
from pathlib import Path

try:
    from PIL import Image
    import numpy as np
except ImportError as e:
    missing = "Pillow numpy" if "PIL" in str(e) else "numpy"
    print(f"Missing dependency. Run:  pip install {missing}")
    sys.exit(1)

PROJECT_DIR = Path(__file__).resolve().parent
SRC = PROJECT_DIR / "chloe_icon_source.png"
DST = PROJECT_DIR / "chloe_icon.ico"

# Standard Windows icon sizes. 256 is the high-DPI / large-icon view in
# Explorer; 48 is medium icons; 32 is the taskbar; 16 is small icons +
# window title bar. All embedded so the OS picks the closest match without
# scaling artifacts.
SIZES = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]

# Brightness threshold (0..765 = R+G+B summed) below which a pixel is
# considered "background" when the source has no real alpha. 60 ≈ very dark.
# Tune up if the icon's edges are getting nibbled, down if dark background
# is leaking into the bbox.
BG_LUMA_THRESHOLD = 60

# Breathing margin around the detected content, expressed as a fraction of
# the larger source dimension. 2% gives a small visual gap inside the icon.
PAD_FRACTION = 0.02


def find_content_bbox(img: Image.Image) -> tuple[int, int, int, int]:
    """Return (left, top, right, bottom) of the non-background region.

    Uses the alpha channel if present and non-trivial; otherwise uses a
    brightness threshold to separate foreground from a dark background."""
    arr = np.array(img)  # H x W x 4 (RGBA)

    # Try alpha channel first
    if arr.shape[2] == 4:
        alpha = arr[:, :, 3]
        if alpha.min() < 200:
            # Alpha is meaningful (not fully opaque everywhere)
            mask = alpha > 16
            ys, xs = np.where(mask)
            if ys.size > 0:
                print("  using alpha channel for content detection")
                return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    # Fall back to brightness threshold against dark background
    rgb = arr[:, :, :3]
    luma = rgb.sum(axis=2)
    mask = luma > BG_LUMA_THRESHOLD
    ys, xs = np.where(mask)
    if ys.size == 0:
        print("  WARNING: couldn't detect content; using full image")
        h, w = img.size[1], img.size[0]
        return 0, 0, w, h
    print(f"  using brightness threshold {BG_LUMA_THRESHOLD} for content detection")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def main():
    if not SRC.exists():
        print(f"ERROR: source PNG not found at {SRC}")
        print(f"  → save your icon image there as 'chloe_icon_source.png'")
        sys.exit(1)

    print(f"loading {SRC.name} ({SRC.stat().st_size / 1024:.1f} KB)")
    img = Image.open(SRC).convert("RGBA")
    print(f"  source dimensions: {img.size[0]}x{img.size[1]}")

    # Step 1: detect content bounding box
    left, top, right, bottom = find_content_bbox(img)
    bw, bh = right - left, bottom - top
    print(f"  content bbox: ({left},{top}) → ({right},{bottom})  size {bw}x{bh}")

    # Step 2: pad the bbox by a small percentage so the icon doesn't crop
    # right at the visual edge
    pad = int(max(img.size) * PAD_FRACTION)
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(img.size[0], right + pad)
    bottom = min(img.size[1], bottom + pad)
    cropped = img.crop((left, top, right, bottom))
    cw, ch = cropped.size
    print(f"  cropped dimensions: {cw}x{ch}  (with {pad}px breathing margin)")

    # Step 3: pad to square on a TRANSPARENT canvas. This preserves aspect
    # ratio (no stretching) and gives Windows clean alpha-cut corners.
    side = max(cw, ch)
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    ox = (side - cw) // 2
    oy = (side - ch) // 2
    square.paste(cropped, (ox, oy))
    print(f"  padded to {side}x{side} (transparent corners)")

    # Step 4: save the multi-resolution .ico
    square.save(DST, format="ICO", sizes=SIZES)
    print(f"saved {DST.name} ({DST.stat().st_size / 1024:.1f} KB) "
          f"with {len(SIZES)} embedded resolutions")
    print()
    print(f"next: run build.bat to embed the icon in Chloe.exe")


if __name__ == "__main__":
    main()
