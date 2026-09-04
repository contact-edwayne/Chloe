"""Weekly lint runner for Chloe's brain.

Loads .env, runs BRAIN.lint(), prints a one-line summary plus details.
Designed to be invoked from Task Scheduler via lint_weekly.bat.

Output is appended to lint_weekly.log next to this script — Edward can
inspect it on Mondays. Findings (orphans, contradictions) get persisted
to wiki/gaps.md by the lint operation itself.
"""
import os
import sys
import datetime
from pathlib import Path

# Load .env from this script's directory so GROQ_API_KEY etc. are present
# before we import brain_wiring (which does lazy init based on env).
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / '.env')
except ImportError:
    pass  # fall back to whatever's already in the environment

from brain_wiring import BRAIN  # noqa: E402

stamp = datetime.datetime.now().isoformat(timespec='seconds')
print(f"[{stamp}] lint starting…")
try:
    result = BRAIN.lint()
except Exception as e:
    print(f"[{stamp}] lint FAILED: {type(e).__name__}: {e}")
    sys.exit(1)

orphans = result.get('orphans', [])
contradictions = result.get('contradictions', [])
pages = result.get('pages_scanned', 0)

print(f"[{stamp}] lint complete | {pages} pages | "
      f"{len(orphans)} orphans | {len(contradictions)} contradictions")

if orphans:
    sample = ', '.join(orphans[:10])
    suffix = '...' if len(orphans) > 10 else ''
    print(f"  orphans: {sample}{suffix}")

if contradictions:
    print(f"  contradictions surfaced — see C:\\Chloe\\brain\\wiki\\gaps.md")
