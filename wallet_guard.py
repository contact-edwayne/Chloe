"""wallet_guard.py — PIN and daily-cap policy for the Chloe wallet.

This is the safety layer that wraps every spend. It exists so that:

1. Even if the LLM is confused or jailbroken, it can't bypass the
   spend cap or skip the PIN — those checks happen in Python on the
   server side, not in the prompt.
2. The PIN is stored as an argon2 hash, not plain text.
3. The daily ledger is kept locally so even if the SDK loses state,
   the cap still holds for the day.

Public surface:
    set_pin(new_pin)               # one-time: writes argon2 hash
    verify_pin(pin) -> bool        # constant-time check
    daily_spent_sat() -> int       # what's spent so far today
    daily_cap_sat() -> int         # the cap value
    authorize_send(amount, pin) -> (ok, reason)   # the gate
    record_send(amount, payment_hash)             # call AFTER success

CLI:
    python wallet_guard.py set-pin
    python wallet_guard.py status
"""
from __future__ import annotations

import datetime as _dt
import getpass as _gp
import os
import sqlite3
import sys
from pathlib import Path

# Match wallet.py's secrets dir
SECRETS_DIR = Path(os.environ.get("CHLOE_WALLET_SECRETS_DIR", r"C:\Chloe\secrets"))
PIN_FILE = SECRETS_DIR / "wallet.pin"
LEDGER_DB = SECRETS_DIR / "wallet_spend.db"

DEFAULT_DAILY_CAP_SAT = 10_000  # ~a few US dollars at typical BTC prices
MIN_PIN_LEN = 4
MAX_PIN_LEN = 32


def daily_cap_sat() -> int:
    raw = os.environ.get("CHLOE_WALLET_DAILY_CAP_SAT", "")
    if raw.strip().isdigit():
        v = int(raw)
        if v >= 0:
            return v
    return DEFAULT_DAILY_CAP_SAT


# ─── PIN storage (argon2id) ────────────────────────────────────────────────
def _argon2():
    """Lazy-import argon2 with a helpful error if not installed."""
    try:
        from argon2 import PasswordHasher  # type: ignore
        from argon2.exceptions import VerifyMismatchError  # type: ignore
        return PasswordHasher, VerifyMismatchError
    except ImportError as e:
        raise RuntimeError(
            "argon2-cffi is required for the wallet PIN. Install with:\n"
            "    pip install argon2-cffi"
        ) from e


def set_pin(new_pin: str) -> None:
    """Write an argon2id hash of `new_pin` to the PIN file. Overwrites
    any existing PIN (the seed remains unchanged)."""
    if not isinstance(new_pin, str):
        raise ValueError("PIN must be a string")
    pin = new_pin.strip()
    if len(pin) < MIN_PIN_LEN or len(pin) > MAX_PIN_LEN:
        raise ValueError(
            f"PIN must be between {MIN_PIN_LEN} and {MAX_PIN_LEN} characters"
        )
    PasswordHasher, _ = _argon2()
    ph = PasswordHasher()
    digest = ph.hash(pin)
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    PIN_FILE.write_text(digest, encoding="utf-8")
    if os.name == "nt":
        try:
            import subprocess
            user = os.environ.get("USERNAME", "")
            if user:
                subprocess.run(
                    ["icacls", str(PIN_FILE), "/inheritance:r",
                     "/grant", f"{user}:F"],
                    capture_output=True, timeout=5
                )
        except Exception:
            pass


def verify_pin(pin: str) -> bool:
    """Constant-time PIN check via argon2."""
    if not PIN_FILE.exists():
        return False
    if not isinstance(pin, str):
        return False
    PasswordHasher, VerifyMismatchError = _argon2()
    ph = PasswordHasher()
    digest = PIN_FILE.read_text(encoding="utf-8").strip()
    try:
        return bool(ph.verify(digest, pin.strip()))
    except VerifyMismatchError:
        return False
    except Exception:
        return False


# ─── Daily-spend ledger ────────────────────────────────────────────────────
def _conn():
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(LEDGER_DB))
    c.execute("""
        CREATE TABLE IF NOT EXISTS spends (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL,
            day TEXT NOT NULL,
            amount_sat INTEGER NOT NULL,
            payment_hash TEXT
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_spends_day ON spends(day)")
    return c


def _today_key() -> str:
    """Local-time YYYY-MM-DD. The cap rolls over at local midnight."""
    return _dt.date.today().isoformat()


def daily_spent_sat() -> int:
    c = _conn()
    try:
        row = c.execute(
            "SELECT COALESCE(SUM(amount_sat), 0) FROM spends WHERE day = ?",
            (_today_key(),),
        ).fetchone()
        return int(row[0] or 0)
    finally:
        c.close()


def record_send(amount_sat: int, payment_hash: str | None = None) -> None:
    if not isinstance(amount_sat, int) or amount_sat < 0:
        return
    c = _conn()
    try:
        c.execute(
            "INSERT INTO spends (ts, day, amount_sat, payment_hash) "
            "VALUES (?, ?, ?, ?)",
            (int(_dt.datetime.now().timestamp()),
             _today_key(),
             int(amount_sat),
             (payment_hash or "")),
        )
        c.commit()
    finally:
        c.close()


# ─── The gate ──────────────────────────────────────────────────────────────
def authorize_send(amount_sat: int, pin: str) -> tuple[bool, str]:
    """Return (ok, reason). On True, caller may proceed; on False,
    caller MUST NOT submit the payment.

    Order of checks is deliberate: cap first (cheap, doesn't leak PIN
    correctness), then PIN. We don't reveal whether the PIN was wrong
    vs. cap exceeded in the same message — we report whichever blocked."""
    if not PIN_FILE.exists():
        return False, "PIN is not set. Run `python wallet_guard.py set-pin` once."
    if not isinstance(amount_sat, int) or amount_sat < 1:
        return False, "amount_sat must be a positive integer"

    cap = daily_cap_sat()
    spent = daily_spent_sat()
    if amount_sat > cap:
        return False, (
            f"Single payment exceeds daily cap "
            f"({amount_sat} > {cap} sat). Refusing."
        )
    if spent + amount_sat > cap:
        remaining = max(cap - spent, 0)
        return False, (
            f"Daily cap would be exceeded "
            f"({spent} + {amount_sat} > {cap} sat). "
            f"Remaining today: {remaining} sat."
        )

    if not verify_pin(pin or ""):
        return False, "Incorrect PIN."

    return True, "ok"


# ─── CLI ───────────────────────────────────────────────────────────────────
def _cli(argv: list[str]) -> int:
    if not argv:
        print("usage: python wallet_guard.py {set-pin|status}")
        return 2
    cmd = argv[0]
    if cmd == "set-pin":
        SECRETS_DIR.mkdir(parents=True, exist_ok=True)
        if PIN_FILE.exists():
            print(f"A PIN is already set at {PIN_FILE}.")
            print("Continue and overwrite? Type 'yes' to proceed:")
            if input().strip().lower() != "yes":
                print("Aborted.")
                return 1
        new = _gp.getpass("New PIN: ")
        confirm = _gp.getpass("Confirm PIN: ")
        if new != confirm:
            print("PINs did not match.")
            return 1
        try:
            set_pin(new)
        except Exception as e:
            print(f"Error: {e}")
            return 1
        print(f"PIN written to {PIN_FILE}.")
        return 0
    if cmd == "status":
        print(f"Secrets dir : {SECRETS_DIR}")
        print(f"PIN set     : {PIN_FILE.exists()}")
        print(f"Daily cap   : {daily_cap_sat()} sat")
        print(f"Spent today : {daily_spent_sat()} sat")
        return 0
    print(f"Unknown command: {cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
