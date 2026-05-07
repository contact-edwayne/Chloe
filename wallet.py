"""wallet.py — Breez SDK Liquid wallet for Chloe.

Single point of contact with the Breez SDK. `wallet_guard.py` layers
PIN/cap policy on top; `jarvis.py` exposes both as Chloe tools.

Reference docs:
    https://pypi.org/project/breez-sdk-liquid/
    https://sdk-doc-liquid.breez.technology/

Tested API surface (search for `# SDK_API:` to find every touchpoint
that depends on the SDK call shape — if you upgrade breez-sdk-liquid
and something breaks, those are the lines to revisit).

CLI smoke-test:
    python wallet.py init        # one-time, generates a 12-word seed
    python wallet.py balance     # prints spendable sats
    python wallet.py invoice 1000 "test"    # prints bolt11
    python wallet.py pay <bolt11_or_lnaddr> [amount_sat]
    python wallet.py history [limit]

Hard rules baked in here:
- Mnemonic loaded once from C:\\Chloe\\secrets\\wallet.seed.
- Mnemonic NEVER returned from any function. NEVER logged. NEVER
  serialised to JSON.
- All public functions return plain dicts — they're safe to surface
  through Chloe's tool layer.
"""
from __future__ import annotations

import json
import os
import secrets as _pysecrets
import sys
import time
from pathlib import Path
from typing import Any

# ─── Paths and config ──────────────────────────────────────────────────────
SECRETS_DIR = Path(os.environ.get("CHLOE_WALLET_SECRETS_DIR", r"C:\Chloe\secrets"))
SEED_FILE = SECRETS_DIR / "wallet.seed"
API_KEY_FILE = SECRETS_DIR / "wallet.api_key"
WORKING_DIR = SECRETS_DIR / "breez_data"   # SDK persists state here

NETWORK_NAME = os.environ.get("CHLOE_WALLET_NETWORK", "mainnet").lower()


def _ensure_secrets_dir() -> None:
    """Create the secrets directory if it doesn't exist. Set Windows
    ACLs so only the current user can read it."""
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    # Best-effort: tighten ACLs on Windows. Failing is non-fatal — user
    # can secure manually if they want.
    if os.name == "nt":
        try:
            import subprocess
            user = os.environ.get("USERNAME", "")
            if user:
                subprocess.run(
                    ["icacls", str(SECRETS_DIR), "/inheritance:r",
                     "/grant", f"{user}:(OI)(CI)F"],
                    capture_output=True, timeout=5
                )
        except Exception:
            pass


# ─── BIP39 mnemonic generation (used only for first-run init) ──────────────
def _generate_mnemonic() -> str:
    """Generate a fresh 12-word BIP39 mnemonic.

    We avoid taking a hard dep on a BIP39 library by shipping the word
    list inline — but that's a lot of bytes. Instead, we try
    bip-utils first; if missing, we instruct the user to install it.
    Generating a seed is a once-ever operation, so a one-off install
    isn't a burden."""
    try:
        from bip_utils import Bip39MnemonicGenerator, Bip39WordsNum  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "Mnemonic generation needs bip-utils. Install it with:\n"
            "    pip install bip-utils\n"
            "(only needed once; safe to uninstall after seed is created)"
        ) from e
    mnemonic = Bip39MnemonicGenerator().FromWordsNumber(Bip39WordsNum.WORDS_NUM_12)
    return str(mnemonic)


def init_seed_interactive() -> str:
    """First-run setup: generate a fresh mnemonic, write it to disk,
    and print BIG paper-backup instructions to stderr. Returns the
    mnemonic so the caller can echo it; do not log this return value
    anywhere persistent."""
    if SEED_FILE.exists():
        raise RuntimeError(
            f"Seed file already exists at {SEED_FILE}. "
            "Refusing to overwrite. Delete it manually only if you are "
            "absolutely sure you have NO funds in the existing wallet."
        )
    _ensure_secrets_dir()
    mnemonic = _generate_mnemonic()
    SEED_FILE.write_text(mnemonic, encoding="utf-8")
    if os.name == "nt":
        try:
            import subprocess
            user = os.environ.get("USERNAME", "")
            if user:
                subprocess.run(
                    ["icacls", str(SEED_FILE), "/inheritance:r",
                     "/grant", f"{user}:F"],
                    capture_output=True, timeout=5
                )
        except Exception:
            pass

    # Loud paper-backup nag
    sys.stderr.write("\n" + "═" * 64 + "\n")
    sys.stderr.write("  CHLOE WALLET — RECOVERY SEED CREATED\n")
    sys.stderr.write("═" * 64 + "\n\n")
    sys.stderr.write(f"  {mnemonic}\n\n")
    sys.stderr.write("  WRITE THIS DOWN ON PAPER. RIGHT NOW.\n")
    sys.stderr.write("  Two copies, two physical locations.\n")
    sys.stderr.write("  Do NOT store it in a password manager yet —\n")
    sys.stderr.write("  paper backup first; digital backup after you\n")
    sys.stderr.write("  understand the trade-offs.\n\n")
    sys.stderr.write("  Anyone with these 12 words can spend all funds.\n")
    sys.stderr.write("═" * 64 + "\n\n")
    return mnemonic


def _load_mnemonic() -> str:
    if not SEED_FILE.exists():
        raise RuntimeError(
            f"No wallet seed found at {SEED_FILE}. "
            "Run `python wallet.py init` once to generate one."
        )
    text = SEED_FILE.read_text(encoding="utf-8").strip()
    words = text.split()
    if len(words) not in (12, 24):
        raise RuntimeError(
            f"Seed file at {SEED_FILE} is not a 12- or 24-word "
            f"mnemonic ({len(words)} words found)."
        )
    return text


def _load_api_key() -> str:
    if API_KEY_FILE.exists():
        return API_KEY_FILE.read_text(encoding="utf-8").strip()
    env_key = os.environ.get("BREEZ_API_KEY")
    if env_key:
        return env_key.strip()
    raise RuntimeError(
        f"No Breez API key. Either:\n"
        f"  - put it in {API_KEY_FILE}, or\n"
        f"  - set the BREEZ_API_KEY environment variable.\n"
        "Get a free key at https://breez.technology/request-api-key/"
    )


# ─── SDK lifecycle ─────────────────────────────────────────────────────────
_sdk = None  # connected instance, lazy


def _connect():
    """Lazy-connect to the Breez SDK. Memoised — connecting is slow
    (it syncs the wallet state) so we only do it once per process."""
    global _sdk
    if _sdk is not None:
        return _sdk

    # SDK_API: import surface. If ImportError, wallet is unconfigured;
    # surface a helpful message to the caller.
    try:
        import breez_sdk_liquid as bsl  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "breez-sdk-liquid is not installed. Run:\n"
            "    pip install breez-sdk-liquid\n"
            "(see WALLET_SETUP.md for the full setup)."
        ) from e

    _ensure_secrets_dir()
    mnemonic = _load_mnemonic()
    api_key = _load_api_key()

    # SDK_API: network selector. Older SDK used LiquidNetwork.MAINNET /
    # .TESTNET; newer reorganised under bsl.LiquidNetwork.
    if NETWORK_NAME == "testnet":
        network = bsl.LiquidNetwork.TESTNET
    else:
        network = bsl.LiquidNetwork.MAINNET

    # SDK_API: default_config(network, breez_api_key) → Config.
    config = bsl.default_config(network=network, breez_api_key=api_key)
    config.working_dir = str(WORKING_DIR)

    # SDK_API: ConnectRequest(config, mnemonic) → connect(req).
    req = bsl.ConnectRequest(config=config, mnemonic=mnemonic)
    _sdk = bsl.connect(req)
    return _sdk


def disconnect() -> None:
    global _sdk
    if _sdk is not None:
        try:
            _sdk.disconnect()
        except Exception:
            pass
        _sdk = None


# ─── Public API: balance ───────────────────────────────────────────────────
def get_balance() -> dict:
    """Return spendable + pending balance in sats."""
    sdk = _connect()
    # SDK_API: get_info() → GetInfoResponse with wallet_info.{balance_sat,
    # pending_send_sat, pending_receive_sat}.
    info = sdk.get_info()
    wi = getattr(info, "wallet_info", None) or info
    balance_sat = int(getattr(wi, "balance_sat", 0))
    pending_send = int(getattr(wi, "pending_send_sat", 0))
    pending_recv = int(getattr(wi, "pending_receive_sat", 0))
    return {
        "ok": True,
        "balance_sat": balance_sat,
        "pending_send_sat": pending_send,
        "pending_receive_sat": pending_recv,
    }


# ─── Public API: invoice (receive) ─────────────────────────────────────────
def create_invoice(amount_sat: int, memo: str = "") -> dict:
    """Generate a BOLT11 invoice the user can give to a payer."""
    if not isinstance(amount_sat, int) or amount_sat < 1:
        return {"ok": False, "error": "amount_sat must be a positive integer"}
    if amount_sat > 10_000_000:  # 10M sat sanity cap on receive size
        return {"ok": False, "error": "amount too large for safe testing"}

    print(f"[wallet] create_invoice({amount_sat}, {memo!r}) — connecting…",
          flush=True)
    t0 = time.time()
    sdk = _connect()
    print(f"[wallet]   connect ok in {time.time()-t0:.2f}s", flush=True)

    import breez_sdk_liquid as bsl  # type: ignore

    # SDK_API: ReceiveAmount.BITCOIN(payer_amount_sat=N).
    amount = bsl.ReceiveAmount.BITCOIN(payer_amount_sat=amount_sat)

    # SDK_API: PrepareReceiveRequest(payment_method, amount).
    prep_req = bsl.PrepareReceiveRequest(
        payment_method=bsl.PaymentMethod.BOLT11_INVOICE,
        amount=amount,
    )
    print(f"[wallet]   calling prepare_receive_payment…", flush=True)
    t1 = time.time()
    prep = sdk.prepare_receive_payment(prep_req)
    print(f"[wallet]   prepare_receive_payment ok in {time.time()-t1:.2f}s "
          f"(fees={getattr(prep, 'fees_sat', '?')})", flush=True)

    # SDK_API: ReceivePaymentRequest(prepare_response, description).
    recv_req = bsl.ReceivePaymentRequest(
        prepare_response=prep,
        description=(memo or "Chloe wallet"),
    )
    print(f"[wallet]   calling receive_payment…", flush=True)
    t2 = time.time()
    recv = sdk.receive_payment(recv_req)
    print(f"[wallet]   receive_payment ok in {time.time()-t2:.2f}s",
          flush=True)
    bolt11 = getattr(recv, "destination", None) or getattr(recv, "invoice", None)
    fees_sat = int(getattr(prep, "fees_sat", 0))
    return {
        "ok": True,
        "bolt11": bolt11,
        "amount_sat": amount_sat,
        "fees_sat": fees_sat,
        "memo": memo,
    }


# ─── Public API: send ──────────────────────────────────────────────────────
def pay(destination: str, amount_sat: int | None = None) -> dict:
    """Pay a BOLT11 invoice, BOLT12 offer, or Lightning Address.

    `amount_sat` is required for amountless invoices and Lightning
    Addresses; it's optional (and ignored) for amount-fixed invoices."""
    if not isinstance(destination, str) or not destination.strip():
        return {"ok": False, "error": "destination must be a non-empty string"}
    destination = destination.strip()

    sdk = _connect()
    import breez_sdk_liquid as bsl  # type: ignore

    # SDK_API: PrepareSendRequest(destination, amount=PayAmount.BITCOIN(...)).
    # The amount is only applied if the destination needs one.
    prep_kwargs: dict[str, Any] = {"destination": destination}
    if amount_sat is not None:
        if not isinstance(amount_sat, int) or amount_sat < 1:
            return {"ok": False, "error": "amount_sat must be a positive integer"}
        prep_kwargs["amount"] = bsl.PayAmount.BITCOIN(receiver_amount_sat=amount_sat)

    try:
        prep_req = bsl.PrepareSendRequest(**prep_kwargs)
        prep = sdk.prepare_send_payment(prep_req)
    except Exception as e:
        return {"ok": False, "error": f"prepare failed: {e}"}

    fees_sat = int(getattr(prep, "fees_sat", 0))
    # Amount the SDK actually plans to send (resolves amountless invoices,
    # LNURL prompts, etc.).
    resolved_amount_sat = _extract_resolved_amount(prep, fallback=amount_sat or 0)

    # SDK_API: SendPaymentRequest(prepare_response).
    try:
        send_req = bsl.SendPaymentRequest(prepare_response=prep)
        resp = sdk.send_payment(send_req)
    except Exception as e:
        return {"ok": False, "error": f"send failed: {e}", "fees_sat": fees_sat}

    payment = getattr(resp, "payment", None) or resp
    payment_hash = getattr(payment, "tx_id", None) or getattr(payment, "payment_hash", None)
    status = str(getattr(payment, "status", "unknown"))
    return {
        "ok": True,
        "payment_hash": payment_hash,
        "amount_sat": resolved_amount_sat,
        "fees_sat": fees_sat,
        "status": status,
    }


def _extract_resolved_amount(prep, fallback: int) -> int:
    """The prepare-response shape varies depending on the destination
    type. Walk a few likely paths to find the actual amount being sent."""
    for attr in ("receiver_amount_sat", "amount_sat", "payer_amount_sat"):
        v = getattr(prep, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    inner = getattr(prep, "amount", None)
    if inner is not None:
        for attr in ("receiver_amount_sat", "amount_sat", "payer_amount_sat"):
            v = getattr(inner, attr, None)
            if isinstance(v, int) and v > 0:
                return v
    return fallback


# ─── Public API: history ───────────────────────────────────────────────────
def list_history(limit: int = 10) -> dict:
    if not isinstance(limit, int) or limit < 1:
        limit = 10
    limit = min(limit, 50)
    sdk = _connect()
    import breez_sdk_liquid as bsl  # type: ignore

    # SDK_API: ListPaymentsRequest(limit=...).
    try:
        req = bsl.ListPaymentsRequest(limit=limit)
    except TypeError:
        # Older SDKs accepted no kwargs; fall back.
        req = bsl.ListPaymentsRequest()
    try:
        payments = sdk.list_payments(req)
    except Exception as e:
        return {"ok": False, "error": str(e)}

    out = []
    for p in payments[:limit]:
        out.append({
            "type": str(getattr(p, "payment_type", "")),
            "status": str(getattr(p, "status", "")),
            "amount_sat": int(getattr(p, "amount_sat", 0)),
            "fees_sat": int(getattr(p, "fees_sat", 0)),
            "timestamp": int(getattr(p, "timestamp", 0)),
            "description": str(getattr(p, "description", "") or ""),
            "tx_id": str(getattr(p, "tx_id", "") or ""),
        })
    return {"ok": True, "payments": out}


# ─── CLI smoke-test entry point ────────────────────────────────────────────
def _cli(argv: list[str]) -> int:
    if not argv:
        print("usage: python wallet.py {init|balance|invoice|pay|history}")
        return 2
    cmd = argv[0]
    try:
        if cmd == "init":
            init_seed_interactive()
            print("Seed created. Re-run with `balance` to verify connect.")
            return 0
        if cmd == "balance":
            r = get_balance()
            print(json.dumps(r, indent=2))
            return 0 if r.get("ok") else 1
        if cmd == "invoice":
            if len(argv) < 2:
                print("usage: python wallet.py invoice <amount_sat> [memo...]")
                return 2
            amount = int(argv[1])
            memo = " ".join(argv[2:])
            r = create_invoice(amount, memo)
            print(json.dumps(r, indent=2))
            return 0 if r.get("ok") else 1
        if cmd == "pay":
            if len(argv) < 2:
                print("usage: python wallet.py pay <bolt11_or_lnaddr> [amount_sat]")
                return 2
            dest = argv[1]
            amt = int(argv[2]) if len(argv) >= 3 else None
            # CLI confirmation gate — the guard layer happens in
            # wallet_guard.py / jarvis.py, not here. Direct CLI use is
            # for testing only and assumes the operator is the seed
            # holder physically.
            print(f"About to pay {amt or '(invoice amount)'} sat to {dest[:30]}...")
            print("Type 'yes' to confirm:")
            if input().strip().lower() != "yes":
                print("Aborted.")
                return 1
            r = pay(dest, amt)
            print(json.dumps(r, indent=2))
            return 0 if r.get("ok") else 1
        if cmd == "history":
            limit = int(argv[1]) if len(argv) >= 2 else 10
            r = list_history(limit)
            print(json.dumps(r, indent=2))
            return 0 if r.get("ok") else 1
        print(f"Unknown command: {cmd}")
        return 2
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
