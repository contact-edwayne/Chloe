"""test_wallet_balance_timing.py - Regression test for the wallet SDK
cold-connect latency fix (2026-09-06 Round 10, action item #7 coverage):
get_balance() used to call _connect()/get_info() with zero timing
visibility anywhere. A fresh-boot "What's my wallet balance?" once
logged 115.49s total against only 26.57s of measured Ollama round time
-- an 89s gap that landed entirely inside this one function. Fixed by
timing _connect() and get_info() directly and logging either one if it
takes more than 1s, plus a boot-time _warm_wallet() thread in jarvis.py
that pays the SDK's slow first-connect cost proactively (not covered
here -- see this file's own module docstring note below for why).

wallet.py is safe to import directly (no live connection happens at
import time, only inside _connect()/get_balance() themselves) -- unlike
jarvis.py, see test_grounding_and_barge_in.py's docstring for why that
one needs ast-extraction instead.

This test never touches the real Breez SDK: it monkeypatches
wallet._connect with a controllable fake that can simulate a slow or
fast connect, and a fake sdk object with a controllable get_info().

Run from the jarvis dir:
    python test_wallet_balance_timing.py
Exit code 0 on success, non-zero on any failure.
"""
import contextlib
import io
import time
import types

import wallet

PASSED = 0
FAILED = 0


def check(label, cond, detail=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  PASS  {label}")
    else:
        FAILED += 1
        print(f"  FAIL  {label}" + (f"  ({detail})" if detail else ""))


class _FakeWalletInfo:
    def __init__(self, balance_sat=0, pending_send_sat=0, pending_receive_sat=0):
        self.balance_sat = balance_sat
        self.pending_send_sat = pending_send_sat
        self.pending_receive_sat = pending_receive_sat


class _FakeInfo:
    def __init__(self, wallet_info):
        self.wallet_info = wallet_info


class _FakeSdk:
    def __init__(self, wallet_info, get_info_delay=0.0):
        self._wallet_info = wallet_info
        self._get_info_delay = get_info_delay

    def get_info(self):
        if self._get_info_delay:
            time.sleep(self._get_info_delay)
        return _FakeInfo(self._wallet_info)


def _patch_connect(monkeypatch_target, *, connect_delay=0.0,
                    balance_sat=1234, get_info_delay=0.0):
    """Replace wallet._connect with a fake that sleeps `connect_delay`
    seconds then returns a fake sdk whose get_info() sleeps
    `get_info_delay` seconds. Returns the original _connect so the
    caller can restore it."""
    original = wallet._connect

    def _fake_connect():
        if connect_delay:
            time.sleep(connect_delay)
        return _FakeSdk(_FakeWalletInfo(balance_sat=balance_sat),
                         get_info_delay=get_info_delay)

    wallet._connect = _fake_connect
    return original


def test_fast_connect_and_fast_get_info_logs_nothing():
    original = _patch_connect(wallet, connect_delay=0.0, get_info_delay=0.0)
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = wallet.get_balance()
        log = buf.getvalue()
        check("a fast connect + fast get_info returns ok with the right "
              "balance", result == {"ok": True, "balance_sat": 1234,
                                     "pending_send_sat": 0,
                                     "pending_receive_sat": 0}, result)
        check("no timing line is printed when both calls are fast "
              "(under the 1s threshold)", "[wallet]" not in log, log)
    finally:
        wallet._connect = original


def test_slow_connect_logs_connect_timing():
    original = _patch_connect(wallet, connect_delay=1.1, get_info_delay=0.0)
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            wallet.get_balance()
        log = buf.getvalue()
        check("a slow (>1s) _connect() logs its own timing line -- this "
              "is the exact visibility that was missing during the "
              "live 89s hidden gap",
              "_connect() took" in log, log)
        check("get_info() timing line is NOT printed when get_info "
              "itself was fast", "get_info() took" not in log, log)
    finally:
        wallet._connect = original


def test_slow_get_info_logs_get_info_timing():
    original = _patch_connect(wallet, connect_delay=0.0, get_info_delay=1.1)
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            wallet.get_balance()
        log = buf.getvalue()
        check("a slow (>1s) get_info() logs its own timing line, "
              "separately from _connect()'s",
              "get_info() took" in log, log)
        check("_connect() timing line is NOT printed when _connect "
              "itself was fast", "_connect() took" not in log, log)
    finally:
        wallet._connect = original


def test_pending_amounts_pass_through():
    original = _patch_connect(wallet, connect_delay=0.0, get_info_delay=0.0)
    try:
        wallet._sdk = None  # _connect is faked but result isn't memoised here
        fake_sdk = _FakeSdk(_FakeWalletInfo(balance_sat=500,
                                             pending_send_sat=10,
                                             pending_receive_sat=20))
        wallet._connect = lambda: fake_sdk
        result = wallet.get_balance()
        check("pending_send_sat and pending_receive_sat both pass "
              "through to the returned dict, not just balance_sat",
              result["pending_send_sat"] == 10
              and result["pending_receive_sat"] == 20, result)
    finally:
        wallet._connect = original


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
    print(f"\n{PASSED} passed, {FAILED} failed")
    raise SystemExit(1 if FAILED else 0)
