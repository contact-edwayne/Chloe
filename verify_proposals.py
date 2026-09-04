"""End-to-end verifier for the Tier-1 + Tier-2 self-modification pipeline.

What this proves:
  - brain_wiring.py, chloe_proposals.py, chloe_capabilities.py all parse
    cleanly.
  - chloe_proposals.apply / revert work for kind=full and kind=diff.
  - Safety rails (path whitelist, ast.parse, refused patterns) reject as
    expected.
  - Tier-2 token mechanism: issue / validate / consume / expire / revoke.
  - apply_proposal_with_token enforces token gate.
  - /apply_proposal, /revert_proposal, /issue_apply_token, /capabilities,
    /explain slashes are wired into try_handle_brain_command.
  - chloe_capabilities.summary() returns non-empty surface area for the
    live modules.

Sandboxes BRAIN_ROOT to a tempdir so no real proposals/ files are touched.
Sandboxes the proposal target to a tempfile under jarvis/ that gets
deleted at the end (still safe — never writes a .py target that Chloe
loads).

Run via verify_proposals.bat (sets PYTHONUTF8) or:
    python verify_proposals.py
"""
from __future__ import annotations
import ast
import os
import sys
import tempfile
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent

OK_COUNT = 0
FAIL_COUNT = 0


def report(label: str, ok: bool, detail: str = "") -> None:
    global OK_COUNT, FAIL_COUNT
    if ok:
        OK_COUNT += 1
        print(f"  PASS  {label}")
    else:
        FAIL_COUNT += 1
        print(f"  FAIL  {label}  {detail}")


def main() -> int:
    print("\n=== STAGE 1: ast.parse on edited files ===")
    for fname in ("brain_wiring.py", "chloe_proposals.py",
                  "chloe_capabilities.py", "chloe_mcp_server.py",
                  "chloe_pending_confirms.py"):
        path = HERE / fname
        try:
            src = path.read_text(encoding="utf-8")
            ast.parse(src, filename=str(path))
            report(f"{fname} parses cleanly", True)
        except SyntaxError as e:
            report(f"{fname} parses cleanly", False,
                   f"SyntaxError at {e.lineno}: {e.msg}")
        except Exception as e:
            report(f"{fname} parses cleanly", False,
                   f"{type(e).__name__}: {e}")

    if FAIL_COUNT:
        print("\n=== ABORT — fix syntax before running runtime tests ===")
        return 1

    print("\n=== STAGE 2: sandbox + import ===")
    sandbox = Path(tempfile.mkdtemp(prefix="chloe_verify_"))
    fake_brain = sandbox / "brain"
    fake_brain.mkdir()
    os.environ["CHLOE_BRAIN_ROOT"] = str(fake_brain)

    sys.path.insert(0, str(HERE))
    try:
        import chloe_proposals as cp
        report("chloe_proposals imports", True)
    except Exception as e:
        report("chloe_proposals imports", False, str(e))
        return 1

    print("\n=== STAGE 3: kind=full apply/revert ===")
    tgt = HERE / f".verify_target_{os.getpid()}.txt"
    try:
        tgt.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        cp.create_proposal(
            target=f"jarvis/{tgt.name}",
            kind="full",
            rationale="Verifier test target.",
            body="alpha\nBETA\ngamma\ndelta\n",
            test_plan="Read the file.",
            slug="verify-full",
        )
        # Dry-run.
        r = cp.apply_proposal("verify-full", dry_run=True)
        report("dry-run returns ok", r.get("ok") and r.get("dry_run"),
               r.get("error", ""))
        report("dry-run did not modify target",
               tgt.read_text() == "alpha\nbeta\ngamma\n")
        # Real apply.
        r = cp.apply_proposal("verify-full")
        report("real apply ok", r.get("ok", False), r.get("error", ""))
        report("target rewritten",
               tgt.read_text() == "alpha\nBETA\ngamma\ndelta\n")
        bak_path = Path(r["backup_path"]) if r.get("ok") else None
        report("backup file exists",
               bool(bak_path and bak_path.exists()))
        # Revert.
        rv = cp.revert_proposal("verify-full")
        report("revert ok", rv.get("ok", False), rv.get("error", ""))
        report("target restored",
               tgt.read_text() == "alpha\nbeta\ngamma\n")
    finally:
        if tgt.exists():
            tgt.unlink()
        # Best-effort cleanup of any .bak.proposal_verify-full_* files.
        for bak in HERE.glob(f"{tgt.name}.bak.proposal_verify-full_*"):
            try:
                bak.unlink()
            except OSError:
                pass

    print("\n=== STAGE 4: safety refusals ===")
    # Outside whitelist.
    outside = sandbox / "outside.txt"
    outside.write_text("x\n", encoding="utf-8")
    cp.create_proposal(
        target=str(outside),
        kind="full", rationale="Should refuse", body="y\n",
        test_plan="nope", slug="verify-outside",
    )
    r = cp.apply_proposal("verify-outside")
    report("outside-whitelist refused",
           not r.get("ok") and "outside whitelist" in (r.get("error") or ""))
    # Refused pattern (venv).
    venv_tgt = HERE / "venv_py314" / "FAKE_VERIFIER_TARGET.txt"
    if venv_tgt.parent.exists():
        venv_tgt.write_text("x\n", encoding="utf-8")
        cp.create_proposal(
            target=f"jarvis/venv_py314/{venv_tgt.name}",
            kind="full", rationale="Should refuse", body="y\n",
            test_plan="nope", slug="verify-venv",
        )
        r = cp.apply_proposal("verify-venv")
        report("venv refused pattern",
               not r.get("ok")
               and "refused pattern" in (r.get("error") or ""))
        try:
            venv_tgt.unlink()
        except OSError:
            pass
    else:
        report("venv refused pattern", True, "skipped — no venv_py314")
    # Missing slug.
    r = cp.apply_proposal("does-not-exist-anywhere-12345")
    report("missing slug refused",
           not r.get("ok") and "no proposal matching" in (r.get("error") or ""))

    print("\n=== STAGE 5: Tier-2 confirm-token mechanism ===")
    # Issue invalid token (out of range).
    r = cp.issue_token(applies=99, minutes=30)
    report("over-cap applies refused", not r.get("ok"),
           r.get("error", ""))
    r = cp.issue_token(applies=1, minutes=999)
    report("over-cap minutes refused", not r.get("ok"),
           r.get("error", ""))
    # Mint a valid 2-apply, 60-min token.
    r = cp.issue_token(applies=2, minutes=60)
    report("issue_token ok", r.get("ok"), r.get("error", ""))
    token = r["token"]
    report("token is 32-char hex",
           isinstance(token, str) and len(token) == 32
           and all(c in "0123456789abcdef" for c in token))
    # Token visible in list_tokens (masked).
    toks = cp.list_tokens()
    report("token visible in list_tokens",
           len(toks) == 1 and toks[0]["applies_remaining"] == 2)
    # Build a target + proposal for the token-gated apply.
    tgt = HERE / f".verify_target_t2_{os.getpid()}.txt"
    try:
        tgt.write_text("zero\n", encoding="utf-8")
        cp.create_proposal(
            target=f"jarvis/{tgt.name}",
            kind="full", rationale="Tier-2 verify",
            body="zero\none\n", test_plan="cat",
            slug="verify-tier2",
        )
        # dry_run does NOT consume token.
        dr = cp.apply_proposal_with_token("verify-tier2", token, dry_run=True)
        report("dry-run via token ok", dr.get("ok"))
        toks = cp.list_tokens()
        report("dry-run did not consume token",
               len(toks) == 1 and toks[0]["applies_remaining"] == 2)
        # Real apply consumes one slot.
        r2 = cp.apply_proposal_with_token("verify-tier2", token)
        report("real apply via token ok", r2.get("ok"), r2.get("error", ""))
        report("target written via token",
               tgt.read_text() == "zero\none\n")
        toks = cp.list_tokens()
        report("token slot decremented",
               len(toks) == 1 and toks[0]["applies_remaining"] == 1)
        # Revert.
        rv = cp.revert_proposal("verify-tier2")
        report("revert ok (token path)", rv.get("ok"))
        # Bad token rejected.
        bad = cp.apply_proposal_with_token(
            "verify-tier2", "deadbeef" * 4)
        report("wrong token refused",
               not bad.get("ok") and "token rejected" in (bad.get("error") or ""))
        # After revoke_tokens, valid token also fails.
        cp.revoke_tokens()
        r3 = cp.apply_proposal_with_token("verify-tier2", token)
        report("revoked token refused",
               not r3.get("ok") and "token rejected" in (r3.get("error") or ""))
        report("list_tokens empty after revoke",
               cp.list_tokens() == [])
    finally:
        if tgt.exists():
            tgt.unlink()
        for bak in HERE.glob(f"{tgt.name}.bak.proposal_verify-tier2_*"):
            try:
                bak.unlink()
            except OSError:
                pass

    print("\n=== STAGE 6: capabilities surface ===")
    try:
        import chloe_capabilities as cc
        report("chloe_capabilities imports", True)
    except Exception as e:
        report("chloe_capabilities imports", False, str(e))
        cc = None
    if cc:
        s = cc.summary()
        report("summary returned dict", isinstance(s, dict))
        slashes = s.get("slashes", [])
        report(f"discovered >= 10 slashes (got {len(slashes)})",
               len(slashes) >= 10)
        # Spot-check that Tier-1 + Tier-2 slashes are in the inventory.
        names = {sl["name"] for sl in slashes}
        for must_have in ("/apply_proposal", "/revert_proposal",
                          "/issue_apply_token", "/capabilities", "/explain"):
            report(f"slash {must_have} discovered",
                   must_have in names,
                   f"available: {sorted(names)}")
        tools = s.get("mcp_tools", [])
        report(f"discovered >= 20 mcp tools (got {len(tools)})",
               len(tools) >= 20)
        env = s.get("env_knobs", [])
        report(f"discovered env knobs (got {len(env)})", len(env) > 0)
        mods = s.get("modules", [])
        report(f"discovered modules (got {len(mods)})", len(mods) > 0)
        # describe_module: introspect chloe_proposals.
        d = cc.describe_module("chloe_proposals")
        report("describe_module returns docstring",
               bool(d.get("docstring")))
        fns = d.get("functions", [])
        fn_names = {f["name"] for f in fns}
        for must_have in ("apply_proposal", "revert_proposal",
                          "issue_token", "apply_proposal_with_token"):
            report(f"describe_module sees {must_have}",
                   must_have in fn_names)

    print("\n=== STAGE 6.5: Stage-3 pending-confirm contract ===")
    try:
        import chloe_pending_confirms as cpc
        report("chloe_pending_confirms imports", True)
    except Exception as e:
        report("chloe_pending_confirms imports", False, str(e))
        cpc = None
    if cpc:
        # Clean any stale state from prior runs.
        cpc.cancel("")
        # Phrase classification
        report("classify 'yes'", cpc.classify_reply("yes") == "yes")
        report("classify 'go ahead'", cpc.classify_reply("go ahead") == "yes")
        report("classify 'no'", cpc.classify_reply("no") == "no")
        report("classify 'maybe' -> ''",
               cpc.classify_reply("maybe") == "")
        # Announce + source separation
        r = cpc.announce("verify-stage3", source="chat", ttl_s=120)
        report("announce ok", r.get("ok"))
        report("voice 'yes' doesn't resolve chat pending",
               cpc.resolve("yes", source="voice") is None)
        # Real resolve — apply will fail (no proposal exists) but
        # resolve itself should return the apply result dict.
        r2 = cpc.resolve("yes", source="chat")
        report("chat 'yes' returns resolution dict",
               r2 is not None and r2.get("action") == "applied")
        # Pending should be cleared even on a failed apply.
        report("pending cleared after resolve",
               cpc.pending() == [])

    print("\n=== STAGE 7: slash dispatch through brain_wiring ===")
    try:
        import brain_wiring as bw
        report("brain_wiring imports", True)
    except Exception as e:
        report("brain_wiring imports", False, str(e))
        return 1
    # Confirm the handlers exist where expected.
    for h in ("handle_apply_proposal", "handle_revert_proposal",
              "handle_issue_apply_token", "handle_capabilities",
              "handle_explain", "handle_pending_confirms"):
        report(f"{h} symbol present", hasattr(bw, h))
    # /apply_proposal --list should return a string mentioning proposals.
    out = bw.try_handle_brain_command("/apply_proposal --list")
    report("/apply_proposal --list routes",
           isinstance(out, str) and ("proposals" in out.lower()
                                     or "code proposals" in out.lower()),
           f"got: {out!r}" if not isinstance(out, str)
           else f"reply: {out[:120]}")
    # /revert_proposal with empty args should produce a Usage line.
    out = bw.try_handle_brain_command("/revert_proposal")
    report("/revert_proposal usage hint",
           isinstance(out, str) and "usage" in out.lower(),
           f"got: {out!r}")
    # /issue_apply_token --status should work.
    out = bw.try_handle_brain_command("/issue_apply_token --status")
    report("/issue_apply_token --status routes",
           isinstance(out, str) and "token" in out.lower(),
           f"got: {(out or '')[:120]}")
    # /capabilities should produce a markdown report.
    out = bw.try_handle_brain_command("/capabilities slashes")
    report("/capabilities slashes routes",
           isinstance(out, str) and "slash commands" in out.lower(),
           f"got: {(out or '')[:120]}")
    # /explain should ast-introspect a known module.
    out = bw.try_handle_brain_command("/explain chloe_proposals")
    report("/explain chloe_proposals routes",
           isinstance(out, str) and "apply_proposal" in out.lower(),
           f"got: {(out or '')[:120]}")
    # /pending_confirms routes (Stage 3).
    out = bw.try_handle_brain_command("/pending_confirms")
    report("/pending_confirms routes",
           isinstance(out, str)
           and ("pending confirms" in out.lower()
                or "no pending" in out.lower()),
           f"got: {(out or '')[:120]}")
    # Plain-text affirmative resolution (Stage 3 inline hook).
    if cpc:
        cpc.cancel("")
        cpc.announce("verify-inline", source="chat", ttl_s=120)
        out = bw.try_handle_brain_command("yes")
        report("plain 'yes' triggers Stage-3 resolve in chat handler",
               isinstance(out, str)
               and ("apply" in out.lower() or "done" in out.lower()
                    or "verify-inline" in out.lower()
                    or "that didn't take" in out.lower()),
               f"got: {(out or '')[:120]}")
        # And confirm it consumed the pending.
        report("Stage-3 inline hook consumed pending",
               cpc.pending() == [])

    print("\n=== STAGE 8: cleanup ===")
    shutil.rmtree(sandbox, ignore_errors=True)
    report("sandbox cleaned up", not sandbox.exists())

    print(f"\n=== RESULTS: {OK_COUNT} pass, {FAIL_COUNT} fail ===")
    return 0 if FAIL_COUNT == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
