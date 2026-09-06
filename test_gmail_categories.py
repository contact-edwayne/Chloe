"""test_gmail_categories.py - Regression test for action item #9 (audit
Part 13, "Generalize Gmail category mapping beyond Primary"): Gmail's
web UI splits the Inbox into five tabs (Primary, Social, Promotions,
Updates, Forums); the code hardcoded "category:primary" for every plain
INBOX read (a real, live-confirmed 2026-09-06 fix -- an unfiltered
search surfaced notification mail Ed's own Primary tab doesn't show),
but had no phrase mapped to the other four at all, so "check my
promotions" was an honest miss with no path forward.

resolve_folder() now recognizes both the original real-IMAP-folder
aliases (Sent, Drafts, Spam, Trash, Starred, Important, All Mail) and
Gmail's own category words, returning a 3-tuple (mailbox, display,
category) instead of the old 2-tuple. This test covers: the new
category resolution itself, that the default (no folder given) is
unchanged from before this feature existed, and that non-INBOX
mailboxes correctly get category=None (categories don't apply there).

email_client.py is safe to import directly (see
test_email_encoding.py's docstring).

Run from the jarvis dir:
    python test_gmail_categories.py
Exit code 0 on success, non-zero on any failure.
"""
import email_client

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


def test_no_folder_given_defaults_to_inbox_primary():
    result = email_client.resolve_folder(None)
    check("resolve_folder(None) still defaults to (INBOX, Inbox, "
          "primary) -- unchanged from before this feature existed",
          result == ("INBOX", "Inbox", "primary"), result)
    result_empty = email_client.resolve_folder("")
    check("resolve_folder('') matches resolve_folder(None)",
          result_empty == ("INBOX", "Inbox", "primary"), result_empty)


def test_plain_inbox_word_resolves_to_primary_category():
    result = email_client.resolve_folder("inbox")
    check("resolve_folder('inbox') resolves to (INBOX, Inbox, primary) "
          "-- same default as omitting folder entirely",
          result == ("INBOX", "Inbox", "primary"), result)


def test_each_new_category_resolves_correctly():
    cases = {
        "social": ("INBOX", "Social", "social"),
        "promotions": ("INBOX", "Promotions", "promotions"),
        "promotional": ("INBOX", "Promotions", "promotions"),
        "promo": ("INBOX", "Promotions", "promotions"),
        "promos": ("INBOX", "Promotions", "promotions"),
        "updates": ("INBOX", "Updates", "updates"),
        "forums": ("INBOX", "Forums", "forums"),
        "forum": ("INBOX", "Forums", "forums"),
        "primary": ("INBOX", "Primary", "primary"),
    }
    for spoken, expected in cases.items():
        result = email_client.resolve_folder(spoken)
        check(f"resolve_folder({spoken!r}) == {expected!r}",
              result == expected, result)


def test_case_insensitive_and_whitespace_tolerant():
    result = email_client.resolve_folder("  Promotions  ")
    check("category words resolve case/whitespace-insensitively, same "
          "as the existing folder aliases",
          result == ("INBOX", "Promotions", "promotions"), result)


def test_real_folders_still_resolve_with_category_none():
    cases = {
        "sent": "[Gmail]/Sent Mail",
        "drafts": "[Gmail]/Drafts",
        "spam": "[Gmail]/Spam",
        "trash": "[Gmail]/Trash",
        "starred": "[Gmail]/Starred",
        "important": "[Gmail]/Important",
        "all mail": "[Gmail]/All Mail",
    }
    for spoken, mailbox in cases.items():
        result = email_client.resolve_folder(spoken)
        check(f"resolve_folder({spoken!r}) still resolves to the "
              f"correct real IMAP mailbox with category=None (Gmail "
              f"categories don't apply outside Inbox)",
              result[0] == mailbox and result[2] is None, result)


def test_unrecognized_word_is_still_an_honest_miss():
    result = email_client.resolve_folder("nonexistent-folder-xyz")
    check("an unrecognized folder/category phrase still returns None "
          "(honest miss), not a guess", result is None, result)


def test_category_word_does_not_shadow_a_real_folder_alias():
    # Sanity check that the two alias dicts don't collide on any key --
    # if they did, one would silently shadow the other depending on
    # check order.
    folder_keys = set(email_client._FOLDER_ALIASES.keys())
    category_keys = set(email_client._GMAIL_CATEGORY_ALIASES.keys())
    overlap = folder_keys & category_keys
    check("no spoken word is ambiguous between a real folder alias and "
          "a Gmail category alias", not overlap, overlap)


def test_display_names_message_mentions_the_new_categories():
    check("the honest-miss error message now tells Ed the Inbox "
          "categories are askable, not just the real folders",
          "Social" in email_client._FOLDER_DISPLAY_NAMES
          and "Promotions" in email_client._FOLDER_DISPLAY_NAMES
          and "Updates" in email_client._FOLDER_DISPLAY_NAMES
          and "Forums" in email_client._FOLDER_DISPLAY_NAMES,
          email_client._FOLDER_DISPLAY_NAMES)


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
    print(f"\n{PASSED} passed, {FAILED} failed")
    raise SystemExit(1 if FAILED else 0)
