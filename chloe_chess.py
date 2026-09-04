"""Chess against Chloe — engine, adaptive opponent, and player-style learning.

This is the backend core for the HUD chess panel. It is deliberately
self-contained: it depends only on `python-chess` (pip install chess — pure
Python, no compiled deps) plus the stdlib, so it can be unit-tested without
Chloe's voice/brain stack. The WebSocket layer (jarvis.py) drives it and
broadcasts `state()` to the HUD; persona commentary is generated separately by
the caller (feed `tendencies_summary()` to the LLM) so this module stays pure.

Three responsibilities:

1. **Game state** — `ChessGame` wraps a `chess.Board`, validates the player's
   moves, lets Chloe reply, and reports a JSON-friendly `state()` (FEN, whose
   turn, legal moves, status, result, board grid).

2. **Adaptive opponent** — `select_move()` is a negamax (alpha-beta) search over
   a material + light-positional evaluation, with a difficulty knob that maps to
   search depth and a blunder probability. Difficulty auto-tunes toward ~50%
   results so games stay competitive as Ed improves.

3. **Style learning** — a JSON profile persisted under the brain
   (`<brain>/games/chess_profile.json`) accumulates Ed's opening moves,
   aggression (capture/check rate), blunder rate, average game length, and
   recent results. `record_result()` adapts the difficulty; `tendencies_summary()`
   turns the profile into a one-paragraph read of his style for commentary.

Nothing here imports jarvis/brain_wiring, so `import chloe_chess` is always safe
even if `chess` is missing — `CHESS_AVAILABLE` is False and `ChessGame()` raises
a clear install hint instead of crashing the app at import time.
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from pathlib import Path

try:
    import chess
    CHESS_AVAILABLE = True
except ImportError:  # keep import-safe; the app shouldn't crash without the dep
    chess = None
    CHESS_AVAILABLE = False


# ─── Evaluation ────────────────────────────────────────────────────────────
# Material (centipawns) + a light positional layer (center control, pawn
# advancement, bishop pair). Symmetric by construction — no per-square table
# orientation to get wrong. Plenty for a learning-companion opponent whose
# strength is set by search depth, not eval sophistication.

PIECE_VALUES = {1: 100, 2: 320, 3: 330, 4: 500, 5: 900, 6: 0}  # P N B R Q K
_MATE = 1_000_000

# Central + extended-central squares (computed lazily once chess is available).
_CENTER: set[int] = set()
_EXT_CENTER: set[int] = set()


def _init_squares():
    global _CENTER, _EXT_CENTER
    if _CENTER or not CHESS_AVAILABLE:
        return
    _CENTER = {chess.D4, chess.E4, chess.D5, chess.E5}
    ext = set()
    for f in range(2, 6):       # files c..f
        for r in range(2, 6):   # ranks 3..6
            ext.add(chess.square(f, r))
    _EXT_CENTER = ext - _CENTER


def evaluate(board) -> int:
    """Static evaluation in centipawns, from White's perspective
    (positive = good for White)."""
    if board.is_checkmate():
        # Side to move is mated. Bad for whoever is to move.
        return -_MATE if board.turn == chess.WHITE else _MATE
    if (board.is_stalemate() or board.is_insufficient_material()
            or board.is_seventyfive_moves() or board.is_fivefold_repetition()):
        return 0
    _init_squares()
    score = 0
    bishops = {chess.WHITE: 0, chess.BLACK: 0}
    for sq, piece in board.piece_map().items():
        s = PIECE_VALUES[piece.piece_type]
        pt = piece.piece_type
        if pt in (1, 2, 3):  # pawn / knight / bishop like the center
            if sq in _CENTER:
                s += 30
            elif sq in _EXT_CENTER:
                s += 12
        if pt == 1:  # pawn advancement
            rank = chess.square_rank(sq)
            s += (rank if piece.color == chess.WHITE else 7 - rank) * 5
        if pt == 3:
            bishops[piece.color] += 1
        score += s if piece.color == chess.WHITE else -s
    if bishops[chess.WHITE] >= 2:
        score += 30
    if bishops[chess.BLACK] >= 2:
        score -= 30
    return score


def _ordered_moves(board):
    """Captures first (cheap move ordering for alpha-beta). Promotions count
    as high-value via the captured-piece heuristic."""
    moves = list(board.legal_moves)
    def key(mv):
        if board.is_capture(mv):
            victim = board.piece_at(mv.to_square)
            vval = PIECE_VALUES[victim.piece_type] if victim else 100  # en passant
            attacker = board.piece_at(mv.from_square)
            aval = PIECE_VALUES[attacker.piece_type] if attacker else 0
            return -(vval * 10 - aval)  # MVV-LVA, negated so captures sort first
        return 0
    moves.sort(key=key)
    return moves


def _negamax(board, depth, alpha, beta) -> int:
    """Negamax with alpha-beta. Returns score from the perspective of the
    side to move. Mate scores carry a depth term so sooner mates score higher."""
    if board.is_checkmate():
        return -(_MATE + depth)  # side to move is mated
    if board.is_game_over():
        return 0  # stalemate / draw
    if depth == 0:
        sign = 1 if board.turn == chess.WHITE else -1
        return sign * evaluate(board)
    best = -_MATE * 2
    for mv in _ordered_moves(board):
        board.push(mv)
        val = -_negamax(board, depth - 1, -beta, -alpha)
        board.pop()
        if val > best:
            best = val
        if val > alpha:
            alpha = val
        if alpha >= beta:
            break
    return best


# difficulty -> (search depth, blunder probability)
DIFFICULTY = {
    1: (1, 0.45),
    2: (2, 0.25),
    3: (3, 0.10),
    4: (3, 0.03),
    5: (4, 0.00),
}
MIN_DIFFICULTY, MAX_DIFFICULTY = 1, 5


def _book_key(moves_san: list[str], extra: str | None = None) -> str:
    """Stringify an opening prefix for opening_book lookup. Truncated to
    _BOOK_DEPTH plies. Optionally appends one more SAN move."""
    seq = list(moves_san or [])
    if extra is not None:
        seq.append(extra)
    return "|".join(seq[:_BOOK_DEPTH])


def record_opening(profile: dict, moves_san: list[str], result: str) -> None:
    """Log every prefix of the opening (up to _BOOK_DEPTH plies) with the
    game's result. Recording prefixes — not just the leaf — lets the book
    lookup work at any ply by matching the longest known prefix.
    `result` is from Ed's POV: "win" | "loss" | "draw"."""
    if not moves_san:
        return
    book = profile.setdefault("opening_book", {})
    if not isinstance(book, dict):
        book = {}
        profile["opening_book"] = book
    bump = {"win": "ed_w", "loss": "ed_l", "draw": "ed_d"}.get(result, "ed_d")
    seq = list(moves_san)[:_BOOK_DEPTH]
    # Update prefix entries from length 1 .. len(seq) so any future Chloe
    # decision at any opening ply can find a match.
    for n in range(1, len(seq) + 1):
        key = "|".join(seq[:n])
        entry = book.get(key) or {"games": 0, "ed_w": 0, "ed_l": 0, "ed_d": 0}
        entry["games"] += 1
        entry[bump] += 1
        book[key] = entry


def book_move(profile: dict, board, moves_san: list[str],
              rng: random.Random | None = None):
    """If we're early in the game and have book data, return a legal move that
    historically gives Ed trouble. Returns None when there's no useful signal
    (caller should fall back to engine select_move).

    Scoring: for each candidate next move, compute Ed's loss rate over all
    book entries that START with the resulting prefix, weighted by
    log(sample_size). Pick the highest-scoring move IF its score clears a
    minimum-confidence bar (≥2 games + loss-rate ≥ 0.5 + score > baseline).
    """
    if not CHESS_AVAILABLE or board is None:
        return None
    plies_played = len(moves_san or [])
    if plies_played >= _BOOK_DEPTH:
        return None
    book = (profile or {}).get("opening_book") or {}
    if not book:
        return None
    rng = rng or random.Random()
    import math
    cur_prefix = _book_key(moves_san)
    cur_prefix_parts = cur_prefix.split("|") if cur_prefix else []

    def aggregate(prefix_parts: list[str]) -> tuple[int, int]:
        """(total_games, total_ed_losses) over all book entries whose key
        BEGINS with prefix_parts. Includes the exact-match entry too."""
        if not prefix_parts:
            return (0, 0)
        prefix_str = "|".join(prefix_parts)
        prefix_with_sep = prefix_str + "|"
        g = l_ = 0
        for k, v in book.items():
            if k == prefix_str or k.startswith(prefix_with_sep):
                g += int(v.get("games", 0))
                l_ += int(v.get("ed_l", 0))
        return (g, l_)

    candidates = []
    for mv in board.legal_moves:
        try:
            san = board.san(mv)
        except Exception:
            continue
        next_parts = cur_prefix_parts + [san] if cur_prefix_parts else [san]
        g, losses = aggregate(next_parts)
        if g < 2:
            continue
        loss_rate = losses / g
        score = loss_rate * math.log(g + 1)
        candidates.append((mv, score, loss_rate, g))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_mv, best_score, best_lr, best_g = candidates[0]
    # Confidence gate so a single fluke game can't dictate book moves.
    if best_lr < 0.5 or best_score < 0.35:
        return None
    # Slight randomness among top-tier candidates (within 90% of best score)
    # so games don't become deterministic once a winning line is known.
    top = [c for c in candidates if c[1] >= best_score * 0.9]
    return rng.choice([c[0] for c in top])


def _move_features(board, mv) -> tuple[bool, bool]:
    """(is_capture, is_check). Cheap — no full push/pop where avoidable."""
    is_cap = False
    try:
        is_cap = board.is_capture(mv)
    except Exception:
        pass
    is_chk = False
    try:
        board.push(mv)
        is_chk = board.is_check()
        board.pop()
    except Exception:
        is_chk = False
    return (is_cap, is_chk)


def _style_bias(profile: dict | None) -> str:
    """Coarse playstyle classification from the profile — used to bias move
    selection. 'positional' | 'aggressive' | 'balanced' | 'unknown'."""
    if not profile:
        return "unknown"
    plies = profile.get("player_plies", 0)
    if plies < 40:               # too little data to act on
        return "unknown"
    caps = profile.get("player_captures", 0)
    chks = profile.get("player_checks", 0)
    cap_rate = caps / max(plies, 1)
    chk_rate = chks / max(plies, 1)
    if cap_rate > 0.18 or chk_rate > 0.08:
        return "aggressive"
    if cap_rate < 0.10:
        return "positional"
    return "balanced"


# Eval tolerance (in centipawns) for what counts as a "near-best" move worth
# style-biasing among. Tight enough that bias never costs Chloe real material.
_STYLE_BIAS_TOL = 50


def select_move(board, difficulty: int = 2, rng: random.Random | None = None,
                profile: dict | None = None, moves_san: list[str] | None = None):
    """Pick Chloe's move at the given difficulty.

    Returns a chess.Move or None (no legal moves). Three layers:

    1. **Opening book** — if `profile` carries an opening_book with enough
       history at this ply, return a move from lines Ed has lost in. Skipped
       past _BOOK_DEPTH plies or when the data isn't confident.
    2. **Blunder** — with the difficulty's blunder probability, return a
       random legal move (keeps low difficulties beatable).
    3. **Engine + style bias** — negamax/alpha-beta. Among moves within
       _STYLE_BIAS_TOL of the best eval, pick weighted-random with a bias
       AGAINST Ed's known style: positional player → favor captures/checks;
       aggressive player → favor quiet moves. With no profile / not enough
       data, falls back to the original best-move pick.
    """
    rng = rng or random.Random()
    difficulty = max(MIN_DIFFICULTY, min(MAX_DIFFICULTY, int(difficulty)))
    moves = list(board.legal_moves)
    if not moves:
        return None

    # 1. Opening book — strongest signal early in the game.
    if profile is not None and (moves_san is not None):
        try:
            bm = book_move(profile, board, moves_san, rng=rng)
            if bm is not None:
                return bm
        except Exception:
            pass

    depth, blunder = DIFFICULTY[difficulty]

    # 2. Blunder probability.
    if rng.random() < blunder:
        return rng.choice(moves)

    # 3. Engine search → near-best set → style-biased pick.
    best_val = -_MATE * 4
    scored = []  # (val, move)
    for mv in _ordered_moves(board):
        board.push(mv)
        val = -_negamax(board, depth - 1, -_MATE * 4, _MATE * 4)
        board.pop()
        scored.append((val, mv))
        if val > best_val:
            best_val = val
    if not scored:
        return rng.choice(moves)

    style = _style_bias(profile)
    # Within tolerance of the best eval.
    near = [(v, m) for (v, m) in scored if (best_val - v) <= _STYLE_BIAS_TOL]
    if not near:
        # No move qualified (shouldn't happen since best is itself within 0cp)
        best_only = [m for (v, m) in scored if v == best_val]
        return rng.choice(best_only) if best_only else rng.choice(moves)

    # No bias known yet → original behaviour: pick from exact-best ties.
    if style in ("unknown", "balanced"):
        best_only = [m for (v, m) in near if v == best_val]
        return rng.choice(best_only) if best_only else rng.choice(moves)

    # Apply style bias to the near-best set.
    weights = []
    moves_near = []
    for v, m in near:
        is_cap, is_chk = _move_features(board, m)
        w = 1.0
        # Favor eval-best slightly so bias never overrides a clear gain.
        w *= 1.0 + 0.5 * (1.0 - (best_val - v) / max(_STYLE_BIAS_TOL, 1))
        if style == "positional":
            # Push the position open: reward complications.
            if is_cap or is_chk:
                w *= 2.2
        elif style == "aggressive":
            # Starve the trader: reward quiet positional moves.
            if not (is_cap or is_chk):
                w *= 2.2
        weights.append(w)
        moves_near.append(m)
    if not moves_near:
        return rng.choice(moves)
    return rng.choices(moves_near, weights=weights, k=1)[0]


# ─── Style profile (persisted under the brain) ──────────────────────────────

def _profile_path() -> Path:
    root = os.environ.get("CHLOE_BRAIN_ROOT", r"C:\Chloe\brain")
    return Path(root) / "games" / "chess_profile.json"


DEFAULT_PROFILE = {
    "games_played": 0,
    "player_wins": 0,
    "player_losses": 0,
    "draws": 0,
    "difficulty": 2,            # current adaptive difficulty
    "recent_results": [],       # last ~10: "win"/"loss"/"draw" from Ed's POV
    "opening_moves": {},        # Ed's first-move SAN -> count
    "player_plies": 0,          # total half-moves Ed has played
    "player_captures": 0,
    "player_checks": 0,
    "total_game_plies": 0,      # for average game length
    # opening_book: "|".join(first N SAN moves) -> aggregates from Ed's POV.
    # Updated on every game end with the actual opening played; consumed by
    # `book_move` so Chloe can steer her own opening choice toward lines Ed
    # has historically lost in.
    "opening_book": {},         # { "e4|c5|Nf3": {"games":3,"ed_w":0,"ed_l":2,"ed_d":1} }
}

# How many plies of the opening to track / consult.
_BOOK_DEPTH = 8


def load_profile() -> dict:
    """Load the style profile, filling any missing keys with defaults. Never
    raises — a corrupt/absent file yields a fresh default profile."""
    p = _profile_path()
    data = {}
    try:
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    merged = dict(DEFAULT_PROFILE)
    merged.update({k: v for k, v in data.items() if k in DEFAULT_PROFILE})
    # opening_moves + opening_book must be dicts
    if not isinstance(merged.get("opening_moves"), dict):
        merged["opening_moves"] = {}
    if not isinstance(merged.get("opening_book"), dict):
        merged["opening_book"] = {}
    return merged


def save_profile(profile: dict) -> bool:
    """Persist the profile. Never raises; returns success."""
    try:
        p = _profile_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(profile, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False


def observe_player_move(profile: dict, board_before, move) -> None:
    """Accumulate style stats from one of Ed's moves (call BEFORE pushing it,
    with the board in its pre-move state). Mutates `profile` in place."""
    try:
        profile["player_plies"] = profile.get("player_plies", 0) + 1
        if board_before.is_capture(move):
            profile["player_captures"] = profile.get("player_captures", 0) + 1
        # First move of the game (ply 0 for Ed) — record opening preference.
        if board_before.fullmove_number == 1 and board_before.turn == board_before.turn:
            try:
                san = board_before.san(move)
                if board_before.ply() <= 1:
                    om = profile.setdefault("opening_moves", {})
                    om[san] = om.get(san, 0) + 1
            except Exception:
                pass
        board_before.push(move)
        if board_before.is_check():
            profile["player_checks"] = profile.get("player_checks", 0) + 1
        board_before.pop()
    except Exception:
        pass


def record_result(profile: dict, result: str, game_plies: int = 0) -> dict:
    """Update aggregate stats after a finished game and adapt the difficulty.

    `result` is from Ed's perspective: "win" | "loss" | "draw". Adaptive rule:
    over the last few results, if Ed is winning >60% bump difficulty, if <40%
    drop it — keeping games close. Mutates and returns `profile`."""
    profile["games_played"] = profile.get("games_played", 0) + 1
    if result == "win":
        profile["player_wins"] = profile.get("player_wins", 0) + 1
    elif result == "loss":
        profile["player_losses"] = profile.get("player_losses", 0) + 1
    else:
        profile["draws"] = profile.get("draws", 0) + 1
    if game_plies:
        profile["total_game_plies"] = profile.get("total_game_plies", 0) + game_plies

    recent = list(profile.get("recent_results", []))
    recent.append(result)
    recent = recent[-10:]
    profile["recent_results"] = recent

    # Adapt only once we have a small sample.
    if len(recent) >= 3:
        wins = recent.count("win")
        rate = wins / len(recent)
        diff = int(profile.get("difficulty", 2))
        if rate > 0.6 and diff < MAX_DIFFICULTY:
            profile["difficulty"] = diff + 1
        elif rate < 0.4 and diff > MIN_DIFFICULTY:
            profile["difficulty"] = diff - 1
    return profile


def tendencies_summary(profile: dict) -> str:
    """A one-paragraph, plain-language read of Ed's chess style — feed this to
    the LLM for in-persona commentary, or show it in the HUD."""
    g = profile.get("games_played", 0)
    if not g:
        return "No games on record yet — first one sets the baseline."
    w = profile.get("player_wins", 0)
    l = profile.get("player_losses", 0)
    d = profile.get("draws", 0)
    plies = max(1, profile.get("player_plies", 0))
    cap_rate = profile.get("player_captures", 0) / plies
    chk_rate = profile.get("player_checks", 0) / plies
    om = profile.get("opening_moves", {})
    fav_open = max(om, key=om.get) if om else None
    avg_len = (profile.get("total_game_plies", 0) // g) if g else 0
    aggression = ("aggressive" if cap_rate > 0.18 or chk_rate > 0.08
                  else "positional" if cap_rate < 0.10 else "balanced")
    parts = [
        f"{g} game(s): {w}W/{l}L/{d}D.",
        f"Plays a {aggression} style (capture rate {cap_rate:.0%}, "
        f"check rate {chk_rate:.0%}).",
    ]
    if fav_open:
        parts.append(f"Favorite opening move: {fav_open}.")
    if avg_len:
        parts.append(f"Averages ~{avg_len // 2} full moves per game.")
    parts.append(f"Current difficulty: {profile.get('difficulty', 2)}/5.")
    # Surface a few opening lines Ed has historically lost in, so commentary
    # can taunt about them. Skip if there's no clear pattern.
    book = profile.get("opening_book") or {}
    trouble = []
    for k, v in book.items():
        g = int(v.get("games", 0))
        if g < 2:
            continue
        if int(v.get("ed_l", 0)) / max(g, 1) >= 0.5:
            depth = k.count("|") + 1
            if depth in (2, 3, 4):    # short, recognisable lines
                trouble.append((g, k))
    if trouble:
        trouble.sort(reverse=True)
        worst = trouble[0][1].replace("|", " ")
        parts.append(f"Trouble line for Ed: {worst}.")
    return " ".join(parts)


# ─── Game wrapper ────────────────────────────────────────────────────────────

_UNICODE = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
}


class ChessGame:
    """One game of chess between Ed and Chloe. Drives the board, validates Ed's
    moves, lets Chloe reply at the profile's adaptive difficulty, and records
    style stats. `state()` is the JSON-friendly view the HUD renders."""

    def __init__(self, player_white: bool = True, profile: dict | None = None,
                 difficulty: int | None = None, rng: random.Random | None = None):
        if not CHESS_AVAILABLE:
            raise RuntimeError(
                "python-chess is not installed. Run: pip install chess")
        self.board = chess.Board()
        self.player_color = chess.WHITE if player_white else chess.BLACK
        self.profile = profile if profile is not None else load_profile()
        self.difficulty = int(difficulty if difficulty is not None
                              else self.profile.get("difficulty", 2))
        self.rng = rng or random.Random()
        self.moves_san: list[str] = []
        self._result_recorded = False

    # -- helpers ----------------------------------------------------------
    @property
    def player_turn(self) -> bool:
        return self.board.turn == self.player_color

    def _status(self) -> str:
        b = self.board
        if b.is_checkmate():
            return "checkmate"
        if b.is_stalemate():
            return "stalemate"
        if b.is_insufficient_material():
            return "draw_material"
        if b.can_claim_threefold_repetition():
            return "draw_repetition"
        if b.is_seventyfive_moves() or b.can_claim_fifty_moves():
            return "draw_fifty"
        if b.is_check():
            return "check"
        return "ongoing"

    def _result(self) -> str | None:
        """'win' | 'loss' | 'draw' from Ed's POV, or None if ongoing."""
        b = self.board
        if not b.is_game_over(claim_draw=True):
            return None
        if b.is_checkmate():
            # The side to move is mated -> the OTHER side won.
            winner = not b.turn
            return "win" if winner == self.player_color else "loss"
        return "draw"

    def _grid(self) -> list[list[str]]:
        """8x8 grid (rank 8 at index 0 -> rank 1 at index 7), each cell '' or a
        unicode glyph. Convenient for the HUD to render directly."""
        grid = []
        for rank in range(7, -1, -1):
            row = []
            for file in range(8):
                piece = self.board.piece_at(chess.square(file, rank))
                row.append(_UNICODE[piece.symbol()] if piece else "")
            grid.append(row)
        return grid

    # -- actions ----------------------------------------------------------
    def player_move(self, move_str: str) -> dict:
        """Apply Ed's move (UCI like 'e2e4' or SAN like 'Nf3'). Returns
        {ok, error?, state}. Records style stats on success."""
        if self.board.is_game_over(claim_draw=True):
            return {"ok": False, "error": "game over", "state": self.state()}
        if not self.player_turn:
            return {"ok": False, "error": "not your turn", "state": self.state()}
        mv = None
        try:
            mv = self.board.parse_uci(move_str)
        except Exception:
            try:
                mv = self.board.parse_san(move_str)
            except Exception:
                mv = None
        if mv is None or mv not in self.board.legal_moves:
            return {"ok": False, "error": f"illegal move: {move_str}",
                    "state": self.state()}
        observe_player_move(self.profile, self.board, mv)
        self.moves_san.append(self.board.san(mv))
        self.board.push(mv)
        self._maybe_record_result()
        return {"ok": True, "state": self.state()}

    def chloe_move(self) -> dict:
        """Let Chloe pick and play a move. Returns {ok, move?, state}."""
        if self.board.is_game_over(claim_draw=True):
            return {"ok": False, "error": "game over", "state": self.state()}
        if self.player_turn:
            return {"ok": False, "error": "not chloe's turn",
                    "state": self.state()}
        mv = select_move(self.board, self.difficulty, self.rng,
                         profile=self.profile, moves_san=self.moves_san)
        if mv is None:
            return {"ok": False, "error": "no legal moves",
                    "state": self.state()}
        san = self.board.san(mv)
        self.moves_san.append(san)
        self.board.push(mv)
        self._maybe_record_result()
        return {"ok": True, "move": {"uci": mv.uci(), "san": san},
                "state": self.state()}

    def resign(self) -> dict:
        """Ed resigns — record a loss and end the game."""
        if not self._result_recorded:
            record_result(self.profile, "loss", game_plies=self.board.ply())
            record_opening(self.profile, self.moves_san, "loss")
            save_profile(self.profile)
            self._result_recorded = True
        st = self.state()
        st["status"] = "resigned"
        st["result"] = "loss"
        return {"ok": True, "state": st}

    def _maybe_record_result(self):
        if self._result_recorded:
            return
        res = self._result()
        if res is not None:
            record_result(self.profile, res, game_plies=self.board.ply())
            record_opening(self.profile, self.moves_san, res)
            save_profile(self.profile)
            self._result_recorded = True

    def state(self) -> dict:
        result = self._result()
        return {
            "fen": self.board.fen(),
            "grid": self._grid(),
            "player_color": "white" if self.player_color == chess.WHITE else "black",
            "turn": "player" if self.player_turn else "chloe",
            "legal_moves": sorted(m.uci() for m in self.board.legal_moves)
                           if self.player_turn else [],
            "last_move": self.board.peek().uci() if self.board.move_stack else None,
            "moves_san": list(self.moves_san),
            "status": self._status(),
            "result": result,
            "in_check": self.board.is_check(),
            "difficulty": self.difficulty,
            "eval_cp": evaluate(self.board),
            "game_over": result is not None,
        }
