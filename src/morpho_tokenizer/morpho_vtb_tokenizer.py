"""
morpho_vtb_tokenizer.py

Etymology-aware morphological tokenizer (dynamic programming segmentation).

Key idea:
- Load a surface root lexicon (root -> info dict) and an optional proto-root lexicon.
- Score candidate substrings as "roots" if present in the lexicon; otherwise treat as unknown pieces.
- Use DP to choose a segmentation that maximizes:
    sum(piece_scores) - lambda_penalty * (#pieces)
  plus optional heuristics/adjustments.

IMPORTANT: This module does NOT enforce model-specific limits (e.g., 512 token limit).
Put any model max-length logic in the calling program.

Tunable knobs (recommended range for adjustments: [-100, +100]):
- short1_adjust (≈ [-100, +100]): additive weight for 1-character roots
- short2_adjust  (≈ [-100, +100]): additive weight for 2-character roots
- mid_root_adjust (≈ [-100, +100]): additive weight for mid-length roots (4–9 chars)
- long_root_adjust (≈ [-100, +100]): per-char adjustment when length > LONG_ROOT_THRESHOLD
"""

from __future__ import annotations

import gzip
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Set


# -----------------------------
# Data structures
# -----------------------------

@dataclass(frozen=True)
class RootLexicon:
    roots: Set[str]
    proto_roots: Set[str]
    suffixes: Set[str]

    @staticmethod
    def empty() -> "RootLexicon":
        return RootLexicon(set(), set(), set())


@dataclass
class Piece:
    text: str
    kind: str   # "root" | "unk" | "suf" | etc.
    score: float


@dataclass
class _Step:
    score: float
    back: Optional[int]
    piece: Optional[Piece]


# -----------------------------
# Tokenizer
# -----------------------------

LONG_ROOT_THRESHOLD = 7


class MorphoTokenizer:
    def __init__(
        self,
        lexicon_json: Union[str, Path, Dict[str, dict], None] = None,
        proto_lexicon_json: Union[str, Path, Dict[str, list], None] = None,
        max_morpheme_len: int = 12,
        unk_base_penalty: float = -2.0,
        unk_per_char: float = -0.2,
        add_generic_suffixes: bool = True,
        lambda_penalty: float = 4.5,
        long_unsplit_min_len: int = 5,
        long_unsplit_penalty: float = 3.0,
        # NEW: tunable adjustments (recommend range [-100, +100])
        short1_adjust: float = -8.0,   # penalty/bonus for 1-char roots
        short2_adjust: float = -5.0,   # penalty/bonus for 2-char roots
        mid_root_adjust: float = +2.5, # bonus/penalty for mid-length roots (4–9 chars)
        long_root_adjust: float = -0.5 # per-char adjustment for length > LONG_ROOT_THRESHOLD
    ) -> None:
        # --- Load surface root lexicon ---
        if lexicon_json is None:
            raise ValueError("lexicon_json must be provided (path or dict).")

        if isinstance(lexicon_json, (str, Path)):
            root_dict = self._load_lexicon_dict(Path(lexicon_json))
        else:
            root_dict = lexicon_json

        # --- Load proto-root lexicon (optional but recommended) ---
        proto_dict: Dict[str, list] = {}
        if proto_lexicon_json is not None:
            if isinstance(proto_lexicon_json, (str, Path)):
                proto_dict = self._load_lexicon_dict(Path(proto_lexicon_json))  # type: ignore
            else:
                proto_dict = proto_lexicon_json  # type: ignore

        # --- Build RootLexicon sets ---
        self.lexicon_dict: Dict[str, dict] = root_dict
        self.proto_lexicon_dict: Dict[str, list] = proto_dict

        self.lex = RootLexicon.empty()
        for r in root_dict.keys():
            if isinstance(r, str) and r.strip():
                self.lex.roots.add(r.strip().lower())

        for pr in proto_dict.keys():
            if isinstance(pr, str) and pr.strip():
                self.lex.proto_roots.add(pr.strip().lower())

        # --- Root weights based on frequency/length ---
        self.root_weight: Dict[str, float] = {}
        for key, info in root_dict.items():
            if not isinstance(key, str):
                continue
            k = key.strip().lower()
            if not k:
                continue

            freq_raw = 1
            if isinstance(info, dict):
                freq_raw = info.get("NumSourceWords", 1)

            try:
                freq = int(freq_raw)
            except Exception:
                freq = 1

            # base + log(freq) + small length bonus up to 8 chars
            w = 3.0 + math.log1p(max(freq, 1)) + 0.3 * min(len(k), 8)
            self.root_weight[k] = float(w)

        # --- Parameters ---
        self.max_morpheme_len = int(max_morpheme_len)
        self.unk_base_penalty = float(unk_base_penalty)
        self.unk_per_char = float(unk_per_char)
        self.add_generic_suffixes = bool(add_generic_suffixes)
        self.lambda_penalty = float(lambda_penalty)
        self.long_unsplit_min_len = int(long_unsplit_min_len)
        self.long_unsplit_penalty = float(long_unsplit_penalty)

        self.SHORT1_ADJUST = float(short1_adjust)
        self.SHORT2_ADJUST = float(short2_adjust)
        self.MID_ROOT_ADJUST = float(mid_root_adjust)
        self.LONG_ROOT_ADJUST = float(long_root_adjust)

        # Optional generic suffixes (helps reduce silly splits)
        if self.add_generic_suffixes:
            for suf in ("itis", "osis", "emia", "ology", "ologist", "ectomy", "otomy", "algia", "opathy",
                        "genic", "genesis", "phobia", "philic", "philia", "therapy", "therapeutic",
                        "sclerosis", "plasty", "scopy", "scope", "gram", "graphy"):
                self.lex.suffixes.add(suf)

        # Regex for token splitting in tokenize_and_segment
        self._token_split_re = re.compile(r"[A-Za-z]+|[0-9]+|[^\w\s]+", re.UNICODE)

    # -----------------------------
    # I/O
    # -----------------------------

    def _load_lexicon_dict(self, path: Path) -> Dict:
        if not path.exists():
            raise FileNotFoundError(str(path))
        if path.suffixes[-2:] == [".json", ".gz"]:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        raise ValueError(f"Unsupported lexicon file type: {path.name}")

    # -----------------------------
    # Scoring
    # -----------------------------

    def _score_piece(self, piece: str, is_suffix: bool = False) -> Tuple[float, str]:
        """
        Return (score, kind).
        kind: "root" | "suf" | "unk"
        """
        p = piece.lower()
        L = len(p)

        if is_suffix:
            # suffixes get treated as a known morpheme-like piece
            base = 2.0 + 0.2 * min(L, 8)
            return (base, "suf")

        if p in self.root_weight:
            score = self.root_weight[p]

            # (1) One-character roots: global knob SHORT1_ADJUST
            if L == 1:
                score += self.SHORT1_ADJUST

            # (2) Two-character roots: global knob SHORT2_ADJUST
            if L == 2:
                score += self.SHORT2_ADJUST

            # (3) Mid-length roots (4–9): global knob MID_ROOT_ADJUST
            if 4 <= L <= 9:
                score += self.MID_ROOT_ADJUST

            # (4) Very long roots: apply per-char adjustment beyond threshold
            if L > LONG_ROOT_THRESHOLD:
                score += (L - LONG_ROOT_THRESHOLD) * self.LONG_ROOT_ADJUST

            return (float(score), "root")

        # Unknown piece: base + per-char
        score = self.unk_base_penalty + self.unk_per_char * L
        return (float(score), "unk")

    # -----------------------------
    # DP segmentation
    # -----------------------------

    def _segment_word_dp(self, word: str) -> List[Piece]:
        """
        DP over lowercase word; returns Pieces with lowercase text.
        """
        w = word.lower()
        n = len(w)
        if n == 0:
            return []

        # dp[i] = best score up to position i
        dp: List[_Step] = [_Step(score=-1e18, back=None, piece=None) for _ in range(n + 1)]
        dp[0] = _Step(score=0.0, back=None, piece=None)

        for i in range(n):
            if dp[i].score <= -1e17:
                continue

            # Try suffix match (optional)
            for suf in self.lex.suffixes:
                if w.startswith(suf, i):
                    j = i + len(suf)
                    if j <= n:
                        s, kind = self._score_piece(suf, is_suffix=True)
                        cand = dp[i].score + s - self.lambda_penalty
                        if cand > dp[j].score:
                            dp[j] = _Step(score=cand, back=i, piece=Piece(suf, kind, s))

            # Try root/unknown pieces up to max_morpheme_len
            max_j = min(n, i + self.max_morpheme_len)
            for j in range(i + 1, max_j + 1):
                piece = w[i:j]
                s, kind = self._score_piece(piece, is_suffix=False)
                cand = dp[i].score + s - self.lambda_penalty
                if cand > dp[j].score:
                    dp[j] = _Step(score=cand, back=i, piece=Piece(piece, kind, s))

        # Backtrack
        if dp[n].back is None:
            # Fallback: whole word as unknown
            s, kind = self._score_piece(w, is_suffix=False)
            return [Piece(w, kind, s)]

        pieces: List[Piece] = []
        cur = n
        while cur > 0:
            step = dp[cur]
            if step.piece is None or step.back is None:
                break
            pieces.append(step.piece)
            cur = step.back
        pieces.reverse()

        # Penalize "no split" for long words (encourage splitting)
        if len(pieces) == 1 and len(w) >= self.long_unsplit_min_len:
            pieces[0].score -= self.long_unsplit_penalty

        return pieces

    # -----------------------------
    # Public API
    # -----------------------------

    def segment_word(self, word: str) -> List[Piece]:
        """Segment a single word into morphemic pieces, preserving original casing."""
        lower_pieces = self._segment_word_dp(word)
        out: List[Piece] = []
        pos = 0
        for p in lower_pieces:
            length = len(p.text)
            orig = word[pos:pos + length]
            out.append(Piece(orig, p.kind, p.score))
            pos += length
        return out

    def tokenize_and_segment(self, text: str) -> List[Tuple[str, List[Piece]]]:
        """Split text into tokens and segment alphabetic tokens.

        Returns list of (token, [Piece, ...]).
        """
        tokens = self._token_split_re.findall(text)
        out: List[Tuple[str, List[Piece]]] = []
        for tok in tokens:
            if tok.isalpha():
                out.append((tok, self.segment_word(tok)))
            else:
                out.append((tok, []))
        return out
