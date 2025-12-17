"""
morpho_vtb_tokenizer.py

Etymology-aware morphological tokenizer using a Viterbi-style DP segmenter.

OPTION 1 IMPLEMENTATION:
- Adds a root “variant expansion” pass so that lexicon entries like:
    "laryngologist", "laryngological", "laryngology", ...
  contribute reusable sub-roots like:
    "laryng", "laryngo", "logist", "logy", "ology", ...
  (when detectable by simple patterns)

This helps segment words like:
  otolaryngologist  -> oto + laryng + ologist
  otorhinolaryngologist -> oto + rhino + laryng + ologist

This version keeps the 512-token limit OUT of this module (that belongs in the caller).

Tunable adjustments:
- short1_adjust  (≈ [-100, +100]): additive weight for 1-character roots
- short2_adjust  (≈ [-100, +100]): additive weight for 2-character roots
- mid_root_adjust (≈ [-100, +100]): additive weight for mid-length roots (e.g., 4–9 chars)
- long_root_adjust (≈ [-100, +100]): per-character adjustment for length > LONG_ROOT_THRESHOLD (default 7)

These are applied on top of a base root weight derived from frequency and length.

The goal is to favor analyses like:
    hyperparathyroidism -> hyper + para + thyroid + ism
rather than many 1–2 character fragments, while still letting you tune behavior
without editing this module again.
"""

from __future__ import annotations

import json
import gzip
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Set


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class RootLexicon:
    roots: Set[str]
    prefixes: Set[str]
    suffixes: Set[str]

    @staticmethod
    def empty() -> "RootLexicon":
        return RootLexicon(set(), set(), set())


@dataclass
class Piece:
    text: str
    kind: str   # 'root' | 'suffix' | 'prefix' | 'unk'
    score: float


@dataclass
class _Step:
    score: float
    back: Optional[int]
    piece: Optional[Piece]


# -----------------------------
# MorphoTokenizer
# -----------------------------

class MorphoTokenizer:
    """Viterbi-based morphological tokenizer using Wiktionary-derived lexicons."""

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
        short1_adjust: float = -12.0,  # NEW: strong penalty for 1-char roots by default
        short2_adjust: float = -5.0,   # penalty/bonus for 2-char roots
        mid_root_adjust: float = +2.5, # bonus/penalty for mid-length roots (4–9 chars)
        long_root_adjust: float = -0.5 # per-char adjustment for length > LONG_ROOT_THRESHOLD
    ) -> None:
        # --- Load surface root lexicon ---
        if lexicon_json is None:
            raise ValueError("lexicon_json must be provided (path or dict).")

        if isinstance(lexicon_json, (str, Path)):
            root_dict = self._load_lexicon_dict(Path(lexicon_json))
        elif isinstance(lexicon_json, dict):
            root_dict = lexicon_json
        else:
            raise TypeError("lexicon_json must be a path or dict.")

        # OPTION 1: expand root lexicon with inferred sub-roots / variants
        root_dict = self._expand_root_dict_with_variants(root_dict)

        self.lexicon_dict: Dict[str, dict] = root_dict
        self.lex = RootLexicon.empty()

        for r in root_dict.keys():
            if isinstance(r, str) and r.strip():
                self.lex.roots.add(r.strip().lower())

        # Optional generic suffixes
        if add_generic_suffixes:
            generic_suffixes = {
                "s", "es", "ed", "ing",
                "ity", "ive", "ize", "ise",
                "ia", "ism", "ist", "al", "ic", "ous", "ary", "ory", "ship",
                "tion", "sion", "ment", "ium", "logy", "ology",
                "ologist", "ological", "ologists",
            }
            self.lex.suffixes |= generic_suffixes

        # Weight tables
        self.root_weight: Dict[str, float] = {}
        self.prefix_weight: Dict[str, float] = {}
        self.suffix_weight: Dict[str, float] = {}

        # Root weights based on NumSourceWords and length (BASE score only)
        #
        # Final score used in _score_piece = base_weight + adjustments
        # where adjustments depend on:
        #   - length == 1          -> short1_adjust
        #   - length == 2          -> short2_adjust
        #   - 4 <= length <= 9     -> mid_root_adjust
        #   - length > 7           -> long_root_adjust * (length - 7)
        #
        for r, info in root_dict.items():
            if not isinstance(r, str) or not r.strip():
                continue
            key = r.strip().lower()
            freq_raw = info.get("NumSourceWords", 1)
            try:
                freq = int(freq_raw)
            except Exception:
                freq = 1

            L = len(key)
            # Base weight: frequency + length (mildly pro-longer roots)
            base = 3.0 + math.log1p(max(freq, 1)) + 0.3 * min(L, 8)
            self.root_weight[key] = float(base)

        # Simple heuristic suffix weights
        for s in self.lex.suffixes:
            self.suffix_weight[s] = 3.0 + 0.2 * min(len(s), 10)

        # Prefix weights (empty by default, but you can populate self.lex.prefixes manually)
        for p in self.lex.prefixes:
            self.prefix_weight[p] = 2.8 + 0.2 * min(len(p), 10)

        # DP / scoring config
        self.UNK_BASE_PENALTY = float(unk_base_penalty)
        self.UNK_PER_CHAR = float(unk_per_char)
        self.MAX_MORPHEME_LEN = int(max_morpheme_len)

        # Global DP behavior
        self.LAMBDA_PENALTY = float(lambda_penalty)
        self.LONG_UNSPLIT_MIN_LEN = int(long_unsplit_min_len)
        self.LONG_UNSPLIT_PENALTY = float(long_unsplit_penalty)

        # Length-threshold for "long" roots
        self.LONG_ROOT_THRESHOLD = 7

        # NEW: user-tunable adjustments
        self.SHORT1_ADJUST = float(short1_adjust)
        self.SHORT2_ADJUST = float(short2_adjust)
        self.MID_ROOT_ADJUST = float(mid_root_adjust)
        self.LONG_ROOT_ADJUST = float(long_root_adjust)

        # Regex for text tokenization
        self._token_split_re = re.compile(r"(\w+|[^\w\s]+)", re.UNICODE)

        # Memoization cache for segment_word_dp
        self._cache: Dict[str, List[Piece]] = {}

        # --- Load ProtoRoot lexicon (optional) ---
        if proto_lexicon_json is None:
            proto_dict: Dict[str, list] = {}
        elif isinstance(proto_lexicon_json, (str, Path)):
            proto_dict = self._load_proto_dict(Path(proto_lexicon_json))
        elif isinstance(proto_lexicon_json, dict):
            proto_dict = proto_lexicon_json
        else:
            raise TypeError("proto_lexicon_json must be a path, dict, or None.")

        self.proto_lexicon_dict: Dict[str, list] = proto_dict
        self.root_to_proto: Dict[str, List[str]] = self._build_root_to_proto_map(proto_dict)

    # -----------------------------
    # Lexicon loading
    # -----------------------------

    def _load_lexicon_dict(self, path: Path) -> Dict[str, dict]:
        """Load root lexicon from .json or .json.gz."""
        if not path.exists():
            raise FileNotFoundError(f"Root lexicon not found: {path}")
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        else:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)

    def _load_proto_dict(self, path: Path) -> Dict[str, list]:
        """Load proto-root lexicon from .json or .json.gz."""
        if not path.exists():
            raise FileNotFoundError(f"Proto-root lexicon not found: {path}")
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        else:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)

    def _build_root_to_proto_map(self, proto_dict: Dict[str, list]) -> Dict[str, List[str]]:
        """Build mapping from surface root (lowercased) -> list of ProtoRoot keys."""
        mapping: Dict[str, List[str]] = {}
        for proto, entries in proto_dict.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                root = entry.get("Root") if isinstance(entry, dict) else None
                if not isinstance(root, str) or not root.strip():
                    continue
                key = root.strip().lower()
                mapping.setdefault(key, []).append(proto)
        return mapping

    # -----------------------------
    # OPTION 1: root variant expansion
    # -----------------------------

    def _expand_root_dict_with_variants(self, root_dict: Dict[str, dict]) -> Dict[str, dict]:
        """
        Build a new root_dict that includes:
          - original keys
          - inferred reusable roots based on common biomedical affix patterns

        This is conservative by design: it avoids adding tons of junk, but it will
        add helpful roots like:
          laryng   from laryngologist / laryngology / laryngectomy
          laryngo  from laryngologist / laryngology / laryngological
          ologist  from laryngologist, otolaryngologist, etc.
          logy / ology / logical / ological (as roots/suffix-like roots)
        """
        out: Dict[str, dict] = dict(root_dict)

        def bump(info: dict, extra_sources: int = 1) -> dict:
            # shallow copy and bump NumSourceWords a bit so derived roots aren't too weak
            base = dict(info) if isinstance(info, dict) else {}
            nsw = base.get("NumSourceWords", 1)
            try:
                nsw_i = int(nsw)
            except Exception:
                nsw_i = 1
            base["NumSourceWords"] = max(1, nsw_i) + extra_sources
            return base

        def add_root(key: str, base_info: dict, extra_sources: int = 1):
            k = key.strip().lower()
            if not k:
                return
            # do not overwrite a stronger existing entry
            if k not in out:
                out[k] = bump(base_info, extra_sources=extra_sources)

        # patterns for splitting common medical/Greek-Latin constructions
        suffixes = [
            "ologist", "ology", "ological", "logist", "logy", "logical",
            "ectomy", "itis", "emia", "opathy", "phobia", "plasia", "tomy",
        ]

        # For every word in the lexicon, infer root candidates by stripping known suffixes
        for word, info in root_dict.items():
            if not isinstance(word, str):
                continue
            w = word.strip().lower()
            if len(w) < 6:
                continue

            # Strip hyphens and keep a variant too
            w2 = w.replace("-", "")
            if w2 != w:
                add_root(w2, info, extra_sources=1)

            for suf in suffixes:
                if w.endswith(suf) and len(w) > len(suf) + 2:
                    stem = w[: -len(suf)]
                    add_root(stem, info, extra_sources=2)

                    # also add stem+"o" for "laryng"+"o" => "laryngo" style
                    if not stem.endswith("o") and len(stem) >= 3:
                        add_root(stem + "o", info, extra_sources=2)

                    # add the suffix itself as a root candidate too (if useful)
                    add_root(suf, info, extra_sources=2)

        # Also explicitly add a few extremely common biomedical components
        # (helps even when the lexicon only contains long surface forms)
        common_components = [
            "oto", "rhino", "laryng", "laryngo", "logist", "ologist", "logy", "ology",
        ]
        for cc in common_components:
            if cc not in out:
                out[cc] = {"NumSourceWords": 3}

        return out

    # -----------------------------
    # Piece scoring
    # -----------------------------

    def _score_piece(self, p: str) -> Piece:
        """Score a candidate substring p (lowercased)."""
        if p in self.root_weight:
            score = self.root_weight[p]
            L = len(p)

            # (0) One-character roots: strong penalty knob
            if L == 1:
                score += self.SHORT1_ADJUST

            # (1) Two-character roots: global knob SHORT2_ADJUST
            if L == 2:
                score += self.SHORT2_ADJUST

            # (2) Mid-length roots (e.g., biomedical terms): global knob MID_ROOT_ADJUST
            if 4 <= L <= 9:
                score += self.MID_ROOT_ADJUST

            # (3) Very long roots: penalize or reward beyond length threshold
            if L > self.LONG_ROOT_THRESHOLD:
                score += self.LONG_ROOT_ADJUST * (L - self.LONG_ROOT_THRESHOLD)

            return Piece(p, "root", score)

        if p in self.suffix_weight:
            return Piece(p, "suffix", self.suffix_weight[p])
        if p in self.prefix_weight:
            return Piece(p, "prefix", self.prefix_weight[p])

        # Unknown chunk
        return Piece(p, "unk", self.UNK_BASE_PENALTY + self.UNK_PER_CHAR * len(p))

    # -----------------------------
    # Viterbi DP segmenter
    # -----------------------------

    def _segment_word_dp(self, word: str) -> List[Piece]:
        if word in self._cache:
            return self._cache[word]

        w = word.lower()
        n = len(w)

        if n == 0:
            self._cache[word] = []
            return []

        dp: List[_Step] = [
            _Step(score=float("-inf"), back=None, piece=None) for _ in range(n + 1)
        ]
        dp[0] = _Step(score=0.0, back=None, piece=None)

        for i in range(n):
            if dp[i].score == float("-inf"):
                continue
            max_j = min(n, i + self.MAX_MORPHEME_LEN)
            for j in range(i + 1, max_j + 1):
                sub = w[i:j]
                pc = self._score_piece(sub)

                # Base score: previous DP state + piece score, minus a per-piece λ penalty.
                sc = dp[i].score + pc.score - self.LAMBDA_PENALTY

                # Extra penalty if we keep the entire long word as a single piece.
                if i == 0 and j == n and n >= self.LONG_UNSPLIT_MIN_LEN:
                    sc -= self.LONG_UNSPLIT_PENALTY

                if sc > dp[j].score:
                    dp[j] = _Step(score=sc, back=i, piece=pc)

        if dp[n].score == float("-inf"):
            out = [Piece(word, "unk", 0.0)]
            self._cache[word] = out
            return out

        out: List[Piece] = []
        idx = n
        while idx > 0:
            st = dp[idx]
            if st.back is None or st.piece is None:
                out = [Piece(word, "unk", 0.0)]
                break
            out.append(st.piece)
            idx = st.back

        out.reverse()
        self._cache[word] = out
        return out

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

    # -----------------------------
    # ProtoRoot helpers
    # -----------------------------

    def get_proto_roots_for(self, root: str) -> List[str]:
        """Get ProtoRoot IDs for a surface root (case-insensitive)."""
        return self.root_to_proto.get(root.lower(), [])

    def get_proto_roots_for_piece(self, piece: Piece) -> List[str]:
        """Convenience helper to get ProtoRoots for a root Piece."""
        if piece.kind != "root":
            return []
        return self.get_proto_roots_for(piece.text.lower())

    # -----------------------------
    # Visualization helpers
    # -----------------------------

    def visualize_word(self, word: str, show_proto: bool = False) -> None:
        """Print a human-readable segmentation of a word."""
        pieces = self.segment_word(word)
        parts = []
        for p in pieces:
            label = p.kind
            if show_proto and p.kind == "root":
                protos = self.get_proto_roots_for_piece(p)
                if protos:
                    label += ":" + ",".join(protos[:3])
            parts.append(f"{p.text}[{label}]")
        print(word, "->", " + ".join(parts))
