"""
Etymology-aware morphological tokenizer using a Viterbi-style DP segmenter.

This version includes "lexicon-induced root backoff":

- Infer additional candidate roots from lexicon entries by:
  * stripping common suffixes (e.g., -ology, -ologist, -itis, -ectomy, -al, -ic, etc.)
  * converting combining forms ending in 'o' (e.g., laryngo -> laryng)
  * optional proto-root projection (adds proto-dict "Root" fields as candidates)

It also adds tunable penalties/bonuses for short roots:
- short1_adjust: adjustment for 1-character roots (default strongly negative)
- short2_adjust: adjustment for 2-character roots (default negative)
- short3_adjust: adjustment for 3-character roots (default negative)

Goal: prefer analyses like
    Otolaryngologist -> oto + laryng + ologist
rather than many 1–3 character fragments.
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

        # Tunable adjustments (recommend range [-100, +100])
        short1_adjust: float = -25.0,  # NEW: 1-char roots (very negative by default)
        short2_adjust: float = -5.0,   # 2-char roots
        short3_adjust: float = -2.5,   # NEW: 3-char roots (mild negative default)

        mid_root_adjust: float = +2.5, # mid-length roots (4–9 chars)
        long_root_adjust: float = -0.5,# per-char adjustment for length > LONG_ROOT_THRESHOLD

        # NEW: lexicon-induced backoff behavior
        infer_roots_from_lexicon: bool = True,
        infer_combining_o: bool = True,          # laryngo -> laryng
        infer_o_prefix_for_logist: bool = True,  # logist -> ologist
        include_proto_roots: bool = True,        # add proto-dict "Root" fields
        inferred_root_penalty: float = -0.8,     # slight penalty vs direct lexicon roots
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

        self.lexicon_dict: Dict[str, dict] = root_dict
        self.lex = RootLexicon.empty()

        # Optional generic suffixes (used only for scoring suffix pieces, not root inference)
        if add_generic_suffixes:
            generic_suffixes = {
                "s", "es", "ed", "ing",
                "ity", "ive", "ize", "ise",
                "ia", "ism", "ist", "al", "ic", "ous", "ary", "ory", "ship",
                "tion", "sion", "ment", "ium", "logy", "ology",
            }
            self.lex.suffixes |= generic_suffixes

        # Weight tables
        self.root_weight: Dict[str, float] = {}
        self.extra_root_weight: Dict[str, float] = {}  # NEW: inferred roots
        self.prefix_weight: Dict[str, float] = {}
        self.suffix_weight: Dict[str, float] = {}

        # Root weights based on NumSourceWords and length (BASE score only)
        for r, info in root_dict.items():
            if not isinstance(r, str) or not r.strip():
                continue
            key = r.strip().lower()

            # keep original lexicon keys as "direct" roots
            self.lex.roots.add(key)

            freq_raw = info.get("NumSourceWords", 1)
            try:
                freq = int(freq_raw)
            except Exception:
                freq = 1

            L = len(key)
            base = 3.0 + math.log1p(max(freq, 1)) + 0.3 * min(L, 8)
            self.root_weight[key] = float(base)

        # Simple heuristic suffix weights
        for s in self.lex.suffixes:
            self.suffix_weight[s] = 3.0 + 0.2 * min(len(s), 6)

        # Prefix weights (empty by default)
        for p in self.lex.prefixes:
            self.prefix_weight[p] = 2.8 + 0.2 * min(len(p), 6)

        # DP / scoring config
        self.UNK_BASE_PENALTY = float(unk_base_penalty)
        self.UNK_PER_CHAR = float(unk_per_char)
        self.MAX_MORPHEME_LEN = int(max_morpheme_len)

        self.LAMBDA_PENALTY = float(lambda_penalty)
        self.LONG_UNSPLIT_MIN_LEN = int(long_unsplit_min_len)
        self.LONG_UNSPLIT_PENALTY = float(long_unsplit_penalty)

        # Length-threshold for "long" roots
        self.LONG_ROOT_THRESHOLD = 7

        # User-tunable adjustments
        self.SHORT1_ADJUST = float(short1_adjust)
        self.SHORT2_ADJUST = float(short2_adjust)
        self.SHORT3_ADJUST = float(short3_adjust)
        self.MID_ROOT_ADJUST = float(mid_root_adjust)
        self.LONG_ROOT_ADJUST = float(long_root_adjust)

        # Backoff config
        self.INFER_ROOTS_FROM_LEXICON = bool(infer_roots_from_lexicon)
        self.INFER_COMBINING_O = bool(infer_combining_o)
        self.INFER_O_PREFIX_FOR_LOGIST = bool(infer_o_prefix_for_logist)
        self.INCLUDE_PROTO_ROOTS = bool(include_proto_roots)
        self.INFERRED_ROOT_PENALTY = float(inferred_root_penalty)

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

        # NEW: augment candidate roots using lexicon-induced backoff
        if self.INFER_ROOTS_FROM_LEXICON or self.INCLUDE_PROTO_ROOTS:
            self._augment_candidate_roots()

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
                root = entry.get("Root")
                if not isinstance(root, str) or not root.strip():
                    continue
                key = root.strip().lower()
                mapping.setdefault(key, []).append(proto)
        return mapping

    # -----------------------------
    # NEW: Lexicon-induced root backoff
    # -----------------------------

    def _augment_candidate_roots(self) -> None:
        """
        Add inferred root candidates into self.lex.roots and self.extra_root_weight.

        We infer from:
          - surface lexicon keys (suffix stripping + combining-form -o removal)
          - proto lexicon entries (Root fields)
        """
        # A compact, defensible list: common biomedical / technical suffixes
        # (single-step stripping only, to avoid exploding the candidate set).
        suffixes = [
            "ologist", "ology", "ological", "logist", "logy", "logical",
            "ectomy", "itis", "emia", "osis", "iasis",
            "al", "ial", "ic", "ical", "ous", "ary", "ory",
            "ism", "ist", "ment", "tion", "sion", "ness",
            "ize", "ise", "ized", "ising", "izing", "ation",
            "able", "ible", "ive", "ity",
        ]

        def add_inferred(root: str, base_from: str) -> None:
            r = root.lower().strip()
            if not r or len(r) < 3:
                return
            if r in self.root_weight or r in self.extra_root_weight:
                return

            # derive a base score from the source token if possible
            src = base_from.lower().strip()
            src_base = self.root_weight.get(src, 3.0 + 0.3 * min(len(src), 8))
            # inferred roots get a small penalty (so direct lexicon roots win ties)
            self.extra_root_weight[r] = float(src_base + self.INFERRED_ROOT_PENALTY)
            self.lex.roots.add(r)

        # 1) From surface lexicon keys
        if self.INFER_ROOTS_FROM_LEXICON:
            for k in list(self.root_weight.keys()):
                key = k.lower()

                # combining form: laryngo -> laryng
                if self.INFER_COMBINING_O and len(key) >= 5 and key.endswith("o"):
                    add_inferred(key[:-1], key)

                # suffix stripping: laryngological -> laryng
                for suf in suffixes:
                    if len(key) > len(suf) + 3 and key.endswith(suf):
                        stem = key[: -len(suf)]
                        # If stem ends with combining 'o', also drop it (laryngo + logist)
                        if self.INFER_COMBINING_O and len(stem) >= 5 and stem.endswith("o"):
                            add_inferred(stem[:-1], key)
                        add_inferred(stem, key)

                # special: logist -> ologist (helps words like otolaryngologist)
                if self.INFER_O_PREFIX_FOR_LOGIST and key.endswith("logist") and not key.endswith("ologist"):
                    add_inferred("o" + key, key)

        # 2) From proto-root Root fields
        if self.INCLUDE_PROTO_ROOTS and self.proto_lexicon_dict:
            for entries in self.proto_lexicon_dict.values():
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    r = entry.get("Root")
                    if isinstance(r, str) and r.strip():
                        root = r.strip().lower()
                        # give proto roots a neutral-ish base score
                        if root not in self.root_weight and root not in self.extra_root_weight and len(root) >= 3:
                            self.extra_root_weight[root] = float(3.2 + 0.3 * min(len(root), 8) + self.INFERRED_ROOT_PENALTY)
                            self.lex.roots.add(root)

    # -----------------------------
    # Piece scoring
    # -----------------------------

    def _score_root(self, p: str) -> Optional[Piece]:
        """Return a root Piece if p is known as direct or inferred root; else None."""
        if p in self.root_weight:
            score = self.root_weight[p]
        elif p in self.extra_root_weight:
            score = self.extra_root_weight[p]
        else:
            return None

        L = len(p)

        # Short-root adjustments (explicit knobs)
        if L == 1:
            score += self.SHORT1_ADJUST
        elif L == 2:
            score += self.SHORT2_ADJUST
        elif L == 3:
            score += self.SHORT3_ADJUST

        # Mid-length roots bonus/penalty
        if 4 <= L <= 9:
            score += self.MID_ROOT_ADJUST

        # Very long roots adjustment beyond threshold
        if L > self.LONG_ROOT_THRESHOLD:
            score += self.LONG_ROOT_ADJUST * (L - self.LONG_ROOT_THRESHOLD)

        return Piece(p, "root", float(score))

    def _score_piece(self, p: str) -> Piece:
        """Score a candidate substring p (lowercased)."""
        root_piece = self._score_root(p)
        if root_piece is not None:
            return root_piece

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
