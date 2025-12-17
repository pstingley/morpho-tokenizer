"""
morpho_vtb_tokenizer.py

Etymology-/morphology-aware tokenizer using a Viterbi-style DP segmenter.

Option 1 (lexicon-induced root backoff):
- Builds an initial root lexicon from the keys of morph_lexicon_by_root.json(.gz)
- Then **infers additional candidate roots** by stripping common biomedical suffixes
  from lexicon entries (e.g., "laryngologist" -> "laryng", "otology" -> "oto").
  These inferred roots are added with a small weight discount so they help when
  appropriate, but won't dominate over strong, explicit roots.

Notes:
- This module intentionally does NOT enforce any model-specific max length (e.g., 512).
  Put max_length/truncation in the *calling* model code (PubMedBERT, etc.).
"""

from __future__ import annotations

import gzip
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Set


@dataclass(frozen=True)
class Piece:
    text: str
    kind: str  # "root" | "prefix" | "suffix" | "unk"
    score: float


@dataclass
class RootLexicon:
    roots: Set[str]
    prefixes: Set[str]
    suffixes: Set[str]

    @staticmethod
    def empty() -> "RootLexicon":
        return RootLexicon(roots=set(), prefixes=set(), suffixes=set())


@dataclass
class _DPState:
    score: float
    back: Optional[int]
    piece: Optional[Piece]


class MorphoTokenizer:
    """
    Viterbi-based morphological tokenizer using Wiktionary-derived lexicons.

    Parameters you most commonly tune:
      - lambda_penalty: discourages over-segmentation (higher => fewer pieces)
      - long_unsplit_penalty: pushes long words to split at least once
      - short1_adjust / short2_adjust: downweight 1–2 char roots (negative)
      - mid_root_adjust: encourage mid-length roots (positive)
      - infer_roots: enable lexicon-induced root backoff
      - infer_weight_discount: discount applied to inferred roots (>=0)

    The DP objective maximizes sum(piece_score) - lambda_penalty * (#pieces)
    with an extra penalty when a long word is kept as a single piece.
    """

    # Default suffixes for lexicon-induced root inference (defensible + minimal)
    DEFAULT_INFER_SUFFIXES = (
        "ologist", "ology", "ological", "logist", "logy",
        "ectomy", "itis", "osis", "emia", "pathy", "phagia",
        "algia", "uria", "genic", "genesis",
        "eal", "ial", "al", "ic", "ical", "ist", "ism", "ous", "ary", "ory",
    )

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
        # Length-based tuning knobs
        short1_adjust: float = -8.0,   # NEW: downweight 1-char roots (negative)
        short2_adjust: float = -5.0,   # downweight 2-char roots (negative)
        mid_root_adjust: float = +2.5, # reward 4–9 char roots (positive)
        long_root_adjust: float = -0.5,# per-char adjustment beyond LONG_ROOT_THRESHOLD
        long_root_threshold: int = 7,
        # Option 1: lexicon-induced root backoff
        infer_roots: bool = True,
        infer_suffixes: Optional[Tuple[str, ...]] = None,
        infer_min_len: int = 3,
        infer_weight_discount: float = 1.25,  # subtract from inferred-root score
    ) -> None:
        # --- Load root lexicon (required) ---
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

        # Core candidate roots = keys of root_dict
        for r in root_dict.keys():
            if isinstance(r, str) and r.strip():
                self.lex.roots.add(r.strip().lower())

        # Optional: add a few generic suffixes (helps biomedical terms a lot)
        if add_generic_suffixes:
            generic_suffixes = {
                "s", "es", "ed", "ing", "er", "est", "ly", "ness", "less",
                "ment", "tion", "sion", "ity", "ous", "ial", "al", "ic",
                # biomedical-ish
                "logy", "logist", "ology", "ologist", "ectomy", "itis", "osis",
            }
            self.lex.suffixes |= generic_suffixes

        # --- Build root weights from lexicon frequency/length ---
        self.root_weight: Dict[str, float] = {}
        for key, info in root_dict.items():
            if not isinstance(key, str) or not key.strip():
                continue
            k = key.strip().lower()
            freq_raw = info.get("NumSourceWords", 1) if isinstance(info, dict) else 1
            try:
                freq = int(freq_raw)
            except Exception:
                freq = 1
            w = 3.0 + math.log1p(max(freq, 1)) + 0.3 * min(len(k), 8)
            self.root_weight[k] = float(w)

        # --- Option 1: infer additional roots by suffix stripping ---
        self.infer_roots_enabled = bool(infer_roots)
        self.infer_min_len = int(infer_min_len)
        self.infer_weight_discount = float(infer_weight_discount)
        self.infer_suffixes = tuple(infer_suffixes) if infer_suffixes else self.DEFAULT_INFER_SUFFIXES

        if self.infer_roots_enabled:
            self._infer_and_add_roots()

        # Suffix/prefix weights
        self.suffix_weight: Dict[str, float] = {}
        self.prefix_weight: Dict[str, float] = {}
        for s in self.lex.suffixes:
            self.suffix_weight[s] = 3.0 + 0.2 * min(len(s), 6)
        for p in self.lex.prefixes:
            self.prefix_weight[p] = 2.8 + 0.2 * min(len(p), 6)

        # DP/scoring config
        self.UNK_BASE_PENALTY = float(unk_base_penalty)
        self.UNK_PER_CHAR = float(unk_per_char)
        self.MAX_MORPHEME_LEN = int(max_morpheme_len)

        self.LAMBDA_PENALTY = float(lambda_penalty)
        self.LONG_UNSPLIT_MIN_LEN = int(long_unsplit_min_len)
        self.LONG_UNSPLIT_PENALTY = float(long_unsplit_penalty)

        self.SHORT1_ADJUST = float(short1_adjust)
        self.SHORT2_ADJUST = float(short2_adjust)
        self.MID_ROOT_ADJUST = float(mid_root_adjust)
        self.LONG_ROOT_ADJUST = float(long_root_adjust)
        self.LONG_ROOT_THRESHOLD = int(long_root_threshold)

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

        # Cache for word segmentation
        self._cache: Dict[str, List[Piece]] = {}

    # -----------------------------
    # Lexicon loading
    # -----------------------------

    def _load_lexicon_dict(self, path: Path) -> Dict[str, dict]:
        if not path.exists():
            raise FileNotFoundError(f"Lexicon file not found: {path}")
        if path.suffix.lower() == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_proto_dict(self, path: Path) -> Dict[str, list]:
        if not path.exists():
            raise FileNotFoundError(f"Proto lexicon file not found: {path}")
        if path.suffix.lower() == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _build_root_to_proto_map(self, proto_dict: Dict[str, list]) -> Dict[str, List[str]]:
        mapping: Dict[str, List[str]] = {}
        for proto, entries in proto_dict.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                root = entry.get("Root")
                if not isinstance(root, str) or not root.strip():
                    continue
                mapping.setdefault(root.strip().lower(), []).append(str(proto))
        return mapping

    # -----------------------------
    # Option 1: root inference
    # -----------------------------

    def _infer_and_add_roots(self) -> None:
        """
        Lexicon-induced root backoff:
        For each lexicon surface form, strip common suffixes; if the remaining stem
        looks like a plausible root, add it as an inferred candidate root.
        """
        inferred: Dict[str, float] = {}

        for k, base_w in list(self.root_weight.items()):
            if len(k) < self.infer_min_len + 1:
                continue

            for suf in self.infer_suffixes:
                if len(suf) < 2:
                    continue
                if not k.endswith(suf):
                    continue

                stem = k[: -len(suf)].strip("-_ ")
                if len(stem) < self.infer_min_len:
                    continue
                if not re.fullmatch(r"[a-z]+", stem):
                    continue

                # Weight: inherit from the parent lexicon entry, but discounted
                w = float(base_w) - self.infer_weight_discount
                inferred[stem] = max(inferred.get(stem, float("-inf")), w)

                # If stem ends with "o" (laryngo), also add version without trailing 'o'
                if stem.endswith("o") and len(stem) - 1 >= self.infer_min_len:
                    stem2 = stem[:-1]
                    inferred[stem2] = max(inferred.get(stem2, float("-inf")), w - 0.15)

        for stem, w in inferred.items():
            self.lex.roots.add(stem)
            if stem not in self.root_weight or self.root_weight[stem] < w:
                self.root_weight[stem] = float(w)

    # -----------------------------
    # Piece scoring
    # -----------------------------

    def _score_piece(self, p: str) -> Piece:
        """Score a candidate substring p (lowercased)."""
        if p in self.root_weight:
            score = float(self.root_weight[p])
            L = len(p)

            # Downweight tiny roots (still allowed)
            if L == 1:
                score += self.SHORT1_ADJUST
            elif L == 2:
                score += self.SHORT2_ADJUST

            # Mid-length roots get a boost
            if 4 <= L <= 9:
                score += self.MID_ROOT_ADJUST

            # Very long roots: per-char adjustment beyond threshold
            if L > self.LONG_ROOT_THRESHOLD:
                score += self.LONG_ROOT_ADJUST * (L - self.LONG_ROOT_THRESHOLD)

            return Piece(p, "root", score)

        if p in self.suffix_weight:
            return Piece(p, "suffix", float(self.suffix_weight[p]))
        if p in self.prefix_weight:
            return Piece(p, "prefix", float(self.prefix_weight[p]))

        return Piece(p, "unk", float(self.UNK_BASE_PENALTY + self.UNK_PER_CHAR * len(p)))

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

        dp: List[_DPState] = [_DPState(float("-inf"), None, None) for _ in range(n + 1)]
        dp[0] = _DPState(0.0, None, None)

        for i in range(n):
            if dp[i].score == float("-inf"):
                continue

            max_j = min(n, i + self.MAX_MORPHEME_LEN)
            for j in range(i + 1, max_j + 1):
                sub = w[i:j]
                piece = self._score_piece(sub)

                score = dp[i].score + piece.score - self.LAMBDA_PENALTY

                if i == 0 and j == n and n >= self.LONG_UNSPLIT_MIN_LEN:
                    score -= self.LONG_UNSPLIT_PENALTY

                if score > dp[j].score:
                    dp[j] = _DPState(score, i, piece)

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

    def segment_word(self, word: str) -> List[Piece]:
        return self._segment_word_dp(word)

    def tokenize_and_segment(self, text: str) -> List[Tuple[str, List[Piece]]]:
        parts = re.findall(r"\w+|[^\w\s]+|\s+", text)
        out: List[Tuple[str, List[Piece]]] = []
        for tok in parts:
            if tok.isspace():
                out.append((tok, []))
            elif re.fullmatch(r"\w+", tok):
                out.append((tok, self.segment_word(tok)))
            else:
                out.append((tok, []))
        return out

    # -----------------------------
    # Proto-root helpers
    # -----------------------------

    def get_proto_roots_for(self, root: str) -> List[str]:
        return self.root_to_proto.get(root.lower(), [])

    def get_proto_roots_for_piece(self, piece: Piece) -> List[str]:
        if piece.kind != "root":
            return []
        return self.get_proto_roots_for(piece.text)

    # -----------------------------
    # Visualization
    # -----------------------------

    def visualize_word(self, word: str, show_proto: bool = False) -> None:
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
