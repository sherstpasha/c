"""
OCR Spell Corrector - Language-agnostic spell correction for OCR text

This module provides a flexible, configurable spell correction system designed
for post-OCR text processing. It uses SymSpell for candidate generation and
applies customizable filters to ensure high-quality corrections.

Key features:
- Language-agnostic design with configurable character mappings
- Prefix/suffix protection to avoid morphological errors
- Selective token correction based on configurable criteria
- Efficient caching and memory-optimized implementation
"""

import re
import unicodedata
import os
import psutil
import pandas as pd
from symspellpy import SymSpell, Verbosity
from rapidfuzz.distance import Levenshtein


class OCRSpellCorrector:
    """
    A spell corrector optimized for OCR text with configurable filters.

    This corrector uses SymSpell for fast candidate generation and applies
    multiple filters to ensure corrections respect linguistic patterns.

    Parameters
    ----------
    words : set or list
        Dictionary words to use for correction
    max_edit : int, default=2
        Maximum edit distance for corrections
    min_token_len : int, default=3
        Minimum token length to consider for correction
    symspell_prefix_len : int, default=7
        Prefix length for SymSpell indexing (affects memory and speed)
    protect_prefix_len : int, default=2
        Number of prefix characters to protect from substitution
    char_map : dict, optional
        Character mapping for normalization (e.g., {'i': 'і', 'I': 'І'})
    skip_hyphen_tokens : bool, default=True
        Whether to skip tokens with hyphens in punctuation
    max_candidates : int, default=10
        Maximum number of candidates to consider per word
    forbidden_suffix_changes : list of tuple, optional
        List of (suffix_from, suffix_to) pairs to forbid
    """

    def __init__(
        self,
        words,
        max_edit=2,
        min_token_len=3,
        symspell_prefix_len=7,
        protect_prefix_len=2,
        char_map=None,
        skip_hyphen_tokens=True,
        max_candidates=10,
        forbidden_suffix_changes=None,
    ):
        self.max_edit = max_edit
        self.min_token_len = min_token_len
        self.protect_prefix_len = protect_prefix_len
        self.char_map = char_map or {}
        self.skip_hyphen_tokens = skip_hyphen_tokens
        self.max_candidates = max_candidates

        # Default forbidden suffix changes (can be overridden for other languages)
        self.forbidden_suffix_changes = forbidden_suffix_changes or [
            ("е", "и"),
            ("и", "е"),
            ("ть", "те"),
            ("ть", "т"),
            ("й", "я"),
            ("я", "й"),
        ]

        self.cache = {}
        self.log = []

        # Statistics tracking
        self.stats = {
            "tokens_processed": 0,
            "tokens_corrected": 0,
            "tokens_skipped": 0,
        }

        # Convert to lowercase set
        self.words = set(w.lower() for w in words)

        # Regex for splitting punctuation
        self._punct_re = re.compile(r"^([^\w]*)([\wА-Яа-яёЁ]+)([^\w]*)$")

        # Initialize SymSpell
        self.symspell = SymSpell(
            max_dictionary_edit_distance=max_edit,
            prefix_length=symspell_prefix_len,
        )

        # Build dictionary
        for w in self.words:
            self.symspell.create_dictionary_entry(self.normalize(w), 1)

        self.print_ram("After SymSpell build")

    def print_ram(self, label):
        """Print current RAM usage."""
        rss = psutil.Process(os.getpid()).memory_info().rss / 1024**2
        print(f"[RAM] {label}: {rss:.1f} MB")

    def normalize(self, s: str) -> str:
        """
        Normalize string using NFKC and custom character mapping.

        Parameters
        ----------
        s : str
            Input string

        Returns
        -------
        str
            Normalized string
        """
        s = unicodedata.normalize("NFKC", s)
        return "".join(self.char_map.get(c, c) for c in s)

    def restore_case(self, src: str, dst: str) -> str:
        """
        Restore case from source string to destination string.

        Parameters
        ----------
        src : str
            Source string (original)
        dst : str
            Destination string (corrected)

        Returns
        -------
        str
            Destination with case restored from source
        """
        if src.isupper():
            return dst.upper()
        if src and src[0].isupper():
            return dst.capitalize()
        return dst.lower()

    def split_token(self, token):
        """
        Split token into prefix punctuation, core word, and suffix punctuation.

        Parameters
        ----------
        token : str
            Input token

        Returns
        -------
        tuple
            (prefix_punct, core_word, suffix_punct) or (None, token, None)
        """
        m = self._punct_re.match(token)
        if not m:
            return None, token, None
        return m.group(1), m.group(2), m.group(3)

    def prefix_ok(self, a: str, b: str) -> bool:
        """
        Check if prefix characters match (considering char_map).

        Parameters
        ----------
        a : str
            First string (original)
        b : str
            Second string (candidate)

        Returns
        -------
        bool
            True if prefixes match within protection rules
        """
        n = self.protect_prefix_len
        if n <= 0:
            return True
        if len(a) < n or len(b) < n:
            return False
        for i in range(n):
            x, y = a[i], b[i]
            if x == y:
                continue
            if self.char_map.get(x) == y:
                continue
            if self.char_map.get(y) == x:
                continue
            return False
        return True

    def suffix_ok(self, a: str, b: str) -> bool:
        """
        Check if suffix change is allowed.

        Parameters
        ----------
        a : str
            First string (original)
        b : str
            Second string (candidate)

        Returns
        -------
        bool
            True if suffix change is allowed
        """
        for x, y in self.forbidden_suffix_changes:
            if a.endswith(x) and b.endswith(y):
                return False
        return True

    def is_candidate(self, token):
        """
        Check if token is eligible for correction.

        Parameters
        ----------
        token : str
            Token to check

        Returns
        -------
        bool
            True if token should be considered for correction
        """
        if len(token) < self.min_token_len:
            return False
        if any(c.isdigit() for c in token):
            return False
        if any(c in token for c in "_/\\|@#$%^&*+=<>[]{}"):
            return False
        letters = sum(c.isalpha() for c in token)
        return letters / len(token) >= 0.8

    def correct_word(self, word):
        """
        Correct a single word.

        Parameters
        ----------
        word : str
            Word to correct

        Returns
        -------
        str
            Corrected word (or original if no correction found)
        """
        word_l = word.lower()

        # Already in dictionary - don't touch
        if word_l in self.words:
            return word

        # Check cache
        if word_l in self.cache:
            return self.cache[word_l]

        # Normalize and get candidates
        norm = self.normalize(word_l)

        suggestions = self.symspell.lookup(
            norm,
            Verbosity.TOP,
            max_edit_distance=self.max_edit,
        )[: self.max_candidates]

        best = word_l
        best_dist = self.max_edit + 1

        # Apply filters and find best candidate
        for s in suggestions:
            cand = s.term

            if not self.prefix_ok(norm, cand):
                continue
            if not self.suffix_ok(word_l, cand):
                continue

            d = Levenshtein.distance(norm, cand)
            if d < best_dist:
                best_dist = d
                best = cand

        # Don't take boundary cases
        if best_dist >= self.max_edit:
            result = word
        else:
            result = self.restore_case(word, best)

        self.cache[word_l] = result
        return result

    def correct_text(self, text, row_id=None, gt_text=None):
        """
        Correct text by processing tokens.

        Parameters
        ----------
        text : str
            Text to correct
        row_id : any, optional
            Row identifier for logging
        gt_text : str, optional
            Ground truth text for logging

        Returns
        -------
        str
            Corrected text
        """
        tokens = text.split()
        out = []
        changed = False

        for t in tokens:
            self.stats["tokens_processed"] += 1

            p, core, s = self.split_token(t)
            if p is None:
                out.append(t)
                self.stats["tokens_skipped"] += 1
                continue

            if self.skip_hyphen_tokens and ("-" in p or "-" in s):
                out.append(t)
                self.stats["tokens_skipped"] += 1
                continue

            if self.is_candidate(core):
                new_core = self.correct_word(core)
                new_tok = f"{p}{new_core}{s}"

                if new_tok != t:
                    self.stats["tokens_corrected"] += 1

                    if row_id is not None:
                        self.log.append(
                            {
                                "row_id": row_id,
                                "token_before": t,
                                "token_after": new_tok,
                                "before_text": text,
                                "after_text": None,
                                "gt_text": gt_text,
                            }
                        )
                    changed = True

                out.append(new_tok)
            else:
                out.append(t)
                self.stats["tokens_skipped"] += 1

        new_text = " ".join(out)

        # Update after_text in log
        if changed and row_id is not None:
            for item in reversed(self.log):
                if item["row_id"] != row_id:
                    break
                if item["after_text"] is None:
                    item["after_text"] = new_text

        return new_text

    def get_statistics(self):
        """
        Get correction statistics.

        Returns
        -------
        dict
            Dictionary with statistics
        """
        total = self.stats["tokens_processed"]
        corrected = self.stats["tokens_corrected"]

        return {
            "tokens_processed": total,
            "tokens_corrected": corrected,
            "tokens_skipped": self.stats["tokens_skipped"],
            "correction_rate": corrected / total if total > 0 else 0.0,
            "cache_size": len(self.cache),
        }

    def save_log(self, path):
        """
        Save correction log to CSV.

        Parameters
        ----------
        path : str
            Output file path
        """
        if self.log:
            pd.DataFrame(self.log).to_csv(path, index=False, encoding="utf-8")
            print(f"Saved {len(self.log)} corrections → {path}")
        else:
            print("No corrections.")

    def reset_statistics(self):
        """Reset statistics counters."""
        self.stats = {
            "tokens_processed": 0,
            "tokens_corrected": 0,
            "tokens_skipped": 0,
        }
