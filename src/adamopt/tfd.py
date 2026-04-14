"""Textual Frequency Distillation (TFD) — Generate model-specific frequency tables.

TFD aligns word frequency estimates with a specific LLM's internal distribution
by analyzing text that the model generates (continuations/completions).

Workflow:
    1. User provides a corpus (text file, one sentence per line).
    2. (Optional) The user has already used the target LLM to generate continuations
       for each sentence and collected them into a "generated corpus" file.
    3. TFD counts word frequencies from the generated corpus.
    4. TFD merges the generated corpus frequencies with baseline wordfreq frequencies
       using median-ratio alignment and multiplicative fusion.
    5. The merged table is saved as a model-specific frequency table (JSON),
       which can be loaded by FrequencyEstimator for subsequent use.

Reference:
    - Adam's Law paper, Section: Textual Frequency Distillation (TFD)
    - Official implementation: frequencylaw-main/newfrequency.py (align_and_merge)
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from adamopt.freq_table import FreqTableManager, _normalize_freq_dict
from adamopt.stopwords import StopwordManager
from adamopt.tokenizer import detect_language, tokenize


# Small epsilon to avoid division by zero
_EPSILON: float = 1e-8


class TFDDistiller:
    """Textual Frequency Distillation engine.

    Generates model-specific word frequency tables by merging baseline
    (wordfreq) frequencies with frequencies counted from model-generated text.

    Example:
        >>> distiller = TFDDistiller(model="qwen2.5-7b")
        >>> result = distiller.distill(
        ...     generated_corpus_path="model_continuations.txt",
        ...     output_path="qwen2.5-7b_freq.json",
        ... )
        >>> print(f"Merged {result['vocab_size']} words")
    """

    def __init__(
        self,
        model: str = "generic",
        language: str | None = None,
        *,
        custom_stopwords: set[str] | None = None,
        cache_dir: str | None = None,
    ) -> None:
        """Initialize the TFD distiller.

        Args:
            model: Target LLM model name.
            language: Language code ("en" or "zh"). Auto-detected if None.
            custom_stopwords: Additional stopwords to exclude from counting.
            cache_dir: Directory for caching frequency tables.
        """
        self.model = model
        self.language = language
        self._stopword_mgr = StopwordManager(custom_stopwords=custom_stopwords)
        self._table_mgr = FreqTableManager(cache_dir=cache_dir)

    def count_corpus_frequencies(
        self,
        corpus_path: str | Path,
        language: str | None = None,
    ) -> dict[str, float]:
        """Count word frequencies from a text corpus file.

        Args:
            corpus_path: Path to the corpus file (one sentence/paragraph per line).
            language: Language for tokenization. Auto-detected if None.

        Returns:
            Normalized word frequency dictionary {word: prob}.
        """
        corpus_path = Path(corpus_path)
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

        word_counts: Counter = Counter()
        lang = language or self.language

        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Auto-detect language from first non-empty line if not set
                if lang is None:
                    lang = detect_language(line)

                tokens = tokenize(line, language=lang)
                # Filter stopwords
                effective = [
                    w for w in tokens if not self._stopword_mgr.is_stopword(w)
                ]
                word_counts.update(effective)

        if not word_counts:
            return {}

        # Normalize to probabilities
        total = sum(word_counts.values())
        return {word: count / total for word, count in word_counts.items()}

    def get_baseline_frequencies(
        self, vocab: set[str], language: str = "en"
    ) -> dict[str, float]:
        """Get baseline word frequencies from wordfreq for a given vocabulary.

        Args:
            vocab: Set of words to look up.
            language: Language code for wordfreq lookup.

        Returns:
            Baseline frequency dictionary {word: freq}.
        """
        try:
            from wordfreq import word_frequency
        except ImportError:
            raise ImportError(
                "The 'wordfreq' package is required. Install: pip install wordfreq"
            )

        baseline: dict[str, float] = {}
        for word in vocab:
            freq = word_frequency(word, language)
            if freq <= 0:
                freq = word_frequency(word.lower(), language)
            baseline[word] = max(freq, _EPSILON)

        return baseline

    @staticmethod
    def align_and_merge(
        baseline_probs: dict[str, float],
        corpus_probs: dict[str, float],
        *,
        epsilon: float = _EPSILON,
    ) -> dict[str, float]:
        """Align and merge baseline frequencies with corpus-derived frequencies.

        Uses the median-ratio method from the TFD paper:
        1. Compute the ratio baseline/corpus for all common words.
        2. Take the median ratio as the alignment factor (gamma).
        3. Scale corpus probs by gamma to align distributions.
        4. Multiply aligned probs: merged = baseline * corpus_aligned.
        5. Re-normalize to sum to 1.

        Args:
            baseline_probs: Baseline word frequencies (e.g., from wordfreq).
            corpus_probs: Frequencies counted from model-generated corpus.
            epsilon: Small value for smoothing.

        Returns:
            Merged and normalized frequency dictionary.
        """
        # Compute alignment factor (gamma) from common words
        common_words = set(baseline_probs.keys()) & set(corpus_probs.keys())
        if len(common_words) > 10:
            ratios = [
                (baseline_probs[w] + epsilon) / (corpus_probs[w] + epsilon)
                for w in common_words
                if corpus_probs[w] > 1e-10 and baseline_probs[w] > 1e-10
            ]
            gamma = statistics.median(ratios) if ratios else 1.0
        else:
            gamma = 1.0

        # Align corpus probs
        corpus_aligned = {w: p * gamma for w, p in corpus_probs.items()}

        # Multiplicative fusion over combined vocabulary
        all_words = set(baseline_probs.keys()) | set(corpus_probs.keys())
        merged: dict[str, float] = {}
        for word in all_words:
            p_base = baseline_probs.get(word, epsilon)
            p_corp = corpus_aligned.get(word, epsilon)
            merged[word] = p_base * p_corp

        # Re-normalize
        total = sum(merged.values()) or 1.0
        return {word: p / total for word, p in merged.items()}

    def distill(
        self,
        generated_corpus_path: str | Path,
        output_path: str | Path | None = None,
        *,
        language: str | None = None,
        save_to_cache: bool = True,
    ) -> dict[str, Any]:
        """Run the full TFD pipeline.

        Steps:
        1. Count word frequencies from the model-generated corpus.
        2. Get baseline frequencies from wordfreq for the same vocabulary.
        3. Align and merge the two frequency distributions.
        4. Save the merged table (optionally to file and/or model cache).

        Args:
            generated_corpus_path: Path to the model-generated text file
                (one sentence/line). This should be text that the target LLM
                produced via continuation/completion of your training data.
            output_path: Where to save the merged frequency table (.json).
                If None, only saves to model cache.
            language: Language for tokenization. Auto-detected if None.
            save_to_cache: Whether to save as a model-specific cached table.

        Returns:
            Dictionary with distillation results:
            - vocab_size: Number of words in the merged table.
            - corpus_vocab_size: Number of unique words in the corpus.
            - output_path: Path where the table was saved (if applicable).
            - cache_path: Path of the model cache file (if applicable).
            - model: Target model name.
        """
        lang = language or self.language or "en"

        # Step 1: Count corpus frequencies
        corpus_probs = self.count_corpus_frequencies(
            generated_corpus_path, language=lang
        )
        if not corpus_probs:
            raise ValueError(
                f"No words found in corpus: {generated_corpus_path}. "
                "Check that the file is non-empty and contains valid text."
            )

        # Step 2: Get baseline frequencies
        all_vocab = set(corpus_probs.keys())
        baseline_probs = self.get_baseline_frequencies(all_vocab, language=lang)

        # Step 3: Merge
        merged_probs = self.align_and_merge(baseline_probs, corpus_probs)

        # Step 4: Save
        result: dict[str, Any] = {
            "vocab_size": len(merged_probs),
            "corpus_vocab_size": len(corpus_probs),
            "model": self.model,
            "language": lang,
            "output_path": None,
            "cache_path": None,
        }

        # Save to file
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    merged_probs, f, ensure_ascii=False, indent=2, sort_keys=False
                )
            result["output_path"] = str(output_path)

        # Save to model cache
        if save_to_cache:
            self._table_mgr.save_table(self.model, merged_probs)
            cache_path = self._table_mgr.get_table_path(self.model)
            result["cache_path"] = str(cache_path) if cache_path else None

        return result
