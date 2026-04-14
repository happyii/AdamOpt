"""Core text frequency estimation engine based on Adam's Law.

Implements word-level frequency (wfreq) and sentence-level frequency (sfreq)
calculation using the Textual Frequency Law formula.

Reference:
    sfreq(x) = (∏_{k=1}^K wfreq(x_k))^(1/K)
    where K is the number of effective words (excluding stopwords).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from adamopt.freq_table import FreqTableManager
from adamopt.stopwords import StopwordManager
from adamopt.tokenizer import detect_language, tokenize


# Minimum frequency value for unknown words (avoid log(0))
_MIN_FREQ: float = 1e-8

# Sliding window size for long text processing
_DEFAULT_WINDOW_SIZE: int = 50
_DEFAULT_WINDOW_STRIDE: int = 25


@dataclass
class FrequencySource:
    """Tracks which frequency source was used for word lookups.

    Attributes:
        level: The lookup level used (1=custom_table, 2=model_cache, 3=wordfreq).
        name: Human-readable name of the source.
        path: File path if applicable (custom table or cache file), None for wordfreq.
    """

    level: int
    name: str
    path: str | None = None

    def __str__(self) -> str:
        if self.path:
            return f"[Level {self.level}] {self.name} ({self.path})"
        return f"[Level {self.level}] {self.name}"


@dataclass
class FrequencyResult:
    """Result of text frequency estimation.

    Attributes:
        text: The original input text.
        sfreq: Sentence-level frequency score (0-1, higher = more common).
        word_frequencies: Dictionary mapping each effective word to its frequency.
        low_freq_words: List of (word, freq) tuples for words below the threshold,
            sorted by frequency ascending (lowest first).
        language: Detected or specified language.
        model: Target model name.
        freq_source: Information about which frequency source was used.
    """

    text: str
    sfreq: float
    word_frequencies: dict[str, float]
    low_freq_words: list[tuple[str, float]]
    language: str
    model: str
    effective_word_count: int = 0
    freq_source: FrequencySource | None = None

    def summary(self) -> str:
        """Return a human-readable summary of the frequency analysis."""
        lines = [
            f"Text Frequency Analysis (model={self.model}, lang={self.language})",
            f"  Sentence Frequency (sfreq): {self.sfreq:.6f}",
            f"  Effective Words: {self.effective_word_count}",
        ]
        if self.freq_source:
            lines.append(f"  Frequency Source: {self.freq_source}")
        if self.low_freq_words:
            lines.append(f"  Low-Frequency Words ({len(self.low_freq_words)}):")
            for word, freq in self.low_freq_words[:10]:
                lines.append(f"    - {word!r}: {freq:.8f}")
            if len(self.low_freq_words) > 10:
                lines.append(f"    ... and {len(self.low_freq_words) - 10} more")
        else:
            lines.append("  Low-Frequency Words: none detected")
        return "\n".join(lines)


class FrequencyEstimator:
    """Text frequency estimator based on Adam's Law (Textual Frequency Law).

    Calculates how "common" a piece of text is relative to LLM pretraining
    distributions, using word-level and sentence-level frequency metrics.

    Example:
        >>> estimator = FrequencyEstimator(model="qwen2.5-7b")
        >>> result = estimator.estimate("What is the capital of France?")
        >>> print(f"sfreq = {result.sfreq:.6f}")
        >>> print(result.summary())
    """

    def __init__(
        self,
        model: str = "generic",
        language: str | None = None,
        *,
        custom_freq_table: dict[str, float] | None = None,
        custom_freq_table_path: str | None = None,
        custom_stopwords: set[str] | None = None,
        low_freq_threshold: float = 0.0001,
        window_size: int = _DEFAULT_WINDOW_SIZE,
        window_stride: int = _DEFAULT_WINDOW_STRIDE,
        cache_dir: str | None = None,
    ) -> None:
        """Initialize the frequency estimator.

        Args:
            model: Target LLM model name (e.g., "qwen2.5-7b", "llama3.3-70b").
                Use "generic" for a model-agnostic baseline.
            language: Force language ("en" or "zh"). Auto-detected if None.
            custom_freq_table: User-provided {word: freq} dictionary to use
                instead of wordfreq.
            custom_freq_table_path: Path to a custom frequency table file
                (.json or .tsv format).
            custom_stopwords: Additional stopwords to include.
            low_freq_threshold: Words with frequency below this value are
                flagged as low-frequency bottlenecks. Default: 0.0001.
            window_size: Sliding window size (in words) for long texts.
            window_stride: Stride for the sliding window.
            cache_dir: Directory for caching frequency tables.
        """
        self.model = model
        self.language = language
        self.low_freq_threshold = low_freq_threshold
        self.window_size = window_size
        self.window_stride = window_stride

        # Initialize stopword manager
        self._stopword_mgr = StopwordManager(custom_stopwords=custom_stopwords)

        # Initialize frequency table manager
        self._table_mgr = FreqTableManager(cache_dir=cache_dir)

        # Load custom frequency table if provided
        self._custom_freq: dict[str, float] | None = None
        self._custom_freq_path: str | None = None
        if custom_freq_table is not None:
            self._custom_freq = custom_freq_table
            self._custom_freq_path = "<dict>"
        elif custom_freq_table_path is not None:
            self._custom_freq = self._table_mgr.load_custom_table(
                custom_freq_table_path, model_name=model
            )
            self._custom_freq_path = str(custom_freq_table_path)

        # Track which frequency source is actually used during lookup
        self._active_source: FrequencySource | None = None

    def estimate(self, text: str) -> FrequencyResult:
        """Estimate the text frequency for a given input.

        Args:
            text: Input text (prompt, instruction, or any text).

        Returns:
            FrequencyResult containing sfreq, word frequencies, low-freq words,
            and the frequency source used.
        """
        # Reset source tracking for this call
        self._active_source = None

        if not text or not text.strip():
            return FrequencyResult(
                text=text,
                sfreq=0.0,
                word_frequencies={},
                low_freq_words=[],
                language=self.language or "en",
                model=self.model,
                effective_word_count=0,
                freq_source=None,
            )

        # Detect language
        lang = self.language or detect_language(text)

        # Tokenize
        tokens = tokenize(text, language=lang)

        # Filter stopwords to get effective words
        effective_words = [
            w for w in tokens if not self._stopword_mgr.is_stopword(w)
        ]

        if not effective_words:
            return FrequencyResult(
                text=text,
                sfreq=0.0,
                word_frequencies={},
                low_freq_words=[],
                language=lang,
                model=self.model,
                effective_word_count=0,
                freq_source=None,
            )

        # Calculate word frequencies
        word_freqs = self._get_word_frequencies(effective_words, lang)

        # Calculate sentence-level frequency using sliding window for long texts
        if len(effective_words) > self.window_size:
            sfreq = self._sliding_window_sfreq(effective_words, lang)
        else:
            sfreq = self._compute_sfreq(word_freqs)

        # Identify low-frequency bottleneck words
        low_freq = [
            (w, f) for w, f in word_freqs.items() if f < self.low_freq_threshold
        ]
        low_freq.sort(key=lambda x: x[1])

        return FrequencyResult(
            text=text,
            sfreq=sfreq,
            word_frequencies=word_freqs,
            low_freq_words=low_freq,
            language=lang,
            model=self.model,
            effective_word_count=len(effective_words),
            freq_source=self._active_source,
        )

    def estimate_batch(self, texts: list[str]) -> list[FrequencyResult]:
        """Estimate text frequency for a batch of inputs.

        Args:
            texts: List of input texts.

        Returns:
            List of FrequencyResult objects.
        """
        return [self.estimate(text) for text in texts]

    def compare(self, text_a: str, text_b: str) -> dict[str, Any]:
        """Compare the frequency of two texts.

        Useful for evaluating which paraphrase has higher frequency.

        Args:
            text_a: First text.
            text_b: Second text.

        Returns:
            Dictionary with comparison results.
        """
        result_a = self.estimate(text_a)
        result_b = self.estimate(text_b)

        return {
            "text_a": {"text": text_a, "sfreq": result_a.sfreq},
            "text_b": {"text": text_b, "sfreq": result_b.sfreq},
            "higher_freq": "a" if result_a.sfreq >= result_b.sfreq else "b",
            "freq_ratio": (
                result_a.sfreq / result_b.sfreq
                if result_b.sfreq > 0
                else float("inf")
            ),
        }

    def _get_word_frequencies(
        self, words: list[str], lang: str
    ) -> dict[str, float]:
        """Get frequency for each word using available frequency sources.

        Priority: custom_freq_table > cached model table > wordfreq library.
        """
        word_freqs: dict[str, float] = {}
        unique_words = set(words)

        for word in unique_words:
            freq = self._lookup_word_freq(word, lang)
            word_freqs[word] = freq

        return word_freqs

    def _lookup_word_freq(self, word: str, lang: str) -> float:
        """Look up frequency for a single word.

        Falls back through: custom table -> model cache -> wordfreq library.
        On first successful lookup, records the source level for user reporting.
        """
        # 1. Custom frequency table
        if self._custom_freq is not None:
            freq = self._custom_freq.get(word) or self._custom_freq.get(word.lower())
            if freq is not None:
                if self._active_source is None:
                    self._active_source = FrequencySource(
                        level=1,
                        name="Custom frequency table",
                        path=self._custom_freq_path,
                    )
                return max(freq, _MIN_FREQ)

        # 2. Cached model table (from TFD or user import)
        cached = self._table_mgr.get_table(self.model)
        if cached is not None:
            freq = cached.get(word) or cached.get(word.lower())
            if freq is not None:
                if self._active_source is None:
                    cache_path = self._table_mgr.get_table_path(self.model)
                    self._active_source = FrequencySource(
                        level=2,
                        name=f"Model cache ({self.model})",
                        path=str(cache_path) if cache_path else None,
                    )
                return max(freq, _MIN_FREQ)

        # 3. wordfreq library (primary source for v0.1)
        if self._active_source is None:
            self._active_source = FrequencySource(
                level=3,
                name="wordfreq library (generic corpus)",
                path=None,
            )
        return self._wordfreq_lookup(word, lang)

    def _wordfreq_lookup(self, word: str, lang: str) -> float:
        """Look up word frequency using the wordfreq library.

        Returns frequency in 0-1 range. Uses zipf_frequency internally
        and converts to a linear probability-like value.
        """
        try:
            from wordfreq import word_frequency

            # wordfreq returns frequency as a fraction (e.g., 0.001 = 0.1%)
            # This is already in 0-1 range, suitable for our sfreq formula
            freq = word_frequency(word, lang)
            if freq > 0:
                return freq

            # Try lowercase if original case failed
            freq = word_frequency(word.lower(), lang)
            if freq > 0:
                return freq

            return _MIN_FREQ

        except ImportError:
            raise ImportError(
                "The 'wordfreq' package is required. Install it with: "
                "pip install wordfreq"
            )

    @staticmethod
    def _compute_sfreq(word_freqs: dict[str, float]) -> float:
        """Compute sentence-level frequency using Adam's Law formula.

        Formula: sfreq(x) = (∏_{k=1}^K wfreq(x_k))^(1/K)
        Equivalent to: exp(mean(log(wfreq(x_k))))

        This is the geometric mean of word frequencies.
        """
        if not word_freqs:
            return 0.0

        freqs = list(word_freqs.values())
        K = len(freqs)

        # Use log-space for numerical stability
        log_freqs = [math.log(max(f, _MIN_FREQ)) for f in freqs]
        mean_log = sum(log_freqs) / K

        return math.exp(mean_log)

    def _sliding_window_sfreq(
        self, words: list[str], lang: str
    ) -> float:
        """Compute sfreq for long texts using sliding window average.

        Splits the text into overlapping windows, computes sfreq for each,
        and returns the average. This avoids length bias in frequency estimation.
        """
        window_sfreqs = []

        for start in range(0, len(words), self.window_stride):
            end = min(start + self.window_size, len(words))
            window_words = words[start:end]
            if not window_words:
                break

            window_freqs = {}
            for w in set(window_words):
                window_freqs[w] = self._lookup_word_freq(w, lang)

            sfreq = self._compute_sfreq(window_freqs)
            window_sfreqs.append(sfreq)

            if end >= len(words):
                break

        if not window_sfreqs:
            return 0.0

        # Geometric mean of window sfreqs
        log_sfreqs = [math.log(max(s, _MIN_FREQ)) for s in window_sfreqs]
        return math.exp(sum(log_sfreqs) / len(log_sfreqs))
