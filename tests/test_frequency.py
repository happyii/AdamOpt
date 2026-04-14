"""Tests for the core frequency estimation module."""

import math
import pytest

from adamopt.frequency import FrequencyEstimator, FrequencyResult


class TestFrequencyEstimator:
    """Test suite for FrequencyEstimator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.estimator = FrequencyEstimator(model="generic")

    # ── Basic functionality ─────────────────────────────────────────

    def test_estimate_english_text(self):
        """Test frequency estimation for simple English text."""
        result = self.estimator.estimate("Why is the sky blue?")
        assert isinstance(result, FrequencyResult)
        assert result.sfreq > 0
        assert result.language == "en"
        assert result.effective_word_count > 0
        assert len(result.word_frequencies) > 0

    def test_estimate_chinese_text(self):
        """Test frequency estimation for Chinese text."""
        result = self.estimator.estimate("天空为什么是蓝色的？")
        assert isinstance(result, FrequencyResult)
        assert result.sfreq > 0
        assert result.language == "zh"
        assert result.effective_word_count > 0

    def test_estimate_empty_text(self):
        """Test that empty text returns zero sfreq."""
        result = self.estimator.estimate("")
        assert result.sfreq == 0.0
        assert result.effective_word_count == 0

    def test_estimate_whitespace_only(self):
        """Test that whitespace-only text returns zero sfreq."""
        result = self.estimator.estimate("   \n\t  ")
        assert result.sfreq == 0.0

    # ── sfreq formula correctness ───────────────────────────────────

    def test_sfreq_is_geometric_mean(self):
        """Verify that sfreq equals the geometric mean of word frequencies."""
        result = self.estimator.estimate("sky blue")
        if result.effective_word_count > 0:
            freqs = list(result.word_frequencies.values())
            # Geometric mean: exp(mean(log(f)))
            expected = math.exp(sum(math.log(f) for f in freqs) / len(freqs))
            assert abs(result.sfreq - expected) < 1e-10

    def test_high_freq_text_higher_sfreq(self):
        """Test that common text has higher sfreq than uncommon text.

        This is the core prediction of Adam's Law.
        """
        high_freq_result = self.estimator.estimate("Why is the sky blue?")
        low_freq_result = self.estimator.estimate(
            "What is the optical causation for the azure hue of the celestial firmament?"
        )
        # High-frequency expression should have higher sfreq
        assert high_freq_result.sfreq > low_freq_result.sfreq

    # ── Low-frequency word detection ────────────────────────────────

    def test_low_freq_words_detected(self):
        """Test that uncommon words are flagged as low-frequency."""
        result = self.estimator.estimate(
            "The celestial firmament exhibits an azure chromatic manifestation"
        )
        # Some of these uncommon words should be flagged
        low_freq_word_set = {w for w, _ in result.low_freq_words}
        # At least some uncommon words should be detected
        uncommon = {"firmament", "celestial", "chromatic", "manifestation", "azure"}
        assert len(low_freq_word_set & uncommon) > 0

    def test_low_freq_words_sorted(self):
        """Test that low-frequency words are sorted ascending by frequency."""
        result = self.estimator.estimate(
            "The celestial firmament exhibits an azure chromatic manifestation"
        )
        if len(result.low_freq_words) >= 2:
            freqs = [f for _, f in result.low_freq_words]
            assert freqs == sorted(freqs)

    # ── Language detection ──────────────────────────────────────────

    def test_auto_detect_english(self):
        """Test automatic English language detection."""
        result = self.estimator.estimate("Hello world, how are you?")
        assert result.language == "en"

    def test_auto_detect_chinese(self):
        """Test automatic Chinese language detection."""
        result = self.estimator.estimate("你好世界，今天天气怎么样？")
        assert result.language == "zh"

    def test_force_language(self):
        """Test forced language setting."""
        estimator = FrequencyEstimator(model="generic", language="en")
        result = estimator.estimate("Hello world")
        assert result.language == "en"

    # ── Compare function ────────────────────────────────────────────

    def test_compare_returns_higher(self):
        """Test that compare correctly identifies the higher-frequency text."""
        result = self.estimator.compare(
            "Why is the sky blue?",
            "What is the optical causation for the azure hue of the celestial firmament?",
        )
        assert result["higher_freq"] == "a"
        assert result["freq_ratio"] > 1.0

    # ── Batch estimation ────────────────────────────────────────────

    def test_estimate_batch(self):
        """Test batch frequency estimation."""
        texts = [
            "What is machine learning?",
            "Explain the concept of machine learning.",
            "Define ML.",
        ]
        results = self.estimator.estimate_batch(texts)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, FrequencyResult)
            assert r.sfreq > 0

    # ── Model handling ──────────────────────────────────────────────

    def test_generic_model(self):
        """Test that generic model works as default."""
        estimator = FrequencyEstimator(model="generic")
        result = estimator.estimate("Hello world")
        assert result.model == "generic"
        assert result.sfreq > 0

    def test_specific_model(self):
        """Test that specific model names are accepted."""
        estimator = FrequencyEstimator(model="qwen2.5-7b")
        result = estimator.estimate("Hello world")
        assert result.model == "qwen2.5-7b"
        assert result.sfreq > 0

    # ── Summary output ──────────────────────────────────────────────

    def test_summary_format(self):
        """Test that summary() returns a readable string."""
        result = self.estimator.estimate("Why is the sky blue?")
        summary = result.summary()
        assert "sfreq" in summary.lower() or "Sentence Frequency" in summary
        assert result.model in summary


class TestCustomFreqTable:
    """Test custom frequency table support."""

    def test_custom_dict(self):
        """Test using a custom frequency dictionary."""
        custom = {"sky": 0.5, "blue": 0.3, "why": 0.8}
        estimator = FrequencyEstimator(
            model="generic", custom_freq_table=custom
        )
        result = estimator.estimate("Why is the sky blue?")
        # Should use custom frequencies for known words
        assert result.sfreq > 0

    def test_custom_table_file(self, tmp_path):
        """Test loading a custom frequency table from JSON file."""
        table_file = tmp_path / "custom_freq.json"
        table_file.write_text('{"hello": 0.5, "world": 0.3}')

        estimator = FrequencyEstimator(
            model="generic",
            custom_freq_table_path=str(table_file),
        )
        result = estimator.estimate("Hello world")
        assert result.sfreq > 0

    def test_custom_tsv_table(self, tmp_path):
        """Test loading a custom frequency table from TSV file."""
        table_file = tmp_path / "custom_freq.tsv"
        table_file.write_text("hello\t0.5\nworld\t0.3\n")

        estimator = FrequencyEstimator(
            model="generic",
            custom_freq_table_path=str(table_file),
        )
        result = estimator.estimate("Hello world")
        assert result.sfreq > 0
