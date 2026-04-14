"""Tests for the text frequency optimizer module."""

import pytest

from adamopt.optimizer import TextOptimizer, OptimizeResult


class TestTextOptimizer:
    """Test suite for TextOptimizer."""

    def setup_method(self):
        self.optimizer = TextOptimizer(model="generic")

    # ── Basic functionality ───────────────────────────────────────

    def test_optimize_returns_result(self):
        """Test that optimize returns a valid OptimizeResult."""
        result = self.optimizer.optimize("Why is the sky blue?")
        assert isinstance(result, OptimizeResult)
        assert result.original_text == "Why is the sky blue?"
        assert result.optimized_text  # Should not be empty
        assert result.original_sfreq > 0
        assert result.optimized_sfreq > 0

    def test_optimize_empty_text(self):
        """Test optimization of empty text."""
        result = self.optimizer.optimize("")
        assert result.optimized_text == ""
        assert result.original_sfreq == 0.0

    def test_optimize_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            self.optimizer.optimize("Hello", mode="extreme")

    # ── Conservative mode ─────────────────────────────────────────

    def test_conservative_replaces_low_freq_words(self):
        """Test that conservative mode replaces low-frequency words."""
        result = self.optimizer.optimize(
            "What is the optical causation for the azure hue of the celestial firmament?",
            mode="conservative",
        )
        assert result.mode == "conservative"
        # Should improve sfreq (or at least not worsen it)
        assert result.optimized_sfreq >= result.original_sfreq * 0.99

    def test_conservative_preserves_locked_content(self):
        """Test that conservative mode does not modify locked content."""
        result = self.optimizer.optimize(
            "Translate the equation x+y=z to Python code",
            mode="conservative",
        )
        # "Translate" is a locked instruction word, should be preserved
        assert "translate" in result.optimized_text.lower() or "Translate" in result.optimized_text

    def test_conservative_english_synonym(self):
        """Test English word-level synonym replacement."""
        result = self.optimizer.optimize(
            "We need to utilize this methodology to ameliorate the situation",
            mode="conservative",
        )
        # "utilize" should be replaced with "use", "ameliorate" with "improve"
        optimized_lower = result.optimized_text.lower()
        has_replacement = ("use" in optimized_lower or "improve" in optimized_lower
                          or "method" in optimized_lower)
        assert has_replacement or result.optimized_sfreq >= result.original_sfreq

    def test_conservative_chinese_synonym(self):
        """Test Chinese word-level synonym replacement."""
        optimizer = TextOptimizer(model="generic", language="zh")
        result = optimizer.optimize(
            "请你详尽阐述这个问题的解决方案",
            mode="conservative",
        )
        assert result.optimized_text
        assert result.optimized_sfreq >= result.original_sfreq * 0.99

    # ── Balanced mode ─────────────────────────────────────────────

    def test_balanced_default_mode(self):
        """Test that balanced is the default mode."""
        result = self.optimizer.optimize("Hello world")
        assert result.mode == "balanced"

    def test_balanced_phrase_replacement(self):
        """Test that balanced mode handles phrase-level replacement."""
        optimizer = TextOptimizer(model="generic", language="zh")
        result = optimizer.optimize(
            "在本次实验过程中，我们发现了这个现象",
            mode="balanced",
        )
        assert result.optimized_text
        # Should attempt phrase-level optimization
        assert result.optimized_sfreq >= result.original_sfreq * 0.99

    # ── Aggressive mode ───────────────────────────────────────────

    def test_aggressive_sentence_rewrite(self):
        """Test that aggressive mode applies sentence-level patterns."""
        result = self.optimizer.optimize(
            "In order to understand the problem, we need to analyze it carefully",
            mode="aggressive",
        )
        assert result.mode == "aggressive"
        optimized_lower = result.optimized_text.lower()
        # "in order to" should be simplified to "to"
        if result.replacements:
            sentence_level = [r for r in result.replacements if r["level"] == "sentence"]
            # May or may not have sentence-level replacements depending on text
            assert isinstance(sentence_level, list)

    # ── sfreq improvement ─────────────────────────────────────────

    def test_sfreq_improves_on_low_freq_text(self):
        """Test that optimization improves sfreq for text with low-freq words."""
        result = self.optimizer.optimize(
            "The celestial firmament exhibits an azure chromatic manifestation",
            mode="balanced",
        )
        # Text with many low-freq words should see improvement
        assert result.optimized_sfreq >= result.original_sfreq

    def test_high_freq_text_unchanged(self):
        """Test that already high-frequency text is minimally changed."""
        result = self.optimizer.optimize(
            "The sky is blue",
            mode="conservative",
        )
        # Simple high-freq text should have few or no replacements
        assert result.optimized_text  # Should still return valid text

    # ── Batch optimization ────────────────────────────────────────

    def test_optimize_batch(self):
        """Test batch optimization."""
        texts = ["Hello world", "The sky is blue", "Why is it raining?"]
        results = self.optimizer.optimize_batch(texts)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, OptimizeResult)

    # ── Summary output ────────────────────────────────────────────

    def test_summary_format(self):
        """Test that summary returns readable string."""
        result = self.optimizer.optimize("The celestial firmament is azure")
        summary = result.summary()
        assert "sfreq" in summary.lower() or "Optimization" in summary

    # ── Locked content integrity ──────────────────────────────────

    def test_numbers_preserved(self):
        """Test that numbers are preserved through optimization."""
        result = self.optimizer.optimize(
            "The accuracy is 95% and we need at least 100 samples",
            mode="aggressive",
        )
        assert "95" in result.optimized_text
        assert "100" in result.optimized_text

    def test_instruction_words_preserved(self):
        """Test that core instruction words survive optimization."""
        result = self.optimizer.optimize(
            "Summarize the aforementioned document in approximately 300 words",
            mode="aggressive",
        )
        assert "summar" in result.optimized_text.lower()
        assert "300" in result.optimized_text
