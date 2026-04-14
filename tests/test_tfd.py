"""Tests for TFD (Textual Frequency Distillation) module."""

import json
import pytest

from adamopt.tfd import TFDDistiller


class TestTFDDistiller:
    """Test suite for TFDDistiller."""

    @pytest.fixture
    def sample_corpus(self, tmp_path):
        """Create a sample corpus file for testing."""
        corpus = tmp_path / "corpus.txt"
        corpus.write_text(
            "The sky is blue and the sun is bright\n"
            "Why is the sky blue during the day\n"
            "The weather is nice today and the sky is clear\n"
            "Blue sky and sunshine make people happy\n"
            "The sky turns orange during sunset\n"
            "Stars appear in the sky at night\n"
            "Birds fly across the blue sky\n"
            "Clouds float in the sky above\n"
            "The morning sky is beautiful\n"
            "Rain falls from the dark sky\n"
        )
        return corpus

    @pytest.fixture
    def chinese_corpus(self, tmp_path):
        """Create a sample Chinese corpus file."""
        corpus = tmp_path / "zh_corpus.txt"
        corpus.write_text(
            "天空是蓝色的非常美丽\n"
            "今天天气很好阳光明媚\n"
            "蓝天白云让人心情愉快\n"
            "傍晚天空变成橙色\n"
            "晚上可以看到星星\n"
        )
        return corpus

    def test_count_corpus_frequencies(self, sample_corpus):
        """Test that corpus word counting produces valid frequencies."""
        distiller = TFDDistiller(model="generic", language="en")
        freqs = distiller.count_corpus_frequencies(sample_corpus, language="en")
        assert len(freqs) > 0
        # Frequencies should sum to ~1.0
        total = sum(freqs.values())
        assert abs(total - 1.0) < 0.01
        # "sky" should be a frequent word in this corpus
        assert "sky" in freqs
        assert freqs["sky"] > 0

    def test_count_chinese_corpus(self, chinese_corpus):
        """Test Chinese corpus word counting."""
        distiller = TFDDistiller(model="generic", language="zh")
        freqs = distiller.count_corpus_frequencies(chinese_corpus, language="zh")
        assert len(freqs) > 0
        total = sum(freqs.values())
        assert abs(total - 1.0) < 0.01

    def test_count_corpus_file_not_found(self):
        """Test that missing corpus raises FileNotFoundError."""
        distiller = TFDDistiller(model="generic")
        with pytest.raises(FileNotFoundError):
            distiller.count_corpus_frequencies("/nonexistent/file.txt")

    def test_get_baseline_frequencies(self):
        """Test baseline frequency lookup from wordfreq."""
        distiller = TFDDistiller(model="generic")
        vocab = {"sky", "blue", "the", "hello", "world"}
        baseline = distiller.get_baseline_frequencies(vocab, language="en")
        assert len(baseline) == len(vocab)
        for word in vocab:
            assert word in baseline
            assert baseline[word] > 0

    def test_align_and_merge(self):
        """Test the align_and_merge algorithm."""
        baseline = {"sky": 0.01, "blue": 0.02, "sun": 0.005, "rare": 0.0001}
        corpus = {"sky": 0.05, "blue": 0.03, "cloud": 0.02}
        merged = TFDDistiller.align_and_merge(baseline, corpus)
        assert len(merged) > 0
        # Should contain words from both sources
        assert "sky" in merged
        assert "blue" in merged
        assert "sun" in merged  # from baseline only
        assert "cloud" in merged  # from corpus only
        # Should be normalized
        total = sum(merged.values())
        assert abs(total - 1.0) < 0.01

    def test_distill_full_pipeline(self, sample_corpus, tmp_path):
        """Test the full TFD distillation pipeline."""
        output_path = tmp_path / "merged_freq.json"
        distiller = TFDDistiller(
            model="test-model",
            language="en",
            cache_dir=str(tmp_path / "cache"),
        )
        result = distiller.distill(
            generated_corpus_path=sample_corpus,
            output_path=str(output_path),
            language="en",
        )
        # Check result fields
        assert result["vocab_size"] > 0
        assert result["corpus_vocab_size"] > 0
        assert result["model"] == "test-model"
        assert result["output_path"] == str(output_path)
        # Check output file exists and is valid JSON
        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)
        assert len(data) == result["vocab_size"]

    def test_distill_saves_to_cache(self, sample_corpus, tmp_path):
        """Test that distillation saves to model cache."""
        cache_dir = tmp_path / "cache"
        distiller = TFDDistiller(
            model="qwen2.5-7b",
            language="en",
            cache_dir=str(cache_dir),
        )
        result = distiller.distill(
            generated_corpus_path=sample_corpus,
            language="en",
            save_to_cache=True,
        )
        assert result["cache_path"] is not None

    def test_distill_no_cache(self, sample_corpus, tmp_path):
        """Test distillation with cache disabled."""
        output = tmp_path / "out.json"
        distiller = TFDDistiller(
            model="generic",
            language="en",
            cache_dir=str(tmp_path / "cache"),
        )
        result = distiller.distill(
            generated_corpus_path=sample_corpus,
            output_path=str(output),
            language="en",
            save_to_cache=False,
        )
        assert result["output_path"] == str(output)
        assert result["cache_path"] is None

    def test_distill_empty_corpus_raises(self, tmp_path):
        """Test that empty corpus raises ValueError."""
        empty = tmp_path / "empty.txt"
        empty.write_text("")
        distiller = TFDDistiller(model="generic", language="en")
        with pytest.raises(ValueError, match="No words found"):
            distiller.distill(empty, language="en")


class TestTFDIntegration:
    """Test TFD integration with FrequencyEstimator."""

    def test_estimator_uses_tfd_table(self, tmp_path):
        """Test that FrequencyEstimator picks up a TFD-cached table."""
        from adamopt.frequency import FrequencyEstimator

        cache_dir = tmp_path / "cache"

        # Create a corpus and run TFD
        corpus = tmp_path / "corpus.txt"
        corpus.write_text(
            "The sky is blue\n" * 20
            + "The sun is bright\n" * 20
        )

        distiller = TFDDistiller(
            model="test-tfd-model",
            language="en",
            cache_dir=str(cache_dir),
        )
        distiller.distill(corpus, language="en", save_to_cache=True)

        # Now create an estimator with the same model and cache dir
        estimator = FrequencyEstimator(
            model="test-tfd-model",
            language="en",
            cache_dir=str(cache_dir),
        )
        result = estimator.estimate("The sky is blue")
        assert result.sfreq > 0
        # Should use model cache (level 2)
        assert result.freq_source is not None
        assert result.freq_source.level == 2
        assert "test-tfd-model" in result.freq_source.name.lower()
