"""Tests for tokenizer utilities."""

import pytest

from adamopt.tokenizer import detect_language, tokenize


class TestDetectLanguage:
    """Test language detection."""

    def test_detect_english(self):
        assert detect_language("Hello, how are you?") == "en"

    def test_detect_chinese(self):
        assert detect_language("你好，今天天气怎么样？") == "zh"

    def test_detect_mixed_mostly_chinese(self):
        assert detect_language("今天学习了Python编程") == "zh"

    def test_detect_empty(self):
        assert detect_language("") == "en"

    def test_detect_numbers_only(self):
        assert detect_language("12345") == "en"


class TestTokenize:
    """Test tokenization."""

    def test_english_basic(self):
        tokens = tokenize("Hello world", language="en")
        assert "hello" in tokens
        assert "world" in tokens

    def test_english_removes_digits(self):
        tokens = tokenize("I have 3 cats", language="en")
        assert "3" not in tokens
        assert "cats" in tokens

    def test_english_handles_punctuation(self):
        tokens = tokenize("Hello, world! How are you?", language="en")
        assert "," not in tokens
        assert "!" not in tokens

    def test_english_hyphenated_words(self):
        tokens = tokenize("state-of-the-art model", language="en")
        assert "state" in tokens
        assert "art" in tokens

    def test_chinese_tokenization(self):
        tokens = tokenize("天空为什么是蓝色的", language="zh")
        assert len(tokens) > 0
        # Should have Chinese tokens
        assert any(ord(t[0]) > 0x4E00 for t in tokens if t)

    def test_auto_detect_and_tokenize(self):
        tokens = tokenize("Why is the sky blue?")
        assert "sky" in tokens or "blue" in tokens
