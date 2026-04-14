"""Tokenizer utilities for Chinese and English text.

Provides language-aware tokenization that adapts to different text types.
"""

from __future__ import annotations

import re
import unicodedata


def detect_language(text: str) -> str:
    """Detect whether text is primarily Chinese or English.

    Args:
        text: Input text string.

    Returns:
        "zh" if the text contains significant Chinese characters, otherwise "en".
    """
    chinese_char_count = sum(1 for ch in text if _is_chinese_char(ch))
    total_alpha = sum(1 for ch in text if ch.isalpha())
    if total_alpha == 0:
        return "en"
    return "zh" if chinese_char_count / total_alpha > 0.3 else "en"


def tokenize(text: str, language: str | None = None) -> list[str]:
    """Tokenize text into words based on language.

    Args:
        text: Input text to tokenize.
        language: Language code ("en" or "zh"). Auto-detected if None.

    Returns:
        List of tokenized words (lowercased for English).
    """
    if language is None:
        language = detect_language(text)

    if language == "zh":
        return _tokenize_chinese(text)
    return _tokenize_english(text)


def _tokenize_english(text: str) -> list[str]:
    """Tokenize English text into lowercase words.

    Strips digits and punctuation, splits on whitespace.
    """
    text = text.lower()
    # Remove digits
    text = re.sub(r"\d+", " ", text)
    # Keep only word chars and spaces (including hyphens within words)
    text = re.sub(r"[^\w\s-]", " ", text)
    # Split hyphenated words
    tokens = []
    for word in text.split():
        word = word.strip("-")
        if not word:
            continue
        if "-" in word:
            parts = [p for p in word.split("-") if p]
            tokens.extend(parts)
        else:
            tokens.append(word)
    # Remove single-character tokens that are not meaningful
    return [t for t in tokens if len(t) > 1 or t in ("i", "a")]


def _tokenize_chinese(text: str) -> list[str]:
    """Tokenize Chinese text using jieba.

    Falls back to character-level segmentation if jieba is unavailable.
    """
    try:
        import jieba

        # Suppress jieba's initialization logs
        jieba.setLogLevel(20)
        words = jieba.lcut(text)
    except ImportError:
        # Fallback: character-level for Chinese, word-level for mixed content
        words = _fallback_chinese_tokenize(text)

    # Clean tokens: strip whitespace, remove pure punctuation/digits
    cleaned = []
    for w in words:
        w = w.strip()
        if not w:
            continue
        # Skip pure punctuation or pure digits
        if re.fullmatch(r"[\W\d]+", w, re.UNICODE):
            continue
        cleaned.append(w)
    return cleaned


def _fallback_chinese_tokenize(text: str) -> list[str]:
    """Simple fallback tokenizer for Chinese text without jieba."""
    tokens = []
    current_type = None  # "zh", "en", or None
    current_token = []

    for ch in text:
        if _is_chinese_char(ch):
            if current_type == "en" and current_token:
                tokens.append("".join(current_token).lower())
                current_token = []
            tokens.append(ch)
            current_type = "zh"
        elif ch.isalpha():
            if current_type == "zh":
                current_token = []
            current_type = "en"
            current_token.append(ch)
        else:
            if current_type == "en" and current_token:
                tokens.append("".join(current_token).lower())
                current_token = []
            current_type = None

    if current_type == "en" and current_token:
        tokens.append("".join(current_token).lower())

    return tokens


def _is_chinese_char(ch: str) -> bool:
    """Check if a character is a CJK Unified Ideograph."""
    cp = ord(ch)
    return (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0x20000 <= cp <= 0x2A6DF)
        or (0x2A700 <= cp <= 0x2B73F)
        or (0x2B740 <= cp <= 0x2B81F)
        or (0x2B820 <= cp <= 0x2CEAF)
        or (0xF900 <= cp <= 0xFAFF)
        or (0x2F800 <= cp <= 0x2FA1F)
    )
