"""Core content locking — Identify and protect content that must not be modified.

Implements the "Semantic Fidelity Golden Rule" from Adam's Law optimization:
content that would cause semantic drift if changed is locked with placeholders
before optimization, and restored afterward.

Lock categories (Chinese & English):
- Core entities: proper nouns, technical terms, brand names
- Core instruction words: task-defining verbs in prompts
- Hard constraints: format/length/output rules
- Logic/negation keywords: words that determine logical flow
- Objective information units: numbers, formulas, code, dates
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from adamopt.tokenizer import detect_language


# ── Built-in lock patterns ─────────────────────────────────────────────────

# Logic/negation keywords that must never be changed
LOGIC_KEYWORDS_EN: set[str] = {
    "not", "no", "never", "neither", "nor", "none", "cannot", "can't",
    "don't", "doesn't", "didn't", "won't", "wouldn't", "shouldn't",
    "mustn't", "isn't", "aren't", "wasn't", "weren't", "haven't",
    "hasn't", "hadn't", "if", "then", "else", "unless", "only",
    "must", "shall", "always", "exactly", "strictly", "exclusively",
    "and", "or", "but", "either", "both", "all", "each", "every",
}

LOGIC_KEYWORDS_ZH: set[str] = {
    "不", "非", "没", "没有", "别", "勿", "无", "未", "莫", "休",
    "不要", "不能", "不可以", "不得", "不允许", "禁止",
    "且", "或", "和", "以及", "并且", "或者",
    "如果", "那么", "否则", "除非", "只有", "仅",
    "必须", "一定", "务必", "一定要", "绝对",
    "每", "所有", "全部", "任何", "各", "每个",
}

# Core instruction verbs commonly used in prompts
INSTRUCTION_WORDS_EN: set[str] = {
    "translate", "summarize", "explain", "describe", "analyze", "classify",
    "generate", "write", "create", "list", "compare", "evaluate",
    "calculate", "solve", "define", "extract", "identify", "convert",
    "rewrite", "paraphrase", "simplify", "elaborate", "outline",
    "code", "implement", "debug", "review", "optimize", "design",
}

INSTRUCTION_WORDS_ZH: set[str] = {
    "翻译", "总结", "解释", "描述", "分析", "分类",
    "生成", "写", "创建", "列出", "比较", "评估",
    "计算", "解决", "定义", "提取", "识别", "转换",
    "改写", "简化", "展开", "概述", "编写",
    "编码", "实现", "调试", "审查", "优化", "设计",
}

# Regex patterns for objective information units
OBJECTIVE_PATTERNS: list[re.Pattern] = [
    # Numbers with optional units (e.g., "300字", "90%", "3.14", "$100")
    re.compile(r'\d+[\.\d]*\s*[%％‰]'),
    re.compile(r'[＄$€£¥]\s*\d+[\.\d]*'),
    re.compile(r'\d+[\.\d]*\s*[a-zA-Z\u4e00-\u9fff]{0,4}(?=\s|$|[，。,\.!！?？])'),
    # Dates and times
    re.compile(r'\d{4}[\-/年]\d{1,2}[\-/月]\d{1,2}[日]?'),
    re.compile(r'\d{1,2}:\d{2}(?::\d{2})?'),
    # Code-like patterns (backtick blocks, variable names)
    re.compile(r'`[^`]+`'),
    re.compile(r'```[\s\S]*?```'),
    # Formulas (simple math expressions)
    re.compile(r'[a-zA-Z_]\w*\s*[=<>≥≤≠±]+\s*[\w\.\+\-\*/]+'),
    # URLs and email addresses
    re.compile(r'https?://\S+'),
    re.compile(r'[\w\.\-]+@[\w\.\-]+\.\w+'),
    # Format constraints in quotes
    re.compile(r'["\u201c][^"\u201d]*["\u201d]'),
    re.compile(r"['\u2018][^'\u2019]*['\u2019]"),
]

# Format/constraint patterns (matched as whole phrases)
CONSTRAINT_PATTERNS: list[re.Pattern] = [
    # English
    re.compile(r'in\s+JSON\s+format', re.IGNORECASE),
    re.compile(r'in\s+\w+\s+format', re.IGNORECASE),
    re.compile(r'no\s+more\s+than\s+\d+\s*\w*', re.IGNORECASE),
    re.compile(r'at\s+(?:most|least)\s+\d+\s*\w*', re.IGNORECASE),
    re.compile(r'within\s+\d+\s*\w*', re.IGNORECASE),
    re.compile(r'exactly\s+\d+\s*\w*', re.IGNORECASE),
    re.compile(r'must\s+(?:be|include|contain|have)', re.IGNORECASE),
    # Chinese
    re.compile(r'用\w+格式(?:输出|回答|返回)'),
    re.compile(r'不超过\d+[字词个条]'),
    re.compile(r'不少于\d+[字词个条]'),
    re.compile(r'分\d+[点条步]'),
    re.compile(r'(?:禁止|不要|不允许|避免)\w{2,8}'),
]


@dataclass
class LockedSpan:
    """A span of text that is locked (must not be modified).

    Attributes:
        start: Start index in the original text.
        end: End index in the original text.
        text: The locked text content.
        category: Why it was locked (e.g., "entity", "logic", "constraint").
        placeholder: The placeholder token used to replace it.
    """
    start: int
    end: int
    text: str
    category: str
    placeholder: str


@dataclass
class LockResult:
    """Result of content locking.

    Attributes:
        original_text: The original input text.
        masked_text: Text with locked content replaced by placeholders.
        locked_spans: List of locked spans with metadata.
    """
    original_text: str
    masked_text: str
    locked_spans: list[LockedSpan] = field(default_factory=list)

    def restore(self, optimized_text: str) -> str:
        """Restore all placeholders in the optimized text back to original content.

        Args:
            optimized_text: Text that has been optimized (with placeholders).

        Returns:
            Text with all placeholders replaced by original locked content.
        """
        result = optimized_text
        # Sort by placeholder length descending to avoid partial replacement issues
        for span in sorted(self.locked_spans, key=lambda s: len(s.placeholder), reverse=True):
            result = result.replace(span.placeholder, span.text)
        return result


class ContentLocker:
    """Identify and lock content that must not be modified during optimization.

    Combines regex patterns, keyword matching, and optional NER to detect
    entities, logic keywords, constraints, and objective information that
    must be preserved exactly.

    Example:
        >>> locker = ContentLocker()
        >>> result = locker.lock("Translate the following to JSON format: x+y=z")
        >>> print(result.masked_text)  # Locked content replaced with [LOCK_n]
        >>> restored = result.restore(optimized_text)
    """

    def __init__(
        self,
        *,
        custom_lock_words: set[str] | None = None,
        use_ner: bool = False,
        language: str | None = None,
    ) -> None:
        """Initialize the content locker.

        Args:
            custom_lock_words: Additional user-defined words/phrases to lock.
            use_ner: Whether to use spaCy NER for entity detection.
                Requires spaCy and a language model to be installed.
            language: Force language ("en" or "zh"). Auto-detected if None.
        """
        self.custom_lock_words = custom_lock_words or set()
        self.use_ner = use_ner
        self.language = language
        self._ner_model = None

    def lock(self, text: str) -> LockResult:
        """Lock all protected content in the text.

        Process order:
        1. Detect language
        2. Find objective patterns (numbers, code, URLs, formulas)
        3. Find constraint patterns (format/length rules)
        4. Find logic/negation keywords
        5. Find instruction words
        6. Find NER entities (if enabled)
        7. Find custom lock words
        8. Merge overlapping spans
        9. Replace locked spans with placeholders

        Args:
            text: Input text to analyze.

        Returns:
            LockResult with masked text and locked span metadata.
        """
        lang = self.language or detect_language(text)
        spans: list[LockedSpan] = []
        counter = 0

        def _add_span(start: int, end: int, category: str) -> None:
            nonlocal counter
            placeholder = f"[LOCK_{counter}]"
            spans.append(LockedSpan(
                start=start, end=end,
                text=text[start:end],
                category=category,
                placeholder=placeholder,
            ))
            counter += 1

        # 1. Objective patterns (numbers, code, URLs, formulas, dates)
        for pattern in OBJECTIVE_PATTERNS:
            for match in pattern.finditer(text):
                _add_span(match.start(), match.end(), "objective")

        # 2. Constraint patterns (format rules, length limits)
        for pattern in CONSTRAINT_PATTERNS:
            for match in pattern.finditer(text):
                _add_span(match.start(), match.end(), "constraint")

        # 3. Logic/negation keywords
        logic_words = LOGIC_KEYWORDS_ZH if lang == "zh" else LOGIC_KEYWORDS_EN
        for word in logic_words:
            # Use word boundary matching for English, direct match for Chinese
            if lang == "zh":
                pattern = re.compile(re.escape(word))
            else:
                pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
            for match in pattern.finditer(text):
                _add_span(match.start(), match.end(), "logic")

        # 4. Instruction words
        instr_words = INSTRUCTION_WORDS_ZH if lang == "zh" else INSTRUCTION_WORDS_EN
        for word in instr_words:
            if lang == "zh":
                pattern = re.compile(re.escape(word))
            else:
                pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
            for match in pattern.finditer(text):
                _add_span(match.start(), match.end(), "instruction")

        # 5. NER entities (optional)
        if self.use_ner:
            ner_spans = self._detect_entities(text, lang)
            for start, end in ner_spans:
                _add_span(start, end, "entity")

        # 6. Custom lock words
        for word in self.custom_lock_words:
            for match in re.finditer(re.escape(word), text):
                _add_span(match.start(), match.end(), "custom")

        # 7. Merge overlapping spans and build masked text
        merged = self._merge_spans(spans, len(text))
        masked_text = self._build_masked_text(text, merged)

        return LockResult(
            original_text=text,
            masked_text=masked_text,
            locked_spans=merged,
        )

    def _detect_entities(self, text: str, lang: str) -> list[tuple[int, int]]:
        """Detect named entities using spaCy NER.

        Returns list of (start, end) character offsets.
        """
        try:
            import spacy
        except ImportError:
            return []

        if self._ner_model is None:
            model_name = "zh_core_web_sm" if lang == "zh" else "en_core_web_sm"
            try:
                self._ner_model = spacy.load(model_name)
            except OSError:
                return []

        doc = self._ner_model(text)
        return [(ent.start_char, ent.end_char) for ent in doc.ents]

    @staticmethod
    def _merge_spans(spans: list[LockedSpan], text_len: int) -> list[LockedSpan]:
        """Merge overlapping locked spans, keeping the larger one on conflict."""
        if not spans:
            return []

        # Sort by start position, then by length descending
        sorted_spans = sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))

        merged: list[LockedSpan] = []
        for span in sorted_spans:
            if span.start >= text_len or span.end <= 0:
                continue
            # Check overlap with last merged span
            if merged and span.start < merged[-1].end:
                # Overlapping: keep the longer one
                if (span.end - span.start) > (merged[-1].end - merged[-1].start):
                    merged[-1] = span
                # Otherwise skip this span (the existing one is longer or equal)
            else:
                merged.append(span)

        # Re-assign sequential placeholders
        for i, span in enumerate(merged):
            span.placeholder = f"[LOCK_{i}]"

        return merged

    @staticmethod
    def _build_masked_text(text: str, spans: list[LockedSpan]) -> str:
        """Replace locked spans with their placeholders."""
        if not spans:
            return text

        # Build from right to left to preserve indices
        result = text
        for span in reversed(spans):
            result = result[:span.start] + span.placeholder + result[span.end:]

        return result
