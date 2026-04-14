"""Text frequency optimizer — Three-tier optimization based on Adam's Law.

Implements the core optimization loop:
1. Lock protected content (ContentLocker)
2. Identify low-frequency bottlenecks (FrequencyEstimator)
3. Generate higher-frequency candidates (synonym/paraphrase)
4. Select best candidates by sfreq
5. Restore locked content

Three modes:
- conservative: Word-level one-to-one synonym replacement only
- balanced: Word-level + phrase-level paraphrase (default, recommended)
- aggressive: Word + phrase + sentence-level rewrite
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from adamopt.content_locker import ContentLocker, LockResult
from adamopt.frequency import FrequencyEstimator, FrequencyResult
from adamopt.tokenizer import detect_language, tokenize
from adamopt.stopwords import StopwordManager


class OptimizeMode(str, Enum):
    """Optimization mode (tier)."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


@dataclass
class OptimizeResult:
    """Result of text frequency optimization.

    Attributes:
        original_text: The input text before optimization.
        optimized_text: The output text after optimization.
        original_sfreq: Sentence-level frequency before optimization.
        optimized_sfreq: Sentence-level frequency after optimization.
        sfreq_improvement: Absolute improvement (optimized - original).
        sfreq_ratio: Improvement ratio (optimized / original).
        mode: The optimization mode used.
        replacements: List of (original_word, replacement, old_freq, new_freq).
        locked_count: Number of locked (protected) spans.
        language: Detected or specified language.
        model: Target model name.
    """
    original_text: str
    optimized_text: str
    original_sfreq: float
    optimized_sfreq: float
    sfreq_improvement: float
    sfreq_ratio: float
    mode: str
    replacements: list[dict[str, Any]] = field(default_factory=list)
    locked_count: int = 0
    language: str = "en"
    model: str = "generic"

    def summary(self) -> str:
        """Return a human-readable summary of the optimization."""
        pct = (self.sfreq_ratio - 1) * 100 if self.sfreq_ratio > 0 else 0
        lines = [
            f"Optimization Result (mode={self.mode}, model={self.model})",
            f"  Original sfreq:  {self.original_sfreq:.8f}",
            f"  Optimized sfreq: {self.optimized_sfreq:.8f}",
            f"  Improvement:     {pct:+.1f}%",
            f"  Locked spans:    {self.locked_count}",
            f"  Replacements:    {len(self.replacements)}",
        ]
        if self.replacements:
            lines.append("  Changes:")
            for r in self.replacements[:10]:
                lines.append(
                    f"    '{r['original']}' → '{r['replacement']}' "
                    f"(freq: {r['old_freq']:.6f} → {r['new_freq']:.6f})"
                )
            if len(self.replacements) > 10:
                lines.append(f"    ... and {len(self.replacements) - 10} more")
        return "\n".join(lines)


# ── Synonym dictionaries (built-in high-frequency replacements) ──────────

# English: Latin-origin → Anglo-Saxon simple words
EN_SYNONYM_MAP: dict[str, list[str]] = {
    "utilize": ["use"],
    "demonstrate": ["show"],
    "facilitate": ["help", "make easy"],
    "implement": ["do", "carry out"],
    "comprehend": ["understand"],
    "approximately": ["about", "around"],
    "subsequent": ["next", "later"],
    "sufficient": ["enough"],
    "additional": ["more", "extra"],
    "considerable": ["big", "large", "much"],
    "commence": ["start", "begin"],
    "terminate": ["end", "stop"],
    "obtain": ["get"],
    "require": ["need"],
    "purchase": ["buy"],
    "attempt": ["try"],
    "assist": ["help"],
    "indicate": ["show", "point to"],
    "accomplish": ["do", "finish"],
    "manufacture": ["make"],
    "numerous": ["many"],
    "elaborate": ["detailed", "explain more"],
    "endeavor": ["try", "effort"],
    "ascertain": ["find out"],
    "elucidate": ["explain", "clarify"],
    "inaugurate": ["start", "open"],
    "methodology": ["method", "way"],
    "paradigm": ["model", "example"],
    "constitute": ["make up", "form"],
    "delineate": ["describe", "outline"],
    "substantiate": ["prove", "support"],
    "predominant": ["main", "major"],
    "enhancement": ["improvement"],
    "deteriorate": ["worsen", "get worse"],
    "encompass": ["include", "cover"],
    "proficiency": ["skill"],
    "cognitive": ["mental", "thinking"],
    "ameliorate": ["improve", "make better"],
    "expedite": ["speed up"],
    "proliferate": ["spread", "grow"],
    "exacerbate": ["worsen", "make worse"],
    "disseminate": ["spread", "share"],
    "juxtapose": ["compare", "place side by side"],
    "corroborate": ["confirm", "support"],
    "illuminate": ["explain", "light up"],
    "articulate": ["express", "state clearly"],
    "consequently": ["so", "therefore"],
    "nevertheless": ["still", "however"],
    "notwithstanding": ["despite", "even so"],
    "aforementioned": ["above", "earlier"],
    "henceforth": ["from now on"],
    "pertaining": ["about", "related to"],
    "celestial": ["sky", "heavenly"],
    "firmament": ["sky"],
    "azure": ["blue"],
    "chromatic": ["color"],
    "manifestation": ["sign", "show"],
    "causation": ["cause"],
    "optical": ["light", "visual"],
}

# Chinese: bookish/formal → colloquial high-frequency
ZH_SYNONYM_MAP: dict[str, list[str]] = {
    "详尽阐述": ["详细说明", "详细讲"],
    "务必": ["一定要", "一定"],
    "罹患": ["得了", "患上"],
    "恶性肿瘤": ["癌症"],
    "鉴于": ["因为", "由于"],
    "倘若": ["如果"],
    "抑或": ["或者", "还是"],
    "亟需": ["急需"],
    "甄别": ["识别", "分辨"],
    "遴选": ["选择", "挑选"],
    "赓续": ["继续"],
    "裨益": ["好处", "帮助"],
    "攸关": ["关系到"],
    "勠力": ["合力", "一起努力"],
    "殊为": ["非常", "特别"],
    "悉数": ["全部"],
    "翔实": ["详细"],
    "竭力": ["尽力"],
    "迄今": ["到现在"],
    "须知": ["要知道"],
    "统筹": ["统一安排"],
    "契合": ["符合", "吻合"],
    "彰显": ["显示", "表现"],
    "旨在": ["目的是"],
    "涵盖": ["包括", "包含"],
    "逾越": ["超过"],
    "抒发": ["表达"],
    "缘由": ["原因"],
    "端倪": ["迹象", "苗头"],
    "权衡": ["考虑", "衡量"],
    "举措": ["措施", "做法"],
    "遐想": ["想象"],
    "呈现出显著的": ["有明显的"],
    "在本次实验过程中": ["在这次实验中"],
    "基于上述的分析": ["根据上面的分析"],
    "在这样的前提条件下": ["在这种情况下"],
    "通俗易懂的": ["好懂的", "易懂的"],
    "给出一个全面且": ["给一个全面、"],
    "解答过程中需要避免使用过于专业的术语": ["回答里别用太专业的词"],
    "同时保证内容的准确性与完整性": ["同时保证内容准确、完整"],
}


class TextOptimizer:
    """Three-tier text frequency optimizer based on Adam's Law.

    Optimizes text to use higher-frequency expressions while preserving
    semantic fidelity. Protected content (entities, logic words, constraints)
    is locked and never modified.

    Example:
        >>> optimizer = TextOptimizer(model="qwen2.5-7b")
        >>> result = optimizer.optimize(
        ...     "What is the optical causation for the azure hue of the celestial firmament?",
        ...     mode="balanced",
        ... )
        >>> print(result.optimized_text)
        >>> print(result.summary())
    """

    def __init__(
        self,
        model: str = "generic",
        language: str | None = None,
        *,
        custom_lock_words: set[str] | None = None,
        custom_synonyms: dict[str, list[str]] | None = None,
        use_ner: bool = False,
        low_freq_threshold: float = 0.0001,
        cache_dir: str | None = None,
    ) -> None:
        """Initialize the text optimizer.

        Args:
            model: Target LLM model name.
            language: Force language ("en" or "zh"). Auto-detected if None.
            custom_lock_words: Additional words/phrases to lock (never modify).
            custom_synonyms: User-provided synonym map {word: [synonyms]}.
            use_ner: Whether to use spaCy NER for entity locking.
            low_freq_threshold: Words below this frequency are optimization targets.
            cache_dir: Directory for frequency table caching.
        """
        self.model = model
        self.language = language
        self.low_freq_threshold = low_freq_threshold

        self._locker = ContentLocker(
            custom_lock_words=custom_lock_words,
            use_ner=use_ner,
            language=language,
        )
        self._estimator = FrequencyEstimator(
            model=model,
            language=language,
            low_freq_threshold=low_freq_threshold,
            cache_dir=cache_dir,
        )
        self._stopword_mgr = StopwordManager()
        self._custom_synonyms = custom_synonyms or {}

    def optimize(
        self,
        text: str,
        mode: str = "balanced",
    ) -> OptimizeResult:
        """Optimize text for higher frequency.

        Args:
            text: Input text to optimize.
            mode: Optimization mode — "conservative", "balanced", or "aggressive".

        Returns:
            OptimizeResult with optimized text, sfreq comparison, and change details.
        """
        if not text or not text.strip():
            return OptimizeResult(
                original_text=text,
                optimized_text=text,
                original_sfreq=0.0,
                optimized_sfreq=0.0,
                sfreq_improvement=0.0,
                sfreq_ratio=1.0,
                mode=mode,
            )

        # Validate mode
        try:
            opt_mode = OptimizeMode(mode.lower())
        except ValueError:
            raise ValueError(
                f"Invalid mode '{mode}'. Use 'conservative', 'balanced', or 'aggressive'."
            )

        lang = self.language or detect_language(text)

        # Step 1: Compute original sfreq
        original_result = self._estimator.estimate(text)

        # Step 2: Lock protected content
        lock_result = self._locker.lock(text)

        # Step 3: Apply optimization by mode
        if opt_mode == OptimizeMode.CONSERVATIVE:
            optimized_masked, replacements = self._optimize_conservative(
                lock_result.masked_text, lang
            )
        elif opt_mode == OptimizeMode.BALANCED:
            optimized_masked, replacements = self._optimize_balanced(
                lock_result.masked_text, lang
            )
        else:  # AGGRESSIVE
            optimized_masked, replacements = self._optimize_aggressive(
                lock_result.masked_text, lang
            )

        # Step 4: Restore locked content
        optimized_text = lock_result.restore(optimized_masked)

        # Step 5: Compute optimized sfreq
        optimized_result = self._estimator.estimate(optimized_text)

        # Calculate improvement
        orig_sfreq = original_result.sfreq
        opt_sfreq = optimized_result.sfreq
        improvement = opt_sfreq - orig_sfreq
        ratio = opt_sfreq / orig_sfreq if orig_sfreq > 0 else 1.0

        return OptimizeResult(
            original_text=text,
            optimized_text=optimized_text,
            original_sfreq=orig_sfreq,
            optimized_sfreq=opt_sfreq,
            sfreq_improvement=improvement,
            sfreq_ratio=ratio,
            mode=opt_mode.value,
            replacements=replacements,
            locked_count=len(lock_result.locked_spans),
            language=lang,
            model=self.model,
        )

    def optimize_batch(
        self, texts: list[str], mode: str = "balanced"
    ) -> list[OptimizeResult]:
        """Optimize a batch of texts.

        Args:
            texts: List of input texts.
            mode: Optimization mode.

        Returns:
            List of OptimizeResult objects.
        """
        return [self.optimize(text, mode=mode) for text in texts]

    # ── Conservative mode: word-level synonym replacement ────────────

    def _optimize_conservative(
        self, text: str, lang: str
    ) -> tuple[str, list[dict[str, Any]]]:
        """Word-level one-to-one synonym replacement.

        Only replaces individual low-frequency words with higher-frequency
        synonyms. Does not change sentence structure or word order.
        """
        synonyms = self._get_synonym_map(lang)
        replacements: list[dict[str, Any]] = []

        # Tokenize to find words and their positions
        result_text = text
        tokens = tokenize(text, language=lang)

        # Find low-frequency words that have synonyms
        for token in tokens:
            if self._stopword_mgr.is_stopword(token):
                continue
            if token.startswith("[LOCK_"):
                continue

            token_freq = self._get_word_freq(token, lang)
            if token_freq >= self.low_freq_threshold:
                continue

            # Find best synonym
            candidates = self._get_candidates(token, synonyms, lang)
            if not candidates:
                continue

            # Pick the highest-frequency candidate
            best = candidates[0]
            if best["freq"] <= token_freq:
                continue

            # Replace in text (case-preserving for English)
            replaced = self._replace_word_in_text(
                result_text, token, best["word"], lang
            )
            if replaced != result_text:
                replacements.append({
                    "original": token,
                    "replacement": best["word"],
                    "old_freq": token_freq,
                    "new_freq": best["freq"],
                    "level": "word",
                })
                result_text = replaced

        return result_text, replacements

    # ── Balanced mode: word + phrase level ────────────────────────────

    def _optimize_balanced(
        self, text: str, lang: str
    ) -> tuple[str, list[dict[str, Any]]]:
        """Word-level replacement + phrase-level paraphrase.

        First applies conservative word-level replacements, then identifies
        low-frequency phrases and replaces them with higher-frequency alternatives.
        """
        # First do word-level optimization
        result_text, replacements = self._optimize_conservative(text, lang)

        # Then do phrase-level optimization
        phrase_synonyms = self._get_synonym_map(lang)
        # Check multi-word entries (phrases)
        for phrase, candidates_list in phrase_synonyms.items():
            if " " not in phrase and len(phrase) < 4 and lang == "en":
                continue  # Skip single words (already handled)
            if lang == "zh" and len(phrase) < 3:
                continue

            if phrase in result_text:
                phrase_freq = self._estimate_phrase_freq(phrase, lang)
                if phrase_freq >= self.low_freq_threshold:
                    continue

                # Find best replacement
                best_replacement = None
                best_freq = phrase_freq
                for candidate in candidates_list:
                    cand_freq = self._estimate_phrase_freq(candidate, lang)
                    if cand_freq > best_freq:
                        best_freq = cand_freq
                        best_replacement = candidate

                if best_replacement:
                    result_text = result_text.replace(phrase, best_replacement, 1)
                    replacements.append({
                        "original": phrase,
                        "replacement": best_replacement,
                        "old_freq": phrase_freq,
                        "new_freq": best_freq,
                        "level": "phrase",
                    })

        return result_text, replacements

    # ── Aggressive mode: word + phrase + sentence rewrite ─────────────

    def _optimize_aggressive(
        self, text: str, lang: str
    ) -> tuple[str, list[dict[str, Any]]]:
        """Word + phrase + sentence-level rewrite.

        Applies balanced optimization first, then attempts sentence-level
        simplification by replacing complex patterns with simpler equivalents.
        """
        # First apply balanced optimization
        result_text, replacements = self._optimize_balanced(text, lang)

        # Sentence-level simplification patterns
        if lang == "zh":
            patterns = [
                (r'针对下述用户提出的问题', '针对下面用户的问题'),
                (r'给出一个全面且通俗易懂的解答', '给一个全面、好懂的回答'),
                (r'解答过程中需要', '回答里'),
                (r'过于专业的术语', '太专业的词'),
                (r'的准确性与完整性', '准确、完整'),
                (r'在\s*此\s*基础\s*上', '然后'),
                (r'与此同时', '同时'),
                (r'就目前而言', '现在'),
                (r'从某种程度上来说', '在一定程度上'),
            ]
        else:
            patterns = [
                (r'\bin order to\b', 'to'),
                (r'\bdue to the fact that\b', 'because'),
                (r'\bin the event that\b', 'if'),
                (r'\bat this point in time\b', 'now'),
                (r'\bwith regard to\b', 'about'),
                (r'\bin spite of the fact that\b', 'although'),
                (r'\bfor the purpose of\b', 'to'),
                (r'\bin the near future\b', 'soon'),
                (r'\bhas the ability to\b', 'can'),
                (r'\bit is necessary that\b', 'must'),
                (r'\bprior to\b', 'before'),
                (r'\bsubsequent to\b', 'after'),
                (r'\bin accordance with\b', 'following'),
                (r'\bwith respect to\b', 'about'),
            ]

        for pattern_str, replacement in patterns:
            pattern = re.compile(pattern_str, re.IGNORECASE if lang == "en" else 0)
            match = pattern.search(result_text)
            if match:
                old_phrase = match.group()
                old_freq = self._estimate_phrase_freq(old_phrase, lang)
                new_freq = self._estimate_phrase_freq(replacement, lang)
                result_text = pattern.sub(replacement, result_text, count=1)
                replacements.append({
                    "original": old_phrase,
                    "replacement": replacement,
                    "old_freq": old_freq,
                    "new_freq": new_freq,
                    "level": "sentence",
                })

        return result_text, replacements

    # ── Helper methods ───────────────────────────────────────────────

    def _get_synonym_map(self, lang: str) -> dict[str, list[str]]:
        """Get combined synonym map for the given language."""
        base = ZH_SYNONYM_MAP.copy() if lang == "zh" else EN_SYNONYM_MAP.copy()
        base.update(self._custom_synonyms)
        return base

    def _get_candidates(
        self, word: str, synonyms: dict[str, list[str]], lang: str
    ) -> list[dict[str, Any]]:
        """Get synonym candidates sorted by frequency (highest first)."""
        lookup = word.lower() if lang == "en" else word
        candidate_words = synonyms.get(lookup, [])
        if not candidate_words:
            return []

        candidates = []
        for cand in candidate_words:
            freq = self._get_word_freq(cand, lang)
            candidates.append({"word": cand, "freq": freq})

        candidates.sort(key=lambda c: c["freq"], reverse=True)
        return candidates

    def _get_word_freq(self, word: str, lang: str) -> float:
        """Get word frequency using the estimator's lookup."""
        return self._estimator._lookup_word_freq(word, lang)

    def _estimate_phrase_freq(self, phrase: str, lang: str) -> float:
        """Estimate the sfreq of a phrase."""
        result = self._estimator.estimate(phrase)
        return result.sfreq

    @staticmethod
    def _replace_word_in_text(text: str, old: str, new: str, lang: str) -> str:
        """Replace a word in text preserving surrounding context."""
        if lang == "zh":
            return text.replace(old, new, 1)
        else:
            pattern = re.compile(r'\b' + re.escape(old) + r'\b', re.IGNORECASE)
            match = pattern.search(text)
            if match:
                original_case = match.group()
                # Preserve original casing pattern
                if original_case[0].isupper() and new[0].islower():
                    new_cased = new[0].upper() + new[1:]
                elif original_case.isupper():
                    new_cased = new.upper()
                else:
                    new_cased = new
                return pattern.sub(new_cased, text, count=1)
            return text
