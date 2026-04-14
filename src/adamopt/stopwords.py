"""Built-in stopwords for Chinese and English.

Provides default stopword lists and supports user-defined custom stopwords.
"""

from __future__ import annotations

# High-coverage English stopwords (function words that carry no semantic meaning)
ENGLISH_STOPWORDS: set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "about", "above", "after",
    "again", "against", "all", "am", "any", "aren't", "because", "before",
    "below", "between", "both", "cannot", "couldn't", "didn't", "doesn't",
    "don't", "down", "during", "each", "few", "further", "hadn't", "hasn't",
    "haven't", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "into", "isn't", "it", "its", "itself", "let's", "me", "more",
    "most", "mustn't", "my", "myself", "no", "nor", "not", "off", "once",
    "only", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "so", "some", "such", "than", "that", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "too", "under", "until", "up", "very", "wasn't", "we",
    "weren't", "what", "when", "where", "which", "while", "who", "whom", "why",
    "with", "won't", "wouldn't", "you", "your", "yours", "yourself",
    "yourselves", "just", "also", "still", "even", "already", "yet",
}

# High-coverage Chinese stopwords (particles, conjunctions, auxiliaries)
CHINESE_STOPWORDS: set[str] = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "吗", "吧", "被",
    "比", "别", "别的", "别说", "并", "并且", "不但", "不过", "不仅",
    "不论", "不如", "不是", "才", "从", "从而", "但", "但是", "当", "当然",
    "得", "等", "地", "第", "对", "对于", "多", "而", "而且", "尔", "凡是",
    "个", "各", "给", "跟", "更", "还", "还是", "还有", "何", "何况",
    "和", "很", "后", "即", "即使", "既", "既然", "将", "将要", "就是",
    "据", "可", "可是", "可以", "来", "另", "另外", "吗", "嘛", "么",
    "每", "们", "那", "那个", "那么", "那些", "呢", "能", "你们", "您",
    "哦", "其", "其实", "其中", "起", "且", "却", "然而", "然后", "让",
    "如", "如果", "如何", "若", "什么", "甚至", "虽", "虽然", "所", "所以",
    "他们", "她们", "它们", "啊", "呀", "哇", "嗯", "哎", "唉",
}


class StopwordManager:
    """Manage stopwords for text frequency calculation.

    Supports built-in Chinese/English stopwords and user-defined custom lists.
    """

    def __init__(
        self,
        languages: list[str] | None = None,
        custom_stopwords: set[str] | None = None,
    ) -> None:
        """Initialize stopword manager.

        Args:
            languages: List of language codes to load built-in stopwords for.
                Supported: "en", "zh". Defaults to ["en", "zh"].
            custom_stopwords: Additional user-defined stopwords to include.
        """
        if languages is None:
            languages = ["en", "zh"]

        self._stopwords: set[str] = set()

        builtin_map = {
            "en": ENGLISH_STOPWORDS,
            "zh": CHINESE_STOPWORDS,
        }
        for lang in languages:
            if lang in builtin_map:
                self._stopwords |= builtin_map[lang]

        if custom_stopwords:
            self._stopwords |= custom_stopwords

    @property
    def stopwords(self) -> set[str]:
        """Return the current stopword set."""
        return self._stopwords

    def is_stopword(self, word: str) -> bool:
        """Check if a word is a stopword."""
        return word.lower() in self._stopwords or word in self._stopwords

    def add(self, words: set[str]) -> None:
        """Add custom stopwords."""
        self._stopwords |= words

    def remove(self, words: set[str]) -> None:
        """Remove stopwords from the set."""
        self._stopwords -= words
