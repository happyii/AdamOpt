"""Tests for stopwords module."""

from adamopt.stopwords import StopwordManager


class TestStopwordManager:
    """Test stopword management."""

    def test_default_includes_english(self):
        mgr = StopwordManager()
        assert mgr.is_stopword("the")
        assert mgr.is_stopword("is")
        assert mgr.is_stopword("and")

    def test_default_includes_chinese(self):
        mgr = StopwordManager()
        assert mgr.is_stopword("的")
        assert mgr.is_stopword("了")
        assert mgr.is_stopword("是")

    def test_content_words_not_stopwords(self):
        mgr = StopwordManager()
        assert not mgr.is_stopword("sky")
        assert not mgr.is_stopword("blue")
        assert not mgr.is_stopword("天空")

    def test_english_only(self):
        mgr = StopwordManager(languages=["en"])
        assert mgr.is_stopword("the")
        assert not mgr.is_stopword("的")

    def test_chinese_only(self):
        mgr = StopwordManager(languages=["zh"])
        assert mgr.is_stopword("的")
        assert not mgr.is_stopword("the")

    def test_custom_stopwords(self):
        mgr = StopwordManager(custom_stopwords={"foo", "bar"})
        assert mgr.is_stopword("foo")
        assert mgr.is_stopword("bar")

    def test_add_stopwords(self):
        mgr = StopwordManager()
        assert not mgr.is_stopword("xyz")
        mgr.add({"xyz"})
        assert mgr.is_stopword("xyz")

    def test_remove_stopwords(self):
        mgr = StopwordManager()
        assert mgr.is_stopword("the")
        mgr.remove({"the"})
        assert not mgr.is_stopword("the")
