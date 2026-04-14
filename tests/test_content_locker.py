"""Tests for the content locking module."""

from adamopt.content_locker import ContentLocker, LockResult


class TestContentLocker:
    """Test suite for ContentLocker."""

    def setup_method(self):
        self.locker = ContentLocker()

    def test_lock_numbers(self):
        """Test that numbers are locked."""
        result = self.locker.lock("The accuracy is 95% on this task")
        assert any(s.category == "objective" for s in result.locked_spans)
        assert "[LOCK_" in result.masked_text

    def test_lock_code_backticks(self):
        """Test that backtick code is locked."""
        result = self.locker.lock("Use `pip install adamopt` to install")
        assert any("`pip install adamopt`" in s.text for s in result.locked_spans)

    def test_lock_urls(self):
        """Test that URLs are locked."""
        result = self.locker.lock("Visit https://github.com/adamopt for details")
        assert any("https://github.com/adamopt" in s.text for s in result.locked_spans)

    def test_lock_logic_keywords_en(self):
        """Test that English logic keywords are locked."""
        result = self.locker.lock("You must not use any external tools")
        locked_texts = {s.text.lower() for s in result.locked_spans}
        assert "not" in locked_texts or "must" in locked_texts

    def test_lock_logic_keywords_zh(self):
        """Test that Chinese logic keywords are locked."""
        result = self.locker.lock("你必须不能使用外部工具")
        locked_texts = {s.text for s in result.locked_spans}
        assert "不能" in locked_texts or "必须" in locked_texts

    def test_lock_instruction_words_en(self):
        """Test that English instruction words are locked."""
        result = self.locker.lock("Translate the following text to French")
        locked_texts = {s.text.lower() for s in result.locked_spans}
        assert "translate" in locked_texts

    def test_lock_instruction_words_zh(self):
        """Test that Chinese instruction words are locked."""
        result = self.locker.lock("请翻译下面的文本为法语")
        locked_texts = {s.text for s in result.locked_spans}
        assert "翻译" in locked_texts

    def test_lock_custom_words(self):
        """Test that custom lock words are protected."""
        locker = ContentLocker(custom_lock_words={"Transformer", "GPT-4o"})
        result = locker.lock("Explain the Transformer architecture used in GPT-4o")
        locked_texts = {s.text for s in result.locked_spans}
        assert "Transformer" in locked_texts
        assert "GPT-4o" in locked_texts

    def test_restore_placeholders(self):
        """Test that locked content is correctly restored."""
        result = self.locker.lock("Translate x+y=z to code")
        restored = result.restore(result.masked_text)
        # Restored text should contain all original locked content
        for span in result.locked_spans:
            assert span.text in restored

    def test_lock_preserves_unlocked_text(self):
        """Test that non-locked text remains in masked output."""
        result = self.locker.lock("The sky is blue today")
        # "sky" and "blue" should not be locked (they are regular words)
        assert "sky" in result.masked_text or "blue" in result.masked_text

    def test_lock_format_constraint(self):
        """Test that format constraints are locked."""
        result = self.locker.lock("Answer in JSON format please")
        locked_texts = " ".join(s.text for s in result.locked_spans)
        assert "JSON format" in locked_texts or "json" in locked_texts.lower()

    def test_empty_text(self):
        """Test locking empty text."""
        result = self.locker.lock("")
        assert result.masked_text == ""
        assert len(result.locked_spans) == 0

    def test_no_lockable_content(self):
        """Test text with no content to lock."""
        result = self.locker.lock("sky blue red green")
        # Simple common words without numbers/logic/instructions should have minimal locking
        assert result.masked_text  # Should still return something
