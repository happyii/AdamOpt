"""Tests for CLI commands."""

from click.testing import CliRunner

from adamopt.cli import main


class TestCLI:
    """Test CLI commands."""

    def setup_method(self):
        self.runner = CliRunner()

    def test_version(self):
        result = self.runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "adamopt" in result.output

    def test_freq_basic(self):
        result = self.runner.invoke(main, ["freq", "Why is the sky blue?"])
        assert result.exit_code == 0
        assert "sfreq" in result.output.lower() or "Frequency" in result.output

    def test_freq_with_model(self):
        result = self.runner.invoke(
            main, ["freq", "Hello world", "--model", "qwen2.5-7b"]
        )
        assert result.exit_code == 0

    def test_freq_json_output(self):
        result = self.runner.invoke(
            main, ["freq", "Hello world", "--json-output"]
        )
        assert result.exit_code == 0
        import json
        data = json.loads(result.output)
        assert "sfreq" in data
        assert "word_frequencies" in data

    def test_freq_chinese(self):
        result = self.runner.invoke(main, ["freq", "天空为什么是蓝色的？"])
        assert result.exit_code == 0

    def test_compare(self):
        result = self.runner.invoke(
            main,
            ["compare", "Why is the sky blue?", "What causes the azure hue of the firmament?"],
        )
        assert result.exit_code == 0
        assert "Recommended" in result.output or "higher" in result.output.lower()

    def test_models(self):
        result = self.runner.invoke(main, ["models"])
        assert result.exit_code == 0
        assert "qwen" in result.output.lower()

    def test_freq_no_input(self):
        result = self.runner.invoke(main, ["freq"], input="")
        # Should show error
        assert result.exit_code != 0 or "Error" in result.output or "error" in result.output.lower()
