"""AdamOpt CLI - Command-line interface for text frequency optimization.

Usage:
    adamopt freq "your text here" --model qwen2.5-7b
    adamopt freq --file input.txt --model llama3.3-70b
    adamopt compare "text A" "text B" --model qwen2.5-7b
"""

from __future__ import annotations

import json
import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from adamopt import __version__
from adamopt.frequency import FrequencyEstimator

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="adamopt")
def main() -> None:
    """AdamOpt - Automatic text frequency optimization for LLMs.

    Based on Adam's Law (Textual Frequency Law), this tool quantifies
    and optimizes text frequency to maximize LLM performance.
    """


@main.command()
@click.argument("text", required=False)
@click.option(
    "--file", "-f",
    type=click.Path(exists=True),
    help="Read text from a file instead of command-line argument.",
)
@click.option(
    "--model", "-m",
    default="generic",
    show_default=True,
    help="Target LLM model (e.g., qwen2.5-7b, llama3.3-70b, deepseek-v3, generic).",
)
@click.option(
    "--language", "-l",
    type=click.Choice(["en", "zh", "auto"]),
    default="auto",
    show_default=True,
    help="Text language. 'auto' for automatic detection.",
)
@click.option(
    "--threshold",
    type=float,
    default=0.0001,
    show_default=True,
    help="Low-frequency threshold. Words below this are flagged.",
)
@click.option(
    "--freq-table",
    type=click.Path(exists=True),
    help="Path to a custom frequency table (.json or .tsv).",
)
@click.option(
    "--json-output", "-j",
    is_flag=True,
    help="Output results in JSON format.",
)
def freq(
    text: str | None,
    file: str | None,
    model: str,
    language: str,
    threshold: float,
    freq_table: str | None,
    json_output: bool,
) -> None:
    """Calculate text frequency for a given input.

    Computes word-level frequencies (wfreq) and sentence-level frequency (sfreq)
    using Adam's Law formula: sfreq = (∏ wfreq_k)^(1/K).

    Examples:

        adamopt freq "What is the capital of France?"

        adamopt freq "天空为什么是蓝色的？" --model qwen2.5-7b

        adamopt freq --file prompt.txt --model llama3.3-70b

        adamopt freq "your text" --json-output
    """
    # Resolve text input
    if text is None and file is None:
        # Try reading from stdin
        if not sys.stdin.isatty():
            text = sys.stdin.read().strip()
        else:
            console.print("[red]Error:[/red] Please provide text or use --file option.")
            raise SystemExit(1)
    elif file is not None:
        with open(file, encoding="utf-8") as f:
            text = f.read().strip()

    if not text:
        console.print("[red]Error:[/red] Empty text provided.")
        raise SystemExit(1)

    # Initialize estimator
    lang = None if language == "auto" else language
    estimator = FrequencyEstimator(
        model=model,
        language=lang,
        low_freq_threshold=threshold,
        custom_freq_table_path=freq_table,
    )

    # Compute frequency
    result = estimator.estimate(text)

    # Output
    if json_output:
        output = {
            "text": result.text,
            "model": result.model,
            "language": result.language,
            "sfreq": result.sfreq,
            "effective_word_count": result.effective_word_count,
            "freq_source": {
                "level": result.freq_source.level,
                "name": result.freq_source.name,
                "path": result.freq_source.path,
            } if result.freq_source else None,
            "word_frequencies": result.word_frequencies,
            "low_freq_words": [
                {"word": w, "frequency": f} for w, f in result.low_freq_words
            ],
        }
        click.echo(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        _render_rich_output(result)


@main.command()
@click.argument("text_a")
@click.argument("text_b")
@click.option(
    "--model", "-m",
    default="generic",
    show_default=True,
    help="Target LLM model.",
)
@click.option(
    "--language", "-l",
    type=click.Choice(["en", "zh", "auto"]),
    default="auto",
    show_default=True,
    help="Text language.",
)
def compare(text_a: str, text_b: str, model: str, language: str) -> None:
    """Compare frequency of two texts.

    Useful for evaluating which paraphrase has higher frequency
    and is more likely to yield better LLM performance.

    Examples:

        adamopt compare "What causes the sky to be blue?" "Why is the sky blue?"

        adamopt compare "罹患恶性肿瘤" "得了癌症" --model qwen2.5-7b
    """
    lang = None if language == "auto" else language
    estimator = FrequencyEstimator(model=model, language=lang)

    result = estimator.compare(text_a, text_b)

    # Render comparison
    table = Table(title="Text Frequency Comparison", show_lines=True)
    table.add_column("", style="bold")
    table.add_column("Text A", max_width=50)
    table.add_column("Text B", max_width=50)

    table.add_row("Text", text_a, text_b)
    table.add_row(
        "sfreq",
        f"{result['text_a']['sfreq']:.8f}",
        f"{result['text_b']['sfreq']:.8f}",
    )

    winner = "A ✓" if result["higher_freq"] == "a" else "B ✓"
    ratio = result["freq_ratio"]
    table.add_row("Higher Freq", winner, f"ratio: {ratio:.2f}x")

    console.print(table)

    winner_text = text_a if result["higher_freq"] == "a" else text_b
    console.print(
        f"\n[green]→ Recommended:[/green] Use [bold]{winner_text!r}[/bold] "
        f"for better LLM performance (higher sfreq)."
    )


@main.command()
def models() -> None:
    """List all supported LLM models."""
    from adamopt.freq_table import FreqTableManager

    mgr = FreqTableManager()

    table = Table(title="Supported Models")
    table.add_column("Model", style="cyan")
    table.add_column("Family", style="green")
    table.add_column("Languages")

    for name, info in mgr.KNOWN_MODELS.items():
        table.add_row(name, info["family"], ", ".join(info["lang"]))

    console.print(table)


@main.command()
@click.argument("corpus_path", type=click.Path(exists=True))
@click.option(
    "--model", "-m",
    required=True,
    help="Target LLM model name (e.g., qwen2.5-7b). The merged table will be cached under this name.",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="Output path for the merged frequency table (.json). If omitted, only saves to model cache.",
)
@click.option(
    "--language", "-l",
    type=click.Choice(["en", "zh", "auto"]),
    default="auto",
    show_default=True,
    help="Corpus language.",
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Do not save to model cache (only write to --output).",
)
def tfd(
    corpus_path: str,
    model: str,
    output: str | None,
    language: str,
    no_cache: bool,
) -> None:
    """Run Textual Frequency Distillation (TFD) to build a model-specific frequency table.

    CORPUS_PATH is a text file containing model-generated continuations (one per line).
    This should be text produced by the target LLM when asked to continue/complete
    sentences from your training data.

    TFD merges word frequencies from this corpus with baseline wordfreq frequencies,
    producing a frequency table that better reflects the target model's internal
    distribution. The result is cached so that subsequent `adamopt freq` calls
    using the same --model will automatically use the TFD-enhanced table.

    Steps to use TFD:

    \b
    1. Prepare your training data (e.g., 1000 sentences).
    2. Use your target LLM to generate 1-2 sentence continuations for each.
    3. Collect all continuations into a text file (one per line).
    4. Run: adamopt tfd continuations.txt --model qwen2.5-7b

    Examples:

        adamopt tfd model_continuations.txt --model qwen2.5-7b

        adamopt tfd generated.txt --model deepseek-v3 --output deepseek_freq.json

        adamopt tfd chinese_corpus.txt --model qwen2.5-7b --language zh
    """
    from adamopt.tfd import TFDDistiller

    lang = None if language == "auto" else language

    distiller = TFDDistiller(model=model, language=lang)

    console.print(f"\n[bold cyan]TFD - Textual Frequency Distillation[/bold cyan]")
    console.print(f"  Model: [bold]{model}[/bold]")
    console.print(f"  Corpus: {corpus_path}")
    console.print(f"  Processing...\n")

    try:
        result = distiller.distill(
            generated_corpus_path=corpus_path,
            output_path=output,
            language=lang,
            save_to_cache=not no_cache,
        )
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)

    # Report results
    table = Table(title="TFD Distillation Complete", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("Model", result["model"])
    table.add_row("Language", result["language"])
    table.add_row("Corpus Vocabulary", str(result["corpus_vocab_size"]))
    table.add_row("Merged Vocabulary", str(result["vocab_size"]))

    if result["output_path"]:
        table.add_row("Output File", result["output_path"])
    if result["cache_path"]:
        table.add_row("Cache File", result["cache_path"])

    console.print(table)

    console.print(
        f"\n[green]✓[/green] TFD complete. Use [bold]adamopt freq --model {model}[/bold] "
        f"to leverage the enhanced frequency table."
    )
    console.print()


@main.command()
@click.argument("text", required=False)
@click.option(
    "--file", "-f",
    type=click.Path(exists=True),
    help="Read text from a file.",
)
@click.option(
    "--model", "-m",
    default="generic",
    show_default=True,
    help="Target LLM model.",
)
@click.option(
    "--mode",
    type=click.Choice(["conservative", "balanced", "aggressive"]),
    default="balanced",
    show_default=True,
    help="Optimization mode: conservative (word-only), balanced (word+phrase), aggressive (full rewrite).",
)
@click.option(
    "--language", "-l",
    type=click.Choice(["en", "zh", "auto"]),
    default="auto",
    show_default=True,
    help="Text language.",
)
@click.option(
    "--json-output", "-j",
    is_flag=True,
    help="Output in JSON format.",
)
def optimize(
    text: str | None,
    file: str | None,
    model: str,
    mode: str,
    language: str,
    json_output: bool,
) -> None:
    """Optimize text for higher frequency using Adam's Law.

    Automatically identifies low-frequency bottleneck words and replaces them
    with higher-frequency synonyms/paraphrases, while preserving semantic fidelity.

    Protected content (entities, logic keywords, constraints, numbers) is
    automatically locked and never modified.

    Examples:

        adamopt optimize "What is the optical causation for the azure hue of the celestial firmament?"

        adamopt optimize "请你详尽阐述Transformer架构的核心设计理念" --mode conservative

        adamopt optimize --file prompt.txt --mode aggressive --model qwen2.5-7b
    """
    from adamopt.optimizer import TextOptimizer

    # Resolve text input
    if text is None and file is None:
        if not sys.stdin.isatty():
            text = sys.stdin.read().strip()
        else:
            console.print("[red]Error:[/red] Please provide text or use --file option.")
            raise SystemExit(1)
    elif file is not None:
        with open(file, encoding="utf-8") as f:
            text = f.read().strip()

    if not text:
        console.print("[red]Error:[/red] Empty text provided.")
        raise SystemExit(1)

    lang = None if language == "auto" else language
    optimizer = TextOptimizer(model=model, language=lang)

    result = optimizer.optimize(text, mode=mode)

    if json_output:
        output = {
            "original_text": result.original_text,
            "optimized_text": result.optimized_text,
            "original_sfreq": result.original_sfreq,
            "optimized_sfreq": result.optimized_sfreq,
            "sfreq_improvement": result.sfreq_improvement,
            "sfreq_ratio": result.sfreq_ratio,
            "mode": result.mode,
            "replacements": result.replacements,
            "locked_count": result.locked_count,
            "language": result.language,
            "model": result.model,
        }
        click.echo(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        _render_optimize_output(result)


def _render_optimize_output(result) -> None:
    """Render optimization result using rich formatting."""
    pct = (result.sfreq_ratio - 1) * 100 if result.sfreq_ratio > 0 else 0

    console.print(
        Panel(
            f"[bold]Model:[/bold] {result.model}  |  "
            f"[bold]Mode:[/bold] {result.mode}  |  "
            f"[bold]Language:[/bold] {result.language}  |  "
            f"[bold]Locked:[/bold] {result.locked_count} spans",
            title="[bold cyan]AdamOpt - Text Frequency Optimization[/bold cyan]",
            border_style="cyan",
        )
    )

    # Before/After comparison
    console.print(f"\n  [bold]Original:[/bold]  {result.original_text}")
    console.print(f"  [bold green]Optimized:[/bold green] {result.optimized_text}")

    # sfreq comparison
    console.print(f"\n  [bold]sfreq:[/bold] {result.original_sfreq:.8f} → {result.optimized_sfreq:.8f}  ", end="")
    if pct > 0:
        console.print(f"[green]+{pct:.1f}%[/green]")
    elif pct == 0:
        console.print("[yellow]no change[/yellow]")
    else:
        console.print(f"[red]{pct:.1f}%[/red]")

    # Replacements table
    if result.replacements:
        rep_table = Table(title=f"\nReplacements ({len(result.replacements)})", show_lines=False)
        rep_table.add_column("Original", style="red")
        rep_table.add_column("→", justify="center", style="dim")
        rep_table.add_column("Replacement", style="green")
        rep_table.add_column("Level")
        rep_table.add_column("Freq Change", justify="right")

        for r in result.replacements[:15]:
            freq_change = f"{r['old_freq']:.6f}→{r['new_freq']:.6f}"
            rep_table.add_row(r["original"], "→", r["replacement"], r["level"], freq_change)

        console.print(rep_table)
    else:
        console.print("\n  [yellow]No replacements made (text is already high-frequency).[/yellow]")

    console.print()


def _render_rich_output(result) -> None:
    """Render frequency result using rich formatting."""
    # Header
    header_parts = [
        f"[bold]Model:[/bold] {result.model}",
        f"[bold]Language:[/bold] {result.language}",
        f"[bold]Effective Words:[/bold] {result.effective_word_count}",
    ]
    console.print(
        Panel(
            "  |  ".join(header_parts),
            title="[bold cyan]AdamOpt - Text Frequency Analysis[/bold cyan]",
            border_style="cyan",
        )
    )

    # Frequency source info
    if result.freq_source:
        src = result.freq_source
        source_text = f"Level {src.level}: {src.name}"
        if src.path:
            source_text += f"\n    Path: {src.path}"
        if src.level == 1:
            color = "magenta"
        elif src.level == 2:
            color = "blue"
        else:
            color = "dim"
        console.print(f"\n  [bold]Frequency Source:[/bold] [{color}]{source_text}[/{color}]")

    # sfreq score with visual indicator
    sfreq = result.sfreq
    if sfreq >= 0.001:
        level = "[green]HIGH[/green]"
    elif sfreq >= 0.0001:
        level = "[yellow]MEDIUM[/yellow]"
    else:
        level = "[red]LOW[/red]"

    console.print(f"\n  [bold]Sentence Frequency (sfreq):[/bold] {sfreq:.8f}  {level}")

    # Word frequency table
    if result.word_frequencies:
        wf_table = Table(title="\nWord Frequencies", show_lines=False)
        wf_table.add_column("Word", style="white")
        wf_table.add_column("Frequency", justify="right")
        wf_table.add_column("Level", justify="center")

        sorted_words = sorted(
            result.word_frequencies.items(), key=lambda x: x[1], reverse=True
        )
        for word, freq in sorted_words[:20]:
            if freq >= 0.001:
                lvl = "[green]●[/green]"
            elif freq >= 0.0001:
                lvl = "[yellow]●[/yellow]"
            else:
                lvl = "[red]●[/red]"
            wf_table.add_row(word, f"{freq:.8f}", lvl)

        if len(sorted_words) > 20:
            wf_table.add_row("...", f"({len(sorted_words) - 20} more)", "")

        console.print(wf_table)

    # Low-frequency words
    if result.low_freq_words:
        console.print(
            f"\n  [bold red]⚠ Low-Frequency Bottlenecks "
            f"({len(result.low_freq_words)} words):[/bold red]"
        )
        for word, freq in result.low_freq_words[:10]:
            console.print(f"    [red]•[/red] {word!r} → {freq:.8f}")
        if len(result.low_freq_words) > 10:
            console.print(
                f"    ... and {len(result.low_freq_words) - 10} more"
            )
    else:
        console.print("\n  [green]✓ No low-frequency bottlenecks detected.[/green]")

    console.print()


if __name__ == "__main__":
    main()
