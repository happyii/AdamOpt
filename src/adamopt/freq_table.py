"""Model-specific frequency table management.

Handles loading, caching, and merging of word frequency tables for different LLMs.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


# Default cache directory
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "adamopt" / "freq_tables"


class FreqTableManager:
    """Manage model-specific word frequency tables.

    Supports loading pre-built tables, importing custom tables,
    and local caching to avoid redundant computation.
    """

    # Registry of known model families and their frequency table identifiers.
    # In v0.1 we rely on wordfreq's generic frequencies for all models;
    # model-specific tables can be added in future versions.
    KNOWN_MODELS: dict[str, dict[str, Any]] = {
        # Qwen series
        "qwen2.5-0.5b": {"family": "qwen2.5", "lang": ["en", "zh"]},
        "qwen2.5-1.5b": {"family": "qwen2.5", "lang": ["en", "zh"]},
        "qwen2.5-3b": {"family": "qwen2.5", "lang": ["en", "zh"]},
        "qwen2.5-7b": {"family": "qwen2.5", "lang": ["en", "zh"]},
        "qwen2.5-14b": {"family": "qwen2.5", "lang": ["en", "zh"]},
        "qwen2.5-32b": {"family": "qwen2.5", "lang": ["en", "zh"]},
        "qwen2.5-72b": {"family": "qwen2.5", "lang": ["en", "zh"]},
        # Llama series
        "llama3.3-8b": {"family": "llama3.3", "lang": ["en"]},
        "llama3.3-70b": {"family": "llama3.3", "lang": ["en"]},
        # DeepSeek
        "deepseek-v3": {"family": "deepseek", "lang": ["en", "zh"]},
        # Generic (uses wordfreq directly)
        "generic": {"family": "generic", "lang": ["en", "zh"]},
    }

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        """Initialize the frequency table manager.

        Args:
            cache_dir: Directory to store cached frequency tables.
                Defaults to ~/.cache/adamopt/freq_tables.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_tables: dict[str, dict[str, float]] = {}

    def get_supported_models(self) -> list[str]:
        """Return list of supported model names."""
        return list(self.KNOWN_MODELS.keys())

    def resolve_model(self, model_name: str) -> str:
        """Resolve a model name to a canonical identifier.

        Performs case-insensitive fuzzy matching. Falls back to 'generic'.

        Args:
            model_name: User-provided model name.

        Returns:
            Canonical model identifier.
        """
        normalized = model_name.lower().strip().replace(" ", "-").replace("_", "-")

        # Exact match
        if normalized in self.KNOWN_MODELS:
            return normalized

        # Partial match: check if the input is a substring of any known model
        for key in self.KNOWN_MODELS:
            if normalized in key or key in normalized:
                return key

        # Family match
        for key, info in self.KNOWN_MODELS.items():
            if normalized in info["family"] or info["family"] in normalized:
                return key

        return "generic"

    def load_custom_table(
        self,
        filepath: str | Path,
        model_name: str = "custom",
        *,
        cache: bool = True,
    ) -> dict[str, float]:
        """Load a custom word frequency table from file.

        Supports formats:
        - TSV: word<TAB>frequency (one per line)
        - JSON: {"word": frequency, ...}

        Args:
            filepath: Path to the frequency table file.
            model_name: Identifier for this custom table.
            cache: Whether to cache the loaded table.

        Returns:
            Dictionary mapping words to their frequency values (0-1).
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Frequency table not found: {filepath}")

        suffix = filepath.suffix.lower()
        freq_dict: dict[str, float] = {}

        if suffix == ".json":
            with open(filepath, encoding="utf-8") as f:
                freq_dict = json.load(f)
        elif suffix in (".tsv", ".txt"):
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        word, freq = parts[0], float(parts[1])
                        freq_dict[word] = freq
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. Use .json, .tsv, or .txt"
            )

        # Normalize to 0-1 range
        freq_dict = _normalize_freq_dict(freq_dict)

        if cache:
            cache_key = f"custom_{model_name}"
            self._loaded_tables[cache_key] = freq_dict
            self._save_to_cache(cache_key, freq_dict)

        return freq_dict

    def get_table(self, model_name: str) -> dict[str, float] | None:
        """Get a cached frequency table for a model.

        Args:
            model_name: Model identifier.

        Returns:
            Frequency dictionary if available, None otherwise.
        """
        resolved = self.resolve_model(model_name)

        # Check in-memory cache
        if resolved in self._loaded_tables:
            return self._loaded_tables[resolved]

        # Check disk cache
        cached = self._load_from_cache(resolved)
        if cached is not None:
            self._loaded_tables[resolved] = cached
            return cached

        return None

    def get_table_path(self, model_name: str) -> Path | None:
        """Get the disk cache path for a model's frequency table.

        Args:
            model_name: Model identifier.

        Returns:
            Path to the cached table file, or None if not cached on disk.
        """
        resolved = self.resolve_model(model_name)
        path = self._cache_path(resolved)
        return path if path.exists() else None

    def save_table(
        self, model_name: str, freq_dict: dict[str, float]
    ) -> None:
        """Save a frequency table for a model.

        Args:
            model_name: Model identifier.
            freq_dict: Word frequency dictionary.
        """
        resolved = self.resolve_model(model_name)
        normalized = _normalize_freq_dict(freq_dict)
        self._loaded_tables[resolved] = normalized
        self._save_to_cache(resolved, normalized)

    def _cache_path(self, key: str) -> Path:
        """Get the cache file path for a given key."""
        safe_name = hashlib.md5(key.encode()).hexdigest()[:12]
        return self.cache_dir / f"{key}_{safe_name}.json"

    def _save_to_cache(self, key: str, freq_dict: dict[str, float]) -> None:
        """Save frequency table to disk cache."""
        path = self._cache_path(key)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(freq_dict, f, ensure_ascii=False)

    def _load_from_cache(self, key: str) -> dict[str, float] | None:
        """Load frequency table from disk cache."""
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None


def _normalize_freq_dict(freq_dict: dict[str, float]) -> dict[str, float]:
    """Normalize frequency values to 0-1 range.

    If max value > 1, assumes raw counts and normalizes by total.
    Otherwise assumes already normalized.
    """
    if not freq_dict:
        return freq_dict

    max_val = max(freq_dict.values())
    if max_val <= 1.0:
        return freq_dict

    total = sum(freq_dict.values())
    if total == 0:
        return freq_dict

    return {word: freq / total for word, freq in freq_dict.items()}
