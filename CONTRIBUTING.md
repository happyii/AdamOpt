# Contributing to AdamOpt

Thank you for your interest in contributing to AdamOpt! This document provides guidelines and information about how to contribute.

## How to Contribute

### Reporting Bugs

1. Check existing [Issues](https://github.com/happyii/adamopt/issues) to avoid duplicates
2. Use the bug report template when creating a new issue
3. Include: Python version, OS, steps to reproduce, expected vs actual behavior

### Suggesting Features

1. Open an issue with the "feature request" label
2. Describe the use case and expected behavior
3. Reference any related research or papers if applicable

### Submitting Code

1. Fork the repository
2. Create a feature branch from `main`: `git checkout -b feature/your-feature`
3. Make your changes following the code style guide below
4. Add tests for new functionality
5. Run tests: `pytest tests/`
6. Run linter: `ruff check src/`
7. Commit with clear messages: `git commit -m "feat: add xxx"`
8. Push and open a Pull Request

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions
- Use type hints for all function signatures
- Write docstrings (Google style) for all public classes and functions
- Maximum line length: 100 characters
- Use `ruff` for linting

## Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

## Development Setup

```bash
git clone https://github.com/happyii/adamopt.git
cd adamopt
pip install -e ".[dev]"
pytest tests/
```

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
