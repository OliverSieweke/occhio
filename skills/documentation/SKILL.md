---
name: documentation
description: Sphinx documentation building and management. Use when building docs, viewing documentation locally with live reload, or managing Sphinx configuration.
---

# Documentation Skill

## Overview

This project uses **Sphinx** for API documentation generation with MyST parser support for markdown.

## Commands

```bash
# Install documentation dependencies
uv sync --extra docs

# Build HTML documentation
make docs html

# Live reload server for development
make docs livehtml

# Clean build artifacts
make docs clean
```

## Output Location

Built documentation is located in `docs/_build/html/`.

## Configuration

Sphinx configuration is in `docs/conf.py`:

- **Theme**: sphinx-rtd-theme
- **Extensions**: sphinx-autobuild, myst-parser, sphinx-llm, sphinx-notfound-page
- **Auto-summary**: Enabled for API reference generation

## Docstring Convention

Use **Google style** docstrings throughout. References:

- [Google Python Style Guide — Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- [Napoleon docs](https://sphinxcontrib-napoleon.readthedocs.io/)

Example:

```python
def sample(self, batch_size: int, generator: Generator | None = None) -> Tensor:
    """Sample a batch of features.

    Args:
        batch_size: Number of samples to draw.
        generator: Optional RNG for reproducibility.

    Returns:
        Tensor of shape ``(batch_size, n_features)``.
    """
```

- Types are **not** repeated in the docstring body — rely on type hints instead.

## When to Build Docs

- After adding/modifying docstrings
- Before releasing new versions
- When updating API documentation
- For local preview during documentation writing
