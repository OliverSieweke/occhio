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
make -C docs html

# Live reload server for development
make -C docs livehtml

# Clean build artifacts
make -C docs clean
```

## Output Location

Built documentation is located in `docs/_build/html/`.

## Configuration

Sphinx configuration is in `docs/conf.py`:

- **Theme**: sphinx-rtd-theme
- **Extensions**: sphinx-autobuild, myst-parser, sphinx-llm, sphinx-notfound-page
- **Auto-summary**: Enabled for API reference generation

## When to Build Docs

- After adding/modifying docstrings
- Before releasing new versions
- When updating API documentation
- For local preview during documentation writing
