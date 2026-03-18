---
name: formatting-and-linting
description: Ruff formatting and linting setup. Use when formatting Python code, checking linting errors, or clearing notebook outputs.
---

# Code Formatting and Linting

## Overview

This project uses **Ruff** for code formatting and linting.

## Commands

```bash
# Format all files
uv run ruff format

# Check linting
uv run ruff check

# Clear notebook outputs before commit
uv run jupyter nbconvert --clear-output --inplace
```

## Configuration

Ruff is configured in `ruff.toml`:

- **Docstring code formatting**: Enabled (code blocks in docstrings are formatted)

## When to Format

- Before committing changes
- After editing Python files
- `nbstripout` is available for stripping notebook outputs
