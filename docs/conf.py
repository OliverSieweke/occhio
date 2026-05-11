# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from pathlib import Path

# Add the project root to sys.path so autodoc can find the module
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "occhio"
author = "Niclas Kupper, Kaushik Reddy, Oliver Sieweke, Kola Ayonrinde"
release = "0.2.0"
html_title = "occhio Documentation"
html_short_title = "occhio"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "notfound.extension",
]

myst_enable_extensions = [
    "dollarmath",
]

# Turn on sphinx.ext.autosummary
autosummary_generate = True

# autosummary_ignore_module_all = False
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

nitpicky = True

# Suppress warnings for external types that Sphinx cannot resolve
# (torch, numpy, plotly, sae_lens, etc.)
nitpick_ignore_regex = [
    (r"py:.*", r"torch\..*"),
    (r"py:.*", r"Tensor"),
    (r"py:.*", r"nn\..*"),
    (r"py:.*", r"numpy\..*"),
    (r"py:.*", r"np\..*"),
    (r"py:.*", r"NDArray.*"),
    (r"py:.*", r"plotly\..*"),
    (r"py:.*", r"go\..*"),
    (r"py:.*", r"pathlib\..*"),
    (r"py:.*", r"Path"),
    (r"py:.*", r"sae_lens\..*"),
    (r"py:.*", r"SyntheticDataEvalResult"),
    (r"py:.*", r"pd\..*"),
    (r"py:.*", r"pandas\..*"),
    (r"py:.*", r"Optimizer"),
    (r"py:.*", r"FigureProxy"),
    (r"py:.*", r"InteractiveFigure"),
    (r"py:.*", r"TrainingSAE"),
    (r"py:.*", r"SAETrainer"),
    (r"py:.*", r"HfApi"),
    (r"py:.*", r"abc\.ABC"),
    (r"py:.*", r"enum\.Enum"),
    (r"py:.*", r"occhio\.visualization\.plots\.feature_representation\._.*"),
    (r"py:.*", r"occhio\.visualization\.plots\.compute\._.*"),
]

# https://docs.readthedocs.com/platform/stable/intro/sphinx.html#set-the-canonical-url
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "https://occhio.dev/")

# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_static_path = ["_static"]

# https://docs.readthedocs.com/platform/stable/reference/robots.html
html_extra_path = ["robots.txt"]

# https://docs.readthedocs.com/platform/stable/guides/adding-custom-css.html
html_css_files = ["custom.css"]

# -- Options for UX ----------------------------------------------------------
html_theme = "furo"
html_favicon = "_static/occhio-dark.svg"
html_show_sphinx = False
html_show_copyright = False
html_last_updated_fmt = "%b %d, %Y"

# Furo theme customization — occhio brand: dark grey #171513 + crème #f4ede2
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#171513",
        "color-brand-content": "#2c2825",
        "color-sidebar-background": "#f4ede2",
        "color-sidebar-brand-text": "#171513",
        "color-sidebar-caption-text": "#6b6560",
        "color-sidebar-link-text": "#2c2825",
        "color-sidebar-link-text--top-level": "#171513",
        "sidebar-caption-font-size": "0.85rem",
    },
    "dark_css_variables": {
        "color-brand-primary": "#f4ede2",
        "color-brand-content": "#e8dfd3",
        "color-background-primary": "#171513",
        "color-background-secondary": "#1e1c19",
        "color-sidebar-background": "#1e1c19",
        "color-sidebar-brand-text": "#f4ede2",
        "color-sidebar-caption-text": "#a09890",
        "color-sidebar-link-text": "#d4cdc2",
        "color-sidebar-link-text--top-level": "#f4ede2",
        "color-foreground-primary": "#f4ede2",
        "color-foreground-secondary": "#d4cdc2",
    },
    "light_logo": "occhio-dark.svg",
    "dark_logo": "occhio-cream.svg",
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/OliverSieweke/occhio/",
    "source_branch": "main",
    "source_directory": "docs/",
}
