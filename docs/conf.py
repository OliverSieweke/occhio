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
html_title = "A Library for Studying Superposition in Toy Models"
html_short_title = "occhio"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
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

# https://docs.readthedocs.com/platform/stable/intro/sphinx.html#set-the-canonical-url
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "/")

# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_static_path = ["_static"]

html_extra_path = ["robots.txt"]

# -- Options for UX ----------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_logo = "_static/logo.png"
html_favicon = "_static/logo.png"
html_show_sphinx = False
html_show_copyright = False
html_last_updated_fmt = "%b %d, %Y"

html_theme_options = {
    # Toc options
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 2,
    "includehidden": True,
    "titles_only": False,
    "logo_only": False,
    # "display_version": True,
    # Style options
    "style_nav_header_background": "#d18770",  # Change this color
}
