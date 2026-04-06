# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import functools
import importlib
import importlib.metadata
import inspect
import os
import sys
from pathlib import Path

# Add the project root to sys.path so autodoc can find the module
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "occhio"
author = "Niclas Kupper, Kaushik Reddy, Oliver Sieweke, Kola Ayonrinde"
release = importlib.metadata.version("occhio")
html_title = "A Library for Studying Superposition in Toy Models"
html_short_title = "occhio"
llms_txt_description = ""  # [2026-03-31 | OliverSieweke] TODO: write description

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "notfound.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
    "sphinx_llm.txt",
]

myst_enable_extensions = [
    "dollarmath",  # $, $$ Maths syntax
    "colon_fence",  # ::: directive syntax
    "fieldlist",  # Field lists in Markdown
    "deflist",  # Definition lists in Markdown
    "tasklist",  # GitHub-style - [ ] / - [x] task lists
]

# https://docs.readthedocs.com/platform/stable/intro/sphinx.html#set-the-canonical-url
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "/")

# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_static_path = ["_static"]

# https://docs.readthedocs.com/platform/stable/reference/robots.html
html_extra_path = ["robots.txt"]

# https://docs.readthedocs.com/platform/stable/guides/adding-custom-css.html
html_css_files = ["custom.css"]

# Turn on sphinx.ext.autosummary
autosummary_generate = True
autosummary_imported_members = False
nitpicky = True
# autosummary_ignore_module_all = False
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

_is_readthedocs = bool(os.environ.get("READTHEDOCS", False))

# Check these...
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",  # Preserve source order instead of alphabetical
}
autodoc_typehints = "description"  # Show type hints in parameter descriptions
autodoc_typehints_description_target = "documented"  # Only for documented params
autodoc_class_signature = "separated"  # Put class signature on its own line

# Docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
todo_include_todos = not _is_readthedocs
todo_emit_warnings = not _is_readthedocs
if _is_readthedocs:
    exclude_patterns.append("dev")
    suppress_warnings = ["toc.excluded"]


intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "plotly": ("https://plotly.com/python-api-reference/", None),
}

# -- Options for UX ----------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"
html_show_sphinx = False
html_show_copyright = False
html_show_sourcelink = False
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


# -- linkcode configuration --------------------------------------------------
def linkcode_resolve(domain, info):
    """Resolve source code links to GitHub.

    Args:
        domain: The language domain (e.g., 'py' for Python)
        info: Dictionary with 'module' and 'fullname' keys

    Returns:
        URL string to the source code on GitHub, or None if not found
    """
    if domain != "py" or not info["module"]:
        return None

    module = importlib.import_module(info["module"])

    repo_root = Path(__file__).parent.parent

    try:
        obj = functools.reduce(getattr, info["fullname"].split("."), module)
    except AttributeError:  # Instance attribute — link to the containing class instead
        parts = info["fullname"].split(".")
        if len(parts) < 2:
            return None
        try:
            obj = functools.reduce(getattr, parts[:-1], module)
        except AttributeError:
            return None

    try:
        inspectable = obj.fget if isinstance(obj, property) else obj
        file_path = (
            Path(inspect.getfile(inspectable))
            .resolve()
            .relative_to(repo_root.resolve())
        )
        source, line = inspect.getsourcelines(inspectable)
    except Exception:
        return None

    return f"https://github.com/OliverSieweke/occhio/blob/main/{file_path}#L{line}-L{line + len(source) - 1}"
