"""Sphinx configuration for the occhio documentation."""

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

from sphinx.directives.other import TocTree

# Add the project root to sys.path so autodoc can find the module
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# Add the docs directory so _ext modules are importable
sys.path.insert(0, str(Path(__file__).parent))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "occhio"
author = "Niclas Kupper, Kaushik Reddy, Oliver Sieweke, Kola Ayonrinde"
show_authors = True
release = importlib.metadata.version("occhio")
html_title = "A Library for Studying Superposition in Toy Models"
html_short_title = "occhio"
llms_txt_description = ""  # [2026-03-31 | OliverSieweke] TODO: write description

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx.ext.duration",
    "sphinx.ext.napoleon",
    "autodoc2",
    "sphinx.ext.mathjax",
    "notfound.extension",
    "sphinx.ext.linkcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx_copybutton",
    "sphinx_llm.txt",
    "sphinxcontrib.bibtex",
]

myst_enable_extensions = [
    "dollarmath",  # $, $$ Maths syntax
    "colon_fence",  # ::: directive syntax
    "fieldlist",  # Field lists in Markdown
    "deflist",  # Definition lists in Markdown
    "tasklist",  # GitHub-style - [ ] / - [x] task lists
    "smartquotes",  # Smart quotes
    "replacements",  # Smart replacements
    "linkify",  # Turn urls into links
]

# https://docs.readthedocs.com/platform/stable/intro/sphinx.html#set-the-canonical-url
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "/")

# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_static_path = ["_static"]

# https://docs.readthedocs.com/platform/stable/reference/robots.html
html_extra_path = ["robots.txt"]

# https://docs.readthedocs.com/platform/stable/guides/adding-custom-css.html
html_css_files = ["custom.css"]
html_js_files = ["custom.js"]

nitpicky = True
nitpick_ignore = [
    # numpy.typing.NDArray is a TypeAlias — numpy exposes it as py:data, not py:class
    ("py:class", "numpy.typing.NDArray"),
    # torch.nn.Parameter is not indexed as py:class in PyTorch's inventory
    ("py:class", "torch.nn.Parameter"),
    # No public intersphinx inventory exists for SAE Lens
    ("py:class", "sae_lens.TrainingSAE"),
    ("py:class", "sae_lens.synthetic.SyntheticDataEvalResult"),
    # autodoc2 indexes at definition location; re-export paths have no py:class entry
    ("py:class", "occhio.distributions.Distribution"),
    ("py:class", "occhio.ModelGrid"),
    ("py:class", "occhio.autoencoders.AutoEncoderBase"),
    # Builtin/NumPy annotations that resolve ambiguously or as non-class objects
    ("py:class", "type"),
    ("py:class", "numpy.uint8"),
    # Intentional placeholder in syntax_lookup.md example
    ("py:func", "new_function"),
]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

_is_readthedocs = bool(os.environ.get("READTHEDOCS", False))

# autodoc2
autodoc2_packages = [
    {
        "path": "../src/occhio",
    },
]
autodoc2_render_plugin = "myst"
autodoc2_output_dir = "content/reference/apidocs"
autodoc2_replace_annotations = [("type[", "typing.Type[")]
autodoc2_docstring_parser_regexes = [
    (r".*", "_ext.google_docstring_parser"),
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False

pygments_style = "lovelace"

todo_include_todos = not _is_readthedocs
todo_emit_warnings = False
suppress_warnings = [
    # autodoc2 parses each embedded docstring in isolation and reports generated
    # section levels without considering the surrounding generated module page.
    "myst.header",
]
if _is_readthedocs:
    exclude_patterns.append("dev")


intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
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

html_context = {"author": author}

html_theme_options = {
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 2,
    "includehidden": True,
    "titles_only": False,
    "logo_only": False,
    # Style options
    "style_nav_header_background": "#d18770",
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
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


# [2026-04-06 | OliverSieweke] TODO: check if needed
# toc_object_entries_show_parents
# maximum_signature_line_length¶
# sphinx_design extension

# -- sphinxcontrib-bibtex ----------------------------------------------------
bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "unsrt"


class LocalTocTree(TocTree):
    """A toctree that is omitted entirely from Read the Docs builds."""

    def run(self):
        """Create the toctree locally, before RTD can validate its entries."""
        if _is_readthedocs:
            return []
        return super().run()


def setup(app):
    """Register documentation-specific Sphinx directives."""
    app.add_directive("local-toctree", LocalTocTree)
    return {"parallel_read_safe": True}
