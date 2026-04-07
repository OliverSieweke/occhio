"""Custom autodoc2 docstring parser that converts Google-style docstrings to MyST.

Registered via ``autodoc2_docstring_parser_regexes`` in conf.py. autodoc2 does
not fire the ``autodoc-process-docstring`` event that ``sphinx.ext.napoleon``
hooks into, so Google-style ``Args:`` / ``Returns:`` sections are never
converted. This parser runs Napoleon's conversion before handing the result
to the MyST parser, enabling dollar-math and other MyST extensions in docstrings.
"""

from myst_parser.parsers.sphinx_ import MystParser
from sphinx.ext.napoleon import GoogleDocstring


class Parser(MystParser):
    """MyST parser that first converts Google-style docstrings via Napoleon."""

    supported = ("google-myst",)

    def parse(self, inputstring: str, document) -> None:
        config = getattr(document.settings, "env", None)
        config = config.config if config is not None else None
        converted = str(GoogleDocstring(inputstring.splitlines(), config))
        super().parse(converted, document)
