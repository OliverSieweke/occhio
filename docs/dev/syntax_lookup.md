# Syntax Lookup

Quick reference for every syntax available in this project's docs.

---

## MyST Markdown — Core

### Headings

````md
# Heading 1

## Heading 2

### Heading 3

#### Heading 4
````

### Inline formatting

````md
**bold**, *italic*, ***bold italic***, ~~strikethrough~~, `inline code`
````

**bold**, *italic*, ***bold italic***, ~~strikethrough~~, `inline code`

### Links

````md
[External link](https://example.com)
[Link with title](https://example.com "Title text")
<https://example.com>
````

[External link](https://example.com)

### Cross-references (MyST)

````md
{doc}`index`
{doc}`Custom text <index>`
{ref}`label-name`
{func}`occhio.ToyModel.fit`
{class}`occhio.ToyModel`
{meth}`occhio.ToyModel.fit`
{mod}`occhio`
{any}`occhio.ToyModel`
````

### Images

````md
![Alt text](/_static/logo.svg)

```{image} /_static/logo.svg
:alt: Logo
:width: 200px
:align: center
```

```{figure} /_static/logo.svg
:alt: Logo
:width: 200px
:align: center

Caption text here.
```
````

```{image} /_static/logo.svg
:alt: Logo
:width: 100px
:align: center
```

### Block quotes

````md
> This is a block quote.
>
> It can span multiple lines.
````

> This is a block quote.
>
> It can span multiple lines.

### Horizontal rule

````md
---
````

---

### Unordered lists

````md
- Item 1
- Item 2
	- Nested item
		- Deeper nested
````

- Item 1
- Item 2
	- Nested item
		- Deeper nested

### Ordered lists

````md
1. First
2. Second
3. Third
````

1. First
2. Second
3. Third

### Code blocks

````md
```python
def hello():
    return "world"
```
````

```python
def hello():
    return "world"
```

### Code block with caption and line numbers

````md
```{code-block} python
:caption: my_script.py
:linenos:
:emphasize-lines: 2

def hello():
    return "world"  # this line is highlighted
```
````

```{code-block} python
:caption: my_script.py
:linenos:
:emphasize-lines: 2

def hello():
    return "world"  # this line is highlighted
```

### Tables

````md
| Feature   | Status |
|-----------|--------|
| Training  | Done   |
| Plotting  | WIP    |
````

| Feature  | Status |
|----------|--------|
| Training | Done   |
| Plotting | WIP    |

### Footnotes

````md
This has a footnote[^1] and another[^note].

[^1]: First footnote.
[^note]: Named footnote.
````

This has a footnote[^1] and another[^note].

[^1]: First footnote.
[^note]: Named footnote.

### Comments (not rendered)

````md
% This is a comment and will not appear in the output.

[//]: # (This is also a comment.)
````

% This is a comment and will not appear in the output.

### Target labels (anchors)

````md
(my-label)=

## Section with a label

Link to it: {ref}`my-label`
````

### Escape characters

````md
\*not italic\*, \`not code\`
````

\*not italic\*, \`not code\`

---

## MyST Directives & Roles — General Syntax

### Backtick-fence directive

````md
```{directivename} argument
:option: value

Content goes here.
```
````

### Colon-fence directive (`colon_fence` extension)

````md
:::{directivename} argument
:option: value

Content goes here.
:::
````

### Nested directives

Use more backticks (or colons) for nesting:

`````md
````{note}
```{warning}
Nested inside a note.
```
````
`````

````{note}
```{warning}
Nested inside a note.
```
````

### Roles

````md
{rolename}`content`
{rolename}`display text <target>`
````

---

## MyST Extension — `dollarmath`

### Inline math

````md
The loss is $L = \frac{1}{n}\sum_i \|x_i - \hat{x}_i\|^2$.
````

The loss is $L = \frac{1}{n}\sum_i \|x_i - \hat{x}_i\|^2$.

### Display math

````md
$$
W^T W = I + \epsilon
$$
````

$$
W^T W = I + \epsilon
$$

### Math directive (with label for cross-referencing)

````md
```{math}
:label: eq-loss
L = \|x - D(E(x))\|^2
```

See equation {eq}`eq-loss`.
````

```{math}
:label: eq-loss
L = \|x - D(E(x))\|^2
```

See equation {eq}`eq-loss`.

---

## MyST Extension — `fieldlist`

````md
:Version: 0.2.0
:Status: Experimental
````

:Version: 0.2.0
:Status: Experimental

---

## MyST Extension — `deflist`

````md
Superposition
: When a model represents more features than it has dimensions.

Sparsity
: The fraction of inputs for which a given feature is inactive.
````

Superposition
: When a model represents more features than it has dimensions.

Sparsity
: The fraction of inputs for which a given feature is inactive.

---

## MyST Extension — `tasklist`

````md
- [x] Implement distributions
- [x] Add autodoc
- [ ] Write tutorial
- [ ] Add benchmarks
````

- [x] Implement distributions
- [x] Add autodoc
- [ ] Write tutorial
- [ ] Add benchmarks

---

## MyST Extension — `smartquotes`

Automatically converts straight quotes to typographic quotes:

````md
"Hello" becomes "Hello"
'world' becomes 'world'
````

"Hello" and 'world' — rendered with smart quotes.

---

## MyST Extension — `replacements`

Automatic text replacements:

````md
(c) (C) (r) (R) (tm) (TM) (p) (P) +- ...
````

(c) (C) (r) (R) (tm) (TM) (p) (P) +- ...

---

## MyST Extension — `linkify`

URLs are automatically turned into links:

````md
https://github.com/OliverSieweke/occhio
````

https://github.com/OliverSieweke/occhio

---

## Sphinx — Admonitions

````md
```{note}
A note.
```

```{warning}
A warning.
```

```{tip}
A tip.
```

```{important}
Important info.
```

```{caution}
Be careful.
```

```{danger}
Dangerous!
```

```{error}
An error.
```

```{hint}
A hint.
```

```{seealso}
Related info.
```

```{admonition} Custom Title
:class: tip
A custom-titled admonition with tip styling.
```
````

```{note}
A note.
```

```{warning}
A warning.
```

```{tip}
A tip.
```

```{important}
Important info.
```

```{caution}
Be careful.
```

```{danger}
Dangerous!
```

```{error}
An error.
```

```{hint}
A hint.
```

```{seealso}
Related info.
```

```{admonition} Custom Title
:class: tip
A custom-titled admonition with tip styling.
```

---

## Sphinx — Toctree

````md
```{toctree}
:maxdepth: 2
:caption: Section Name

page1
page2
```
````

### eval-rst toctree (alternative)

````md
```{eval-rst}
.. toctree::
   :maxdepth: 2
   :caption: Section Name

   page1.md
   page2.md
```
````

---

## Sphinx — Rubric (informal heading, not in toctree)

````md
```{rubric} Not a real heading
```
````

```{rubric} Not a real heading
```

---

## Sphinx — `versionadded` / `versionchanged` / `deprecated`

````md
```{versionadded} 0.3.0
New feature description.
```

```{versionchanged} 0.2.0
Behavior change description.
```

```{deprecated} 0.4.0
Use {func}`new_function` instead.
```
````

```{versionadded} 0.3.0
New feature description.
```

```{versionchanged} 0.2.0
Behavior change description.
```

```{deprecated} 0.4.0
Use {func}`new_function` instead.
```

---

## Sphinx — Glossary & Terms

````md
```{glossary}
Superposition
   When a model represents more features than it has dimensions.

Interference
   Cross-talk between feature representations sharing the same dimensions.
```

See {term}`Superposition`.
````

```{glossary}
Superposition
   When a model represents more features than it has dimensions.

Interference
   Cross-talk between feature representations sharing the same dimensions.
```

See {term}`Superposition`.

---

## Sphinx — Only / Conditional content

````md
```{only} html
This only appears in HTML output.
```
````

---

## `sphinx.ext.mathjax`

Enabled via `dollarmath` above. Also supports the `math` role:

````md
The matrix {math}`W \in \mathbb{R}^{m \times n}` defines the encoding.
````

The matrix {math}`W \in \mathbb{R}^{m \times n}` defines the encoding.

---

## `sphinx.ext.todo`

````md
```{todo}
Implement this section.
```

```{todolist}
```
````

```{todo}
Implement this section.
```

Todos are visible locally but hidden on ReadTheDocs.

---

## `sphinx.ext.intersphinx`

Cross-reference objects in external projects (Python, NumPy, PyTorch, Plotly):

````md
{class}`torch.nn.Module`
{func}`torch.vmap`
{class}`numpy.ndarray`
{func}`python:print`
{class}`python:dict`
{external:py:class}`torch.Tensor`
````

{class}`torch.nn.Module`
{func}`torch.vmap`
{class}`numpy.ndarray`

---

## `autodoc2`

Autodoc2 generates API docs automatically from the `src/occhio` package. The generated files live in `docs/apidocs/`.

### Referencing autodoc2-generated objects

````md
{py:class}`occhio.ToyModel`
{py:meth}`occhio.ToyModel.fit`
{py:func}`occhio.distributions.base.Distribution.sample`
{py:mod}`occhio.distributions`
{py:attr}`occhio.ToyModel.W`
````

### Docstring style (Napoleon — Google style)

```python
def fit(self, steps: int = 10_000, batch_size: int = 256) -> "ToyModel":
    """Train the model.

    Args:
        steps: Number of training steps.
        batch_size: Samples per step.

    Returns:
        The trained model instance.

    Raises:
        ValueError: If steps < 1.

    Example:
        >>> model = ToyModel(distribution, autoencoder)
        >>> model.fit(steps=5000)
    """
```

---

## `sphinx_copybutton`

All code blocks automatically get a copy button. No special syntax needed — it is applied globally.

To exclude prompt characters from being copied:

````md
```{code-block} console
:copyable: true

$ pip install occhio
$ python -c "import occhio"
```
````

```{code-block} console
$ pip install occhio
$ python -c "import occhio"
```

---

## `notfound.extension`

Automatically serves a custom 404 page. No syntax — it works transparently.

---

## `sphinx.ext.linkcode`

Adds `[source]` links on API doc pages pointing to GitHub. No syntax — configured via `linkcode_resolve` in `conf.py`.

---

## `sphinx.ext.duration`

Reports page build durations in the build log. No syntax — purely a build diagnostic.

---

## `sphinx_llm.txt`

Generates `/llms.txt` and `/llms-full.txt` for LLM consumption. No syntax — automatic.

---

## `eval-rst` Escape Hatch

When you need raw reStructuredText inside MyST Markdown:

````md
```{eval-rst}
.. note::
   This is written in raw RST inside a MyST file.
```
````

```{eval-rst}
.. note::
   This is written in raw RST inside a MyST file.
```

---

## MyST Frontmatter (YAML)

````md
---
myst:
  html_meta:
    "description lang=en": "Page description for SEO"
    "keywords": "occhio, superposition, toy models"
  substitutions:
    project: "occhio"
---

# Page Title

This is the {{ project }} documentation.
````

---

## Substitutions (via frontmatter)

````md
---
myst:
  substitutions:
    version: "0.2.0"
---

Current version: {{ version }}
````
