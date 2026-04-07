# {py:mod}`occhio.mcp.servers.alignmentforum`

```{py:module} occhio.mcp.servers.alignmentforum
```

```{autodoc2-docstring} occhio.mcp.servers.alignmentforum
:parser: _ext.google_docstring_parser
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`af_search_posts <occhio.mcp.servers.alignmentforum.af_search_posts>`
  - ```{autodoc2-docstring} occhio.mcp.servers.alignmentforum.af_search_posts
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`af_get_post <occhio.mcp.servers.alignmentforum.af_get_post>`
  - ```{autodoc2-docstring} occhio.mcp.servers.alignmentforum.af_get_post
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`af_get_comments <occhio.mcp.servers.alignmentforum.af_get_comments>`
  - ```{autodoc2-docstring} occhio.mcp.servers.alignmentforum.af_get_comments
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`af_get_user <occhio.mcp.servers.alignmentforum.af_get_user>`
  - ```{autodoc2-docstring} occhio.mcp.servers.alignmentforum.af_get_user
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`af_get_tag <occhio.mcp.servers.alignmentforum.af_get_tag>`
  - ```{autodoc2-docstring} occhio.mcp.servers.alignmentforum.af_get_tag
    :parser: _ext.google_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`READ_ONLY_ANNOTATIONS <occhio.mcp.servers.alignmentforum.READ_ONLY_ANNOTATIONS>`
  - ```{autodoc2-docstring} occhio.mcp.servers.alignmentforum.READ_ONLY_ANNOTATIONS
    :parser: _ext.google_docstring_parser
    :summary:
    ```
````

### API

````{py:data} READ_ONLY_ANNOTATIONS
:canonical: occhio.mcp.servers.alignmentforum.READ_ONLY_ANNOTATIONS
:value: >
   None

```{autodoc2-docstring} occhio.mcp.servers.alignmentforum.READ_ONLY_ANNOTATIONS
:parser: _ext.google_docstring_parser
```

````

````{py:function} af_search_posts(view: str = 'top', limit: int = 10, tag_id: str | None = None) -> str
:canonical: occhio.mcp.servers.alignmentforum.af_search_posts
:async:

```{autodoc2-docstring} occhio.mcp.servers.alignmentforum.af_search_posts
:parser: _ext.google_docstring_parser
```
````

````{py:function} af_get_post(id: str | None = None, slug: str | None = None, max_length: int = 10000) -> str
:canonical: occhio.mcp.servers.alignmentforum.af_get_post
:async:

```{autodoc2-docstring} occhio.mcp.servers.alignmentforum.af_get_post
:parser: _ext.google_docstring_parser
```
````

````{py:function} af_get_comments(post_id: str, limit: int = 20, view: str = 'postCommentsTop') -> str
:canonical: occhio.mcp.servers.alignmentforum.af_get_comments
:async:

```{autodoc2-docstring} occhio.mcp.servers.alignmentforum.af_get_comments
:parser: _ext.google_docstring_parser
```
````

````{py:function} af_get_user(slug: str | None = None, id: str | None = None) -> str
:canonical: occhio.mcp.servers.alignmentforum.af_get_user
:async:

```{autodoc2-docstring} occhio.mcp.servers.alignmentforum.af_get_user
:parser: _ext.google_docstring_parser
```
````

````{py:function} af_get_tag(id: str | None = None, name: str | None = None, limit: int = 10) -> str
:canonical: occhio.mcp.servers.alignmentforum.af_get_tag
:async:

```{autodoc2-docstring} occhio.mcp.servers.alignmentforum.af_get_tag
:parser: _ext.google_docstring_parser
```
````
