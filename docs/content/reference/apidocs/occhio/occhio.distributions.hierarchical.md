# {py:mod}`occhio.distributions.hierarchical`

```{py:module} occhio.distributions.hierarchical
```

```{autodoc2-docstring} occhio.distributions.hierarchical
:parser: _ext.google_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TreeNode <occhio.distributions.hierarchical.TreeNode>`
  - ```{autodoc2-docstring} occhio.distributions.hierarchical.TreeNode
    :parser: _ext.google_docstring_parser
    :summary:
    ```
* - {py:obj}`HierarchicalSparse <occhio.distributions.hierarchical.HierarchicalSparse>`
  - ```{autodoc2-docstring} occhio.distributions.hierarchical.HierarchicalSparse
    :parser: _ext.google_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} TreeNode
:canonical: occhio.distributions.hierarchical.TreeNode

```{autodoc2-docstring} occhio.distributions.hierarchical.TreeNode
:parser: _ext.google_docstring_parser
```

````{py:attribute} index
:canonical: occhio.distributions.hierarchical.TreeNode.index
:type: int
:value: >
   None

```{autodoc2-docstring} occhio.distributions.hierarchical.TreeNode.index
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} depth
:canonical: occhio.distributions.hierarchical.TreeNode.depth
:type: int
:value: >
   None

```{autodoc2-docstring} occhio.distributions.hierarchical.TreeNode.depth
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} parent
:canonical: occhio.distributions.hierarchical.TreeNode.parent
:type: int | None
:value: >
   None

```{autodoc2-docstring} occhio.distributions.hierarchical.TreeNode.parent
:parser: _ext.google_docstring_parser
```

````

````{py:attribute} children
:canonical: occhio.distributions.hierarchical.TreeNode.children
:type: list[int]
:value: >
   None

```{autodoc2-docstring} occhio.distributions.hierarchical.TreeNode.children
:parser: _ext.google_docstring_parser
```

````

`````

`````{py:class} HierarchicalSparse(n_features: int, p_base: float = 0.8, depth_decay: float = 0.9, p_by_depth: list[float] | None = None, max_children: int = 5, device: torch.device | str = 'cpu', generator: torch.Generator | None = None)
:canonical: occhio.distributions.hierarchical.HierarchicalSparse

Bases: {py:obj}`occhio.distributions.base.Distribution`

```{autodoc2-docstring} occhio.distributions.hierarchical.HierarchicalSparse
:parser: _ext.google_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} occhio.distributions.hierarchical.HierarchicalSparse.__init__
:parser: _ext.google_docstring_parser
```

````{py:method} generate_new_tree() -> None
:canonical: occhio.distributions.hierarchical.HierarchicalSparse.generate_new_tree

```{autodoc2-docstring} occhio.distributions.hierarchical.HierarchicalSparse.generate_new_tree
:parser: _ext.google_docstring_parser
```

````

````{py:method} _build_auxiliary_structures() -> None
:canonical: occhio.distributions.hierarchical.HierarchicalSparse._build_auxiliary_structures

```{autodoc2-docstring} occhio.distributions.hierarchical.HierarchicalSparse._build_auxiliary_structures
:parser: _ext.google_docstring_parser
```

````

````{py:method} _get_p_fire(depth: int) -> float
:canonical: occhio.distributions.hierarchical.HierarchicalSparse._get_p_fire

```{autodoc2-docstring} occhio.distributions.hierarchical.HierarchicalSparse._get_p_fire
:parser: _ext.google_docstring_parser
```

````

````{py:method} sample(batch_size: int) -> torch.Tensor
:canonical: occhio.distributions.hierarchical.HierarchicalSparse.sample

```{autodoc2-docstring} occhio.distributions.hierarchical.HierarchicalSparse.sample
:parser: _ext.google_docstring_parser
```

````

````{py:method} get_tree_stats() -> dict
:canonical: occhio.distributions.hierarchical.HierarchicalSparse.get_tree_stats

```{autodoc2-docstring} occhio.distributions.hierarchical.HierarchicalSparse.get_tree_stats
:parser: _ext.google_docstring_parser
```

````

````{py:method} get_expected_active() -> torch.Tensor
:canonical: occhio.distributions.hierarchical.HierarchicalSparse.get_expected_active

```{autodoc2-docstring} occhio.distributions.hierarchical.HierarchicalSparse.get_expected_active
:parser: _ext.google_docstring_parser
```

````

````{py:method} to(device: torch.device | str)
:canonical: occhio.distributions.hierarchical.HierarchicalSparse.to

```{autodoc2-docstring} occhio.distributions.hierarchical.HierarchicalSparse.to
:parser: _ext.google_docstring_parser
```

````

`````
