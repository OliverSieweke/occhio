## Distributions Overview
This document exists to make it easier to find a distribution which might be interesting for your use case. 

### base.py
`Distribution`
The base class manages the `device` the `generator` and includes helpful functions for sampling distributions using the `generator`. 

### sparse.py
This contains the simplest of the distributions. Simply a sparse vector with a given probability distribution.

`SparseUniform` is 0 with probability `p_active` and Uniform([0, 1]) otherwise.
`SparseExponential`is 0 with probability `p_active` and Exp(scale) otherwise.

### correlated.py
This file contains distributions whose structure is encoded in pairs of the input distribution. For all of these distributions the features come in pairs.

`CorrelatedPairs`
We first check if a given pair is active (with probability `p_active`), if so each of the individual entries are active with probability `p_individual`. One can easily check that the correlation between pairs is
$$ \frac{p_i (1- p_a)}{1 - p_a p_i}. $$

`HierarchicalPairs`
We first check if the first entry of any pair is active (with probability `p_active`). If so its pair is active with probability `p_follow`.

Correlation between pairs is
$$ \sqrt{\frac{p_f * (1 - p_a)}{1 - p_f * p_a}} $$

`AnticorrelatedPairs`
Here the pairs are mutually exclusive. Each is active with probability `p_active`. 

### relational.py
Inspired by [this paper](https://arxiv.org/abs/2407.14662v1). Essentially we already overlay 2 sparse distributions, but in different bases. Note that this means that the generated vector will in general not be sparse themselves (in the sense of most entries being equal to zero).

`RelationalSimple`
This distribution adds two `SparseUniform` distributions in different basis. The basis of the first is the standard basis.

`MultiRelational`
This distribution adds $k$ `SparseUniform` distributions in different basis. Each basis is sampled iid from $O(n)$.

### hierarchical.py
`HierarchicalSparse`
This samples a tree. The root node is sampled first. Each node thereafter can only fire if the parent fired. You can specify different probabilities by depth. 

### dag.py
Directed Acyclic Graphs! Both distributions have an underlying DAG structure, but use them in different ways. The DAGs are generated in an [Erdos-Renyi](https://en.wikipedia.org/wiki/Erdős–Rényi_model) type process.

`DAGDistribution`
Here the activations are viewed as binary. A node activates if the parent is active with probability `p_active`. Activations are sampled by Unif([0, 1]).

`DAGBayesianPropagation`
Here the activation size matters. A node is active with probability $1 - \prod_{j\in \textup{active parents}} (1 - a_j)$. 