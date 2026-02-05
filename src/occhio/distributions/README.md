## Distributions

### base.py
`Distribution`

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


`RelationalSimple`
`MultiRelational`

### hierarchical.py
`HierarchicalSparse`

### dag.py
`DAGBayesianPropagation`
`DAGDistribution`