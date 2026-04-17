# Design Spec

Target architecture class:
- coarse clustering or bucketed shortlist generation
- exact reranking over the shortlisted candidates
- compact cluster-local memory layout
- tuned probe count with recall protection

Implementation direction:
1. preserve a valid exact baseline
2. add a clustering / shortlist structure during insert
3. query centroids or buckets first
4. probe a small subset of clusters
5. rerank candidates exactly
6. only then optimize the hot distance kernel and memory layout
