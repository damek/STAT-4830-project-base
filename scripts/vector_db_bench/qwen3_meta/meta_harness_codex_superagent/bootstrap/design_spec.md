# Design Spec

Target architecture class:

- coarse partitioning or IVF-like shortlist generation
- compact contiguous storage for partition-local vectors
- multi-probe shortlist expansion
- exact rerank over shortlisted candidates
- only then deeper SIMD / memory-layout tuning

Avoid fake ANN designs that still depend on:
- global locked full storage in the hot path
- per-candidate linear lookup back into a master vector list
- single-probe only search

Any ANN path should become a real candidate-pruning system, not just new names around brute-force logic.
