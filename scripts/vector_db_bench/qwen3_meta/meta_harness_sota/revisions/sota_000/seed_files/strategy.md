# Strategy

Primary objective:
- reach a valid 4000+ QPS solution

Search principles:
- treat exact full scan as a temporary baseline, not the end state
- make architectural progress toward ANN / IVF-style shortlist generation early
- protect the best valid state with checkpoints and fast rollback
- use quick checkpoints during search and full benchmarks only for credible milestone candidates
- defer low-level SIMD or unsafe tuning until shortlist generation and reranking are structurally sound
