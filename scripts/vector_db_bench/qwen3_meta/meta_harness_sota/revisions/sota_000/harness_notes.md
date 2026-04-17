# Harness Notes

- architectural bet: Qwen reaches SOTA only after an explicit pivot away from exact scan toward ANN / IVF-style shortlist generation and exact reranking.
- intended causal mechanism: give Qwen a strong teacher package, checkpoint/restore tools, and milestone framing so it can sustain long-horizon architectural search within a revision without losing the best valid state.
