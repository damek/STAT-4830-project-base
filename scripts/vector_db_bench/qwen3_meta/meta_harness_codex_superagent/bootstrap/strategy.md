# Strategy

- Optimize for a valid `4000+ QPS` solution, not a pretty exact-search baseline.
- Treat brute-force and exact-scan polishing as temporary scaffolding only.
- Move quickly toward ANN / IVF-style shortlist generation.
- Prefer architecture changes that reduce the scanned candidate set before low-level kernel tuning.
- Use this file as a living strategy document and rewrite it when the campaign learns something important.
