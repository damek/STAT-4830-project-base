# Strategy

- Optimize for a valid `22000+ QPS` solution, not a pretty exact-search baseline.
- Treat `21500 QPS` as the public SOTA target to beat and `22000 QPS` as the working campaign goal.
- Treat brute-force and exact-scan polishing as temporary scaffolding only.
- Move quickly toward ANN / IVF-style shortlist generation.
- Prefer architecture changes that reduce the scanned candidate set before low-level kernel tuning.
- Use this file as a living strategy document and rewrite it when the campaign learns something important.
