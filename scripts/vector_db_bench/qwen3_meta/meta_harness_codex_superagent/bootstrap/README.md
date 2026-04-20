# Codex Superagent Workspace

This workspace is a persistent Codex-only optimization campaign for vector-db-bench.

You may modify:
- the Rust solution in `src/`
- `Cargo.toml`
- the local harness files under `.meta_codex/`
- any helper scripts you create under `.meta_codex/tools/`

The driver will evaluate the workspace after each cycle and promote the best valid state to the mainline snapshot.

Use this directory as your persistent local harness:
- keep strategy current
- record research and decisions
- store scripts that help you move faster in later cycles

Read first:
- `.meta_codex/official_constraints.md`
- `.meta_codex/strategy.md`
- `.meta_codex/design_spec.md`
- `.meta_codex/research_notes.md`
- `.meta_codex/benchmark_policy.md`
- `.meta_codex/incumbent_record.md`
- `.meta_codex/milestones.md`
- `.meta_codex/campaign_journal.md`
