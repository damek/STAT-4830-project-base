# Strategy

- Treat this run as a persistent engineering campaign.
- Treat each cycle as a short actuator burst for Qwen, under the same tool-call budget as the baseline worker setup.
- Assume Codex is the conductor between bursts and will revise the harness over time; your job in this burst is to advance the codebase cleanly and leave a useful checkpoint.
- Keep one promoted mainline branch healthy.
- Use experiments for risky moves and promote them only after valid evidence.
- Prioritize structural ANN transitions over local exact-search polish.
- Stop repeating any benchmark that is failing with the same signature until code or benchmark parameters change.
