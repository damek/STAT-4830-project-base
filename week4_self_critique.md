# Week 4 Self-Critique

## OBSERVE: Reading the Report Critically

Re-reading the report as a first-time reader, the problem statement and technical approach sections are detailed and well-motivated. The initial results are real and reproducible. CP26 matched SOTA, the metrics tables tell a clear improvement story, and the failure cases are honestly documented. However, the report leans heavily on describing *what the framework does* rather than demonstrating *our own understanding and contribution*. A reader could reasonably ask: "Did you build anything, or did you just run someone else's code and report the numbers?" The next steps section proposes interesting directions (multi-agent, reward estimation, token efficiency) but none of them have been prototyped or even scoped with enough specificity to convince a reader they're feasible within the project timeline.

## ORIENT: Analyzing Our Work

### Strengths

- **Concrete, reproducible results.** CP26 reached the known SOTA (2.636) in 5 epochs with full per-step metrics logged. The results are backed by exact numbers, wall-clock times, and documented failure cases.
- **Clear research pivot.** Dropping AC1 to focus on CP26 efficiency is a well-justified decision grounded in data (AC1 cost 50% more credits for weaker results). The report explains the reasoning rather than just stating the choice.
- **Honest documentation of failures.** The debugging section (SLURM bypass, hung checkpoints, SSH kills, missing checkpoints) shows real engineering work and doesn't hide the messy parts.

### Areas for Improvement

- **No ablation or controlled experiment.** Every claim about RL driving improvement is correlational. We have not run the frozen-model baseline, so we cannot attribute any of the gains to the weight updates vs. the PUCT buffer. This is the single biggest gap, without it, the core thesis of the project ("RL improves test-time search") is unsubstantiated.
- **Shallow treatment of the RL algorithm itself.** The report describes the advantage estimator and PUCT sampler at a surface level but does not analyze *how* they interact, whether the hyperparameters are appropriate, or what the training loss curves look like. We report reward means and correctness rates but never examine the actual LoRA weight update magnitudes, KL divergence from the base model, or per-group advantage distributions.
- **Next steps are too broad.** Ideas like "generator-critic architectures" and "multi-agent setups" sound interesting but are underspecified. We have not scoped the implementation effort, identified which parts of the TTT-Discover codebase would need modification, or estimated whether these are feasible given our remaining Tinker credits and timeline.

### Critical Risks and Assumptions

We are assuming the LoRA weight updates are the primary driver of improvement, but we have zero evidence for this; the PUCT buffer alone could explain the results. If the frozen-model ablation shows the buffer is sufficient, we lose our core contribution and need to pivot. Additionally, we are operating on a limited Tinker API credit budget, and every failed or uninformative experiment directly reduces our capacity to run the ablations and optimizations we've planned. We need to be disciplined about which experiments to prioritize.

## DECIDE: Concrete Next Actions

- **Run the frozen-model ablation next week.** Modify `cp26.sh` to disable LoRA weight updates (or set learning rate to 0) while keeping PUCT sampling and evaluation identical. Compare step-by-step metrics against the existing 5-epoch run. This is non-negotiable — without it, the project cannot make any causal claims.
- **Add deeper RL analysis to the results section.** Extract and plot the training loss, KL divergence, per-group advantage distributions, and LoRA weight norms from the existing run's logs. This requires no additional API credits; the data is already in the W&B logs and local metric files. The goal is to show we understand the RL dynamics, not just the end-result scores.
- **Scope one algorithmic optimization concretely.** Pick the most feasible idea from the "Ideas for algorithmic optimization" section (likely token efficiency or reward shaping), identify the exact code changes needed in the TTT-Discover codebase, estimate the Tinker credit cost, and either implement it or write a detailed implementation plan with code snippets. Vague ideas need to become concrete proposals.

## ACT: Resource Needs

We need continued access to the Tinker API with enough credits for at least 2-3 more 5-epoch CP26 runs (the frozen-model ablation plus one optimization experiment). We also need to learn how to extract detailed training diagnostics from the Tinker API, specifically, the training loss per step, gradient norms, and KL divergence. Finally, to implement any reward shaping or token efficiency changes, we need a deeper understanding of the `tinker_cookbook.recipes.ttt.train` module's internal structure, particularly the `compute_advantages` and `generate_samples` code paths.