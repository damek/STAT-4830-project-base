# Week 4 Report Draft 1

## 1. Problem Statement

### What we are optimizing

We study how test-time training on a large language model during inference can improve AI-driven search on strictly verifiable optimization problems.

Large language models are very good at generating solutions but relatively bad at verifying them. The generation-verification gap has been approached in many different ways, including training an LLM or reward model to assess the reward of a given solution (see [https://arxiv.org/abs/2411.15594](https://arxiv.org/abs/2411.15594) among others). Here we take a different approach: reinforce an LLM at test time to progressively improve on a *strictly verifiable* problem, with the reward signal shaped by deterministic evaluation of its own prior attempts.

We are heavily inspired by [AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/), which uses LLMs to generate novel solutions to verifiable math and algorithm design problems in an evolutionary approach, achieving remarkable results including new state-of-the-art constructions in combinatorics and algorithm engineering. AlphaEvolve uses a prompt-engineering harness: it draws existing solutions from a bank, asks the LLM to propose edits, evaluates the result, and repeats. We extend this paradigm by adding reinforcement learning based test-time training on top: rather than relying solely on in-context prompting, we actively fine-tune the model's weights with LoRA adapters after each batch of evaluated samples, steering the model toward higher-quality generations over successive epochs.

To test out the validity of this approach to start, we use the **TTT-Discover** framework, which implements this RL-at-test-time loop using a cloud-based fine-tuning and inference service (Tinker API) for the LLM, combined with local deterministic evaluation. We benchmark on two problems drawn from the testing suite:

1. **AC1 (Autocorrelation Inequality Bound Minimization).** Given a non-negative step function `f = (a_0, ..., a_{n-1})`, minimize the upper bound `C_1 = 2n * max(conv(f, f)) / (sum(f))^2`, where `conv(f, f)` is the discrete autocorrelation. The target is to achieve a bound ≤ 1.5030. The LLM generates Python code that searches for such sequences, and the evaluation function is deterministic and fast.
2. **CP26 (Circle Packing in the Unit Square).** Pack `N = 26` circles in `[0, 1]^2` to maximize the sum of radii, subject to non-overlap constraints `‖c_i - c_j‖ ≥ r_i + r_j` and boundary constraints (all circles fully inside the square). The target sum of radii is ≥ 2.636. The LLM generates Python code that produces circle configurations, which are validated geometrically.

Both problems are self-contained, deterministic, and strictly verifiable. Evaluation requires no human judgment and yields an unambiguous numerical score.

### Why this matters

Many scientific and engineering tasks are hard to solve directly but easy to verify once a candidate is proposed. It is relatively trivial to assess how fast a GPU kernel executes a specific operation, but very hard to write a kernel that beats existing state-of-the-art implementations. If an AI system can reliably improve candidate quality in these settings, it could generalize to broader classes of verifiable discovery problems, from algorithm engineering to materials science to mathematical conjecture generation.

The AlphaEvolve-style agent approach represents a paradigm shift in how we utilize LLMs: rather than relying on fragile one-shot or few-shot generation, the system treats the LLM as a proposal engine inside a loop where a deterministic verifier provides ground-truth feedback. This sidesteps the hallucination problem entirely, as every accepted solution is provably valid.

By adding RL-based test-time training on top of this harness, we aim to unlock the same kind of specialization that made AlphaGo and AlphaStar superhuman in games, but directed at open scientific and engineering problems. The key insight is that test-time RL lets the model "overfit" productively to a single hard problem, concentrating its capacity where it matters most.

### Success metrics

- **Primary:** Best valid objective score found within a fixed epoch budget (5 epochs for initial testing). For AC1, this is the lowest autocorrelation bound achieved; for CP26, the highest sum of radii.
- **Secondary:** Improvement curve - best-so-far score vs. training step, tracked via wandb.
- **Efficiency:** Valid proposal rate (fraction of LLM outputs that parse and produce valid constructions), time-to-best-score, and tokens consumed per improvement.

### Constraints

- Every candidate must pass strict deterministic verification: For AC1, the sequence must be non-negative and produce a finite evaluation. For CP26, all circles must satisfy non-overlap and boundary constraints within numerical tolerance.
- LLM inference and fine-tuning run via the Tinker API (cloud-hosted); evaluation runs locally on CPU.
- We use a fixed model (`openai/gpt-oss-120b`) with LoRA rank 32 adapters - we do not modify the base model weights.
- Compute budget for initial experiments: 5 training epochs per problem, 64 samples per batch, 8 groups of 8 trajectories.

### Data requirements

- No pre-existing training data. Both problems are self-play: the model generates candidates, they are evaluated, and the (prompt, generation, reward) tuples become the training data for the next RL update.
- Logged artifacts per run: generated code, evaluation scores, training loss, LoRA checkpoint weights, PUCT buffer states.
- All metrics and learning curves logged to wandb for reproducibility.

### Failure modes and risks

- **Low valid-proposal rate:** If the model produces mostly unparseable or invalid code, the RL signal is too sparse for meaningful weight updates. (Mitigated by the PUCT sampler, which biases toward previously successful states.)
- **Evaluation bottleneck:** AC1 evaluations can time out at 1000s per sample; if too many samples time out, each epoch takes hours. (Mitigated by parallelizing evaluations across CPU cores via Ray.)
- **Plateau after initial gains:** The model may find a local optimum quickly and fail to escape. (Mitigated by the entropic adaptive beta advantage estimator, which adjusts exploration pressure.)
- **LoRA capacity limits:** Rank-32 LoRA adapters may not have sufficient capacity to encode problem-specific knowledge for very hard instances.

## 2. Technical Approach

### Mathematical framing

We frame each problem as a Markov Decision Process (MDP) where a single "episode" consists of one LLM generation conditioned on the problem prompt and (optionally) prior best solutions:

- **State** `s_t`: the current prompt, which includes the problem specification, evaluation function source code, and the best-known construction from the PUCT buffer.
- **Action** `a_t`: the LLM's generated output, a Python code block that produces a candidate construction.
- **Reward** `r_t`: a scalar derived from deterministic evaluation of the generated construction.

**AC1 reward.** The evaluation function computes `C_1 = 2n * max(conv(f, f)) / (sum(f))^2`. The reward uses a scaled reciprocal: `r = 1 / C_1` when the code is valid and produces a finite bound, and `r = 0` otherwise. Lower bounds yield higher rewards, incentivizing the model to minimize the autocorrelation ratio.

**CP26 reward.** The evaluation validates geometric constraints (non-overlap: `‖c_i - c_j‖ ≥ r_i + r_j`; boundary: each circle fully within `[0, 1]^2`) and computes `score = sum(r_i)` for all 26 circles. The reward is linear: `r = sum(r_i)` when valid, `r = 0` otherwise. Higher packing density yields higher rewards.

### Algorithmic design: TTT-Discover training loop

The system executes a multi-phase loop for each training step (epoch):

**Sampling -** The PUCT (Predictor + Upper Confidence bounds applied to Trees) sampler selects states from a replay buffer. Each state encodes a problem prompt augmented with the best-known solution at that node. For the initial epoch, states are generated from random initialization. The sampler balances exploitation (re-visiting high-reward states) with exploration (trying under-visited branches), using UCB-style scoring.

**LLM generation -** For each sampled state, the model (with current LoRA adapters) generates a batch of candidate solutions via the Tinker API. We generate 64 samples per batch, organized as 8 groups of 8 trajectories each. Generation uses the model `openai/gpt-oss-120b` with LoRA rank 32 and a token budget of 20,000 tokens per generation.

**Local evaluation -** Each generated code block is executed locally in a sandboxed environment with a timeout (1,100s for AC1, 305s for CP26). The evaluation is fully deterministic:

- For AC1: execute the generated `propose_candidate()` function and compute the autocorrelation bound.
- For CP26: execute the generated `run_packing()` function and validate the circle configuration geometrically.

Evaluations are parallelized across CPU cores using **Ray**, with each evaluation task assigned 2 CPU cores.

**Advantage estimation and RL training -** The (prompt, generation, reward) tuples are assembled into a training batch. An entropic adaptive beta advantage estimator computes per-sample advantages, adjusting the KL penalty dynamically to maintain a target entropy level. This prevents the model from collapsing to a single strategy too early. The LoRA adapter weights are updated via the Tinker training API using these advantages, with a learning rate of `4e-5`.

**Buffer update -** The PUCT buffer is updated with new (state, value) pairs from the evaluated samples. High-scoring constructions become new nodes in the search tree, biasing future sampling toward promising regions of the solution space. The buffer grows over epochs, accumulating a library of increasingly strong constructions.

This loop repeats for a fixed number of epochs (5 for our initial experiments).

### Implementation architecture

The system has a **split compute** design:

- **Tinker API (cloud):** Hosts the base LLM and LoRA adapters. Handles all inference (sampling) and weight updates (training). This eliminates the need for local GPUs.
- **Local machine (CPU):** Runs the deterministic evaluators, the PUCT sampler logic, the Ray-based parallelization, and the orchestration code. Our server has 256 CPU cores and 1.5 TB RAM, enabling high parallelism for evaluation.
- **WanDB (cloud):** Experiment tracking and metric visualization.

The orchestration is handled by the `tinker_cookbook.recipes.ttt.train` module, which coordinates the async communication between local evaluation and remote LLM calls. The training script uses Hydra for configuration management, allowing hyperparameters to be specified at launch time.

### Key hyperparameters


| Parameter           | Value                     | Description                                       |
| ------------------- | ------------------------- | ------------------------------------------------- |
| Model               | `openai/gpt-oss-120b`     | Base LLM for generation                           |
| LoRA rank           | 32                        | Rank of low-rank adapters                         |
| Learning rate       | 4e-5                      | LoRA weight update rate                           |
| Batch size          | 64                        | Samples per training step                         |
| Group size          | 8                         | Trajectories per group (for advantage estimation) |
| Max tokens          | 20,000                    | Token budget per LLM generation                   |
| Advantage estimator | Entropic adaptive beta    | KL-regularized advantage computation              |
| Sampler             | PUCT with backpropagation | Tree-search-based state selection                 |
| Eval timeout (AC1)  | 1,100s                    | Max time for AC1 code execution                   |
| Eval timeout (CP26) | 305s                      | Max time for CP26 code execution                  |
| Epochs              | 5                         | Training steps for initial experiments            |


### Validation plan

- **Correctness of evaluation:** Both verifiers are deterministic and unit-tested as part of the TTT-Discover task suite. AC1 uses `numpy.convolve`; CP26 uses pairwise Euclidean distance checks.
- **Training signal quality:** We monitor `frac_mixed` (fraction of groups with both successes and failures). A value near 1.0 indicates the RL training has good contrastive signal. A value near 0.0 (all-good or all-bad groups) would indicate the reward is uninformative.
- **Convergence tracking:** Performance curves on wanDB should show monotonic improvement in best-so-far score. KL divergence from the base model should grow gradually, not spike.
- **Reproducibility:** Fixed random seeds, deterministic evaluation, and full metric logging ensure runs can be reproduced and compared.

### Resource requirements

- **Compute:** 256-core CPU server with 1.5 TB RAM for local evaluation and Ray parallelization. No local GPU required. All LLM compute is handled by the Tinker API.
- **API access:** Tinker API key for model inference and training. Weights & Biases API key for experiment logging.
- **Estimated runtime per epoch:** ~60 minutes for AC1 (dominated by 1000s evaluation timeouts), ~15–30 minutes for CP26 (shorter 305s timeouts). Total for 5 epochs: ~5–6 hours for AC1, ~2–3 hours for CP26.
- **Concurrent execution:** Both problems run simultaneously on the same machine, each using ~64 CPU cores via Ray.

## 3. Initial Results

### Experiment overview

We completed full 5-epoch training runs for both **CP26** (Circle Packing with 26 circles) and **AC1** (Autocorrelation Inequality Bound Minimization), running concurrently on a 256-core / 1.5 TB RAM CPU server. All LLM inference and LoRA weight updates were handled remotely via the Tinker API (`openai/gpt-oss-120b`, LoRA rank 32); local compute was used only for deterministic evaluation (parallelized via Ray) and PUCT buffer management. Metrics were logged to Weights & Biases after each step.

Both runs used 64 samples per batch, 8 groups of 8 trajectories, PUCT with backpropagation sampling, entropic adaptive beta advantage estimation, and a learning rate of `4e-5`. Each run trained from a fresh base model (no prior fine-tuning) for 5 epochs (steps 0–4).

### CP26 results: Circle Packing (N = 26)

**Goal:** Maximize the sum of radii for 26 non-overlapping circles packed inside the unit square. Target: sum ≥ 2.636. Known SOTA: 2.635983.


| Step | Correctness | Format Rate | Avg Performance | Best Solution | Reward (mean) | PUCT Buffer | Wall Time |
| ---- | ----------- | ----------- | --------------- | ------------- | ------------- | ----------- | --------- |
| 0    | 19.9%       | 21.2%       | 2.2297          | 2.6203        | 0.443         | 70          | 30 min    |
| 1    | 31.9%       | 37.7%       | 2.3618          | 2.6288        | 0.754         | 125         | 33 min    |
| 2    | 42.0%       | 71.1%       | 2.4620          | 2.6343        | 1.035         | 217         | 28 min    |
| 3    | 42.8%       | 84.3%       | 2.5091          | 2.6343        | 1.074         | 339         | 39 min    |
| 4    | 61.5%       | 96.8%       | 2.5539          | **2.6360**    | 1.571         | 432         | 24 min    |


**Total training time:** ~2.6 hours (9,297 seconds across 5 steps).

**Key observations:**

- The **best solution of 2.6360** at Step 4 matches the SOTA benchmark target of ≥ 2.636 (the known optimum is 2.635983).
- **Correctness rate tripled** from 19.9% to 61.5% over training, indicating the model learned to generate valid circle configurations far more reliably.
- **Format compliance improved 4.6x** from 21.2% to 96.8%, showing the model learned proper code formatting rapidly — nearly all outputs were syntactically valid by Step 4.
- **Average performance improved by 14.5%** (2.2297 → 2.5539), meaning the typical valid solution improved substantially, not just the best one.
- The PUCT buffer grew from 70 to 432 entries, accumulating a rich library of valid constructions that biased future sampling toward high-quality regions.
- `frac_mixed = 1.0` throughout all 5 steps, confirming the RL training received consistently strong contrastive signal (every group had both successes and failures, ideal for advantage estimation).

### AC1 results: Autocorrelation Inequality Bound Minimization

**Goal:** Minimize the autocorrelation bound `C_1 = 2n * max(conv(f, f)) / (sum(f))^2`. Target: bound ≤ 1.5030.


| Step | Correctness | Format Rate | Best Bound (C₁) | Reward (mean) | Reward (max) | PUCT Buffer | Wall Time |
| ---- | ----------- | ----------- | --------------- | ------------- | ------------ | ----------- | --------- |
| 0    | 23.0%       | 26.7%       | 1.5199          | 0.113         | 0.658        | 70          | 63 min    |
| 1    | 25.5%       | 28.4%       | 1.5180          | 0.127         | 0.659        | 112         | 40 min    |
| 2    | 30.2%       | 38.6%       | 1.5180          | 0.146         | 0.659        | 155         | 38 min    |
| 3    | 37.1%       | 45.9%       | 1.5249          | 0.188         | 0.656        | 219         | 64 min    |
| 4    | 38.1%       | 51.9%       | **1.5146**      | 0.193         | 0.660        | 298         | 32 min    |


**Total training time:** ~3.9 hours (14,191 seconds across 5 steps).

**Key observations:**

- The **best bound of 1.5146** at Step 4 is very close to but does not reach the target of ≤ 1.5030 (a gap of 0.0116, or 0.77%).
- **Correctness rate improved 66%** from 23.0% to 38.1%. AC1 remains harder than CP26 for the model — even at Step 4, fewer than half of outputs produce valid evaluations.
- **Format compliance nearly doubled** from 26.7% to 51.9%, but remains substantially below the 96.8% achieved for CP26. This suggests AC1's code generation task is inherently more challenging.
- **Reward (max) was remarkably stable** at ~0.658–0.660 across all steps. Since reward = 1/C₁, this corresponds to bounds of approximately 1.515–1.520. The best-bound trajectory remained in a narrow range throughout, suggesting the model found a strong local optimum early and struggled to escape it.
- AC1 evaluation timeouts (1,100s vs. 305s for CP26) account for the longer per-step wall times: some samples require extended CPU time before yielding a result.
- `frac_mixed = 1.0` throughout all 5 steps, confirming the RL signal was consistently informative.

### Comparative analysis


| Metric                              | CP26                  | AC1                  |
| ----------------------------------- | --------------------- | -------------------- |
| **Best score achieved**             | 2.6360 (sum of radii) | 1.5146 (bound C₁)    |
| **Target**                          | ≥ 2.636               | ≤ 1.5030             |
| **Target reached?**                 | Yes (matched SOTA)    | No (gap of 0.77%)    |
| **Correctness improvement**         | 19.9% → 61.5% (3.1x)  | 23.0% → 38.1% (1.7x) |
| **Format rate improvement**         | 21.2% → 96.8% (4.6x)  | 26.7% → 51.9% (1.9x) |
| **Avg performance improvement**     | +14.5%                | +4.5%                |
| **Total training time**             | 2.6 hours             | 3.9 hours            |
| **Contrastive signal (frac_mixed)** | 1.0 (all steps)       | 1.0 (all steps)      |


The **CP26 task** responded much more strongly to test-time training than **AC1**. We hypothesize this is because:

1. CP26's solution space (circle coordinates and radii) admits smoother gradient-like improvements, whereas AC1 requires discovering discrete sequences with specific autocorrelation properties.
2. CP26's shorter evaluation timeout (305s vs. 1,100s) allowed faster iteration and more feedback per unit time.
3. Circle packing code is structurally simpler, making it easier for the model to learn correct formatting and valid constructions.

### Failure cases and debugging notes

Several infrastructure and operational issues were encountered and resolved during the initial experiments:

1. **SLURM dependency:** The original codebase assumed a SLURM cluster. We added a `--local` flag to bypass SLURM and run directly on the local machine, simulating single-node execution by setting `SLURM_JOB_NUM_NODES` and `SLURM_PROCID` environment variables.
2. **Dataset configuration error:** The CP26 task required `env=cp problem_idx=26` rather than `env=cp26`, causing an initial crash with `ValueError: Unknown dataset: cp26`.
3. **Hung checkpoint saves:** One CP26 run became stuck at `Starting save_checkpoint` with minimal CPU usage, likely due to a stalled network call to the Tinker API. The process had to be killed and restarted.
4. **Process termination on SSH disconnect:** Both training runs were killed by `SIGHUP` when the SSH session disconnected. This was resolved by running all training inside `tmux` persistent sessions.
5. **Missing local checkpoints:** The default `save_every=5` parameter meant no intermediate checkpoints were written to the local `checkpoints.jsonl` file during a 5-epoch run (the save condition `i_batch > start_batch and i_batch % save_every == 0` skipped batch 0 and only triggered at batch 5, which is one past the final batch). This was fixed by setting `save_every=1`.

### Known limitations

- **AC1 did not reach its target.** The best bound of 1.5146 falls short of the ≤ 1.5030 goal. Additional epochs or hyperparameter tuning (e.g., increased LoRA rank, different learning rate, larger batch sizes) may be needed.
- **Single seed.** Both experiments used `seed=0`. Variance across seeds is unknown, and the results may not be representative.
- **No ablation studies.** We have not yet compared against a frozen-model baseline (sampling without weight updates) to quantify how much of the improvement is attributable to RL fine-tuning vs. PUCT tree search alone.
- **Limited epoch budget.** Five epochs is a minimal training budget. The performance curves for both tasks suggest improvement has not yet plateaued, especially for AC1.
- **No comparison to AlphaEvolve-style prompting.** We have not tested an equivalent prompt-only approach (without LoRA weight updates) under the same compute budget.

## 4. Next Steps

Going forward, we think we should drop AC1 and focus exclusively on circle packing. AC1 consumed 3.9 hours of Tinker API credits for a 5-epoch run — 50% more than CP26 — while producing substantially weaker improvements (best bound still 0.77% from target vs. CP26 matching SOTA). With limited Tinker credits remaining, concentrating on a task where we have already achieved SOTA allows us to shift the research question from *"can we reach SOTA?"* to *"how fast and efficiently can we reach it, and how much does RL actually contribute?"*

### Immediate improvements needed

- **Reduce wasted compute in early epochs.** At Step 0, only 19.9% of samples were valid and only 21.2% had correct formatting — meaning roughly 80% of Tinker API tokens in the first epoch produced no useful training signal. The most impactful improvement would be increasing Step 0 yield through better prompt templates so the model produces valid packing code from the start.
- **Adaptive batch sizing.** Currently we generate a fixed 64 samples per epoch regardless of the model's current skill level. In early epochs when correctness is low, generating more samples (e.g., 128) could provide enough valid proposals for a meaningful weight update. In later epochs when correctness exceeds 60%, fewer samples may suffice, saving API credits.
- **Frozen-model ablation.** The most important unanswered question is: how much of CP26's improvement comes from the LoRA weight updates vs. the PUCT buffer simply accumulating good solutions? Running a frozen-model ablation (same PUCT sampling and evaluation loop, but no weight updates between epochs) is the critical next experiment. If the frozen model reaches SOTA at a similar speed, the RL fine-tuning may not be as significant as we think.

### Ideas for algorithmic optimization

These are longer-horizon research directions for improving the core TTT-Discover training loop:

- **Token efficiency and reasoning compression.** Each LLM generation is budgeted at 20,000 tokens, but much of that output may be verbose reasoning, comments, or boilerplate that does not contribute to solution quality. Constraining the generation format (e.g., requiring only the core numerical output or a minimal code skeleton) could reduce token usage per sample without degrading performance, allowing more samples per API dollar. Alternatively, a two-phase generation — a short "plan" pass followed by a compact "code-only" pass — could let the model reason cheaply before committing tokens to the actual solution.
- **Generator-critic and multi-agent architectures.** The current system uses a single LLM for both proposing solutions and learning from failures. A generator-critic split (where one model proposes candidate packings and a second, lighter model scores or filters them before expensive evaluation) could reduce the number of proposals that reach the verifier while still providing learning signal. Similarly, a multi-agent setup where several independently fine-tuned LoRA adapters propose solutions in parallel, with the best solutions shared across agents, could increase diversity and escape local optima faster than a single model's trajectory.
- **Improved reward estimation.** The current reward is binary at the group level: valid proposals receive their raw score, invalid ones get 0. This creates a sparse signal problem, especially in early epochs. Learned or heuristic reward estimators could assign meaningful intermediate scores. For instance, estimating how close an invalid configuration is to feasibility (number of constraint violations, total overlap magnitude) so the model receives gradient signal even from failed attempts. Unlike static partial-credit shaping, a trained estimator could adapt over epochs as the model's failure modes evolve.
- **PUCT search optimization.** The PUCT buffer grew from 70 to 432 entries over 5 steps, but we have no insight into whether buffer size, diversity, or node selection strategy is optimal. Understanding which buffer entries the sampler actually draws from , and whether old, low-quality entries pollute the state distribution, could reveal opportunities to prune, reweight, or restructure the search tree for faster convergence.

### Technical challenges to address

- **Isolating RL from search.** Even after a frozen-model ablation, disentangling the RL contribution is non-trivial. The LoRA weight updates and the PUCT buffer interact: better weights produce better samples that enrich the buffer, which in turn provides better prompts that amplify the effect of the weights. Designing experiments that cleanly separate these intertwined effects is a methodological challenge.
- **Reward distribution and advantage estimation.** CP26 rewards range from 0 (invalid) to ~2.636 (near-SOTA), with a large mass at 0. The entropic adaptive beta advantage estimator is supposed to handle this, but we don't know if it's optimal for this reward shape. Heavy-tailed or bimodal reward distributions may require alternative advantage estimators (e.g., rank-based or clipped normalization) to provide stable learning signal.

### Questions we need help with

- **What is the optimal PUCT exploration-exploitation balance for circle packing?** The UCB-style scoring in the PUCT sampler has tunable constants that control how aggressively the system explores under-visited states vs. exploiting known high-reward states. We have not tuned these and are using the framework defaults.
- **Is there a learning rate schedule that works better than constant `4e-5`?** A warm-up schedule might prevent destructive early updates (when most samples are invalid), while cosine decay in later epochs could help fine-grained refinement near SOTA.
- **How should we handle the reward distribution's heavy tail?** CP26 rewards range from 0 (invalid) to ~2.636 (near-SOTA), with a large mass at 0. The entropic adaptive beta advantage estimator is supposed to handle this, but we don't know if it's optimal for this reward shape.

### Alternative approaches to try

- **Prompt-only baseline (no RL).** Use the same PUCT sampling and evaluation loop but freeze the model weights entirely. This tests whether the search strategy alone (accumulating good solutions in the buffer and conditioning future prompts on them) is sufficient to reach SOTA, without any fine-tuning.
- **Reward shaping with partial credit.** Instead of the binary valid/invalid reward (0 or `sum(r_i)`), assign partial credit for near-valid configurations (e.g., configurations where circles overlap by a small epsilon). This could provide gradient signal in early epochs when most proposals violate constraints.
- **Curriculum from easier instances.** Train first on CP10 or CP15 (fewer circles, simpler packing), save the LoRA weights, then fine-tune on CP26. The weights learned on easier problems may transfer useful geometric intuitions, reducing the number of CP26 epochs needed.
- **Smaller model, faster iteration.** If a smaller model available through Tinker produces lower-quality but faster and cheaper samples, it might reach SOTA in more epochs but fewer total API credits — a worthwhile tradeoff if credits are the binding constraint.

### What we've learned so far

- **RL-based test-time training works.** CP26 reached the known SOTA (2.636) within 5 epochs and 2.6 hours, starting from a base model with no prior circle-packing knowledge. This validates the core thesis that fine-tuning an LLM's weights at test time, guided by deterministic verification, can solve hard combinatorial optimization problems.
- **The model learns syntax before semantics.** Format compliance (21.2% → 96.8%) improved much faster than correctness (19.9% → 61.5%), which improved faster than solution quality (avg performance +14.5%). This suggests the LoRA updates first teach the model to produce parseable code, then valid configurations, then high-quality ones — a natural curriculum that emerges from the reward signal alone.
- **Contrastive signal remained strong throughout.** `frac_mixed = 1.0` at every step means every training group contained both successes and failures, providing ideal conditions for advantage estimation. The problem difficulty is well-calibrated for 64-sample batches with 8 groups.
- **Most compute is wasted in early epochs.** The 80% invalid-proposal rate at Step 0 represents the biggest efficiency bottleneck. Any intervention that raises initial correctness — better prompts, few-shot examples, or warm-started weights — would have an outsized impact on total time-to-SOTA.
- **Task selection matters enormously.** CP26 learned 3x faster than AC1 in correctness improvement, 4.6x vs. 1.9x in format rate, and reached its target while AC1 did not. For credit-constrained researchers, choosing problems with shorter evaluation times and smoother solution spaces is critical.

