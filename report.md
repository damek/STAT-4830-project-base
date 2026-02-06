# Week 4 Report Draft 1

## 1. Problem Statement

### What we are optimizing

We study how test-time training on a large language model during inference can improve AI-driven search on strictly verifiable optimization problems.

Large language models are very good at generating solutions but relatively bad at verifying them. The generation-verification gap has been approached in many different ways, including training an LLM or reward model to assess the reward of a given solution (see [https://arxiv.org/abs/2505.03335](https://arxiv.org/abs/2505.03335) among others). Here we take a different approach: reinforce an LLM at test time to progressively improve on a *strictly verifiable* problem, with the reward signal shaped by deterministic evaluation of its own prior attempts.

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

**Advantage estimation and RL training -** The (prompt, generation, reward) tuples are assembled into a training batch. An **entropic adaptive beta** advantage estimator computes per-sample advantages, adjusting the KL penalty dynamically to maintain a target entropy level. This prevents the model from collapsing to a single strategy too early. The LoRA adapter weights are updated via the Tinker training API using these advantages, with a learning rate of `4e-5`.

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
- **Training signal quality:** We monitor `frac_mixed` (fraction of groups with both successes and failures) — a value near 1.0 indicates the RL training has good contrastive signal. A value near 0.0 (all-good or all-bad groups) would indicate the reward is uninformative.
- **Convergence tracking:** Performance curves on W&B should show monotonic improvement in best-so-far score. KL divergence from the base model should grow gradually, not spike.
- **Reproducibility:** Fixed random seeds, deterministic evaluation, and full metric logging ensure runs can be reproduced and compared.

### Resource requirements

- **Compute:** 256-core CPU server with 1.5 TB RAM for local evaluation and Ray parallelization. No local GPU required. All LLM compute is handled by the Tinker API.
- **API access:** Tinker API key for model inference and training. Weights & Biases API key for experiment logging.
- **Estimated runtime per epoch:** ~60 minutes for AC1 (dominated by 1000s evaluation timeouts), ~15–30 minutes for CP26 (shorter 305s timeouts). Total for 5 epochs: ~5–6 hours for AC1, ~2–3 hours for CP26.
- **Concurrent execution:** Both problems run simultaneously on the same machine, each using ~64 CPU cores via Ray.

## 3. Initial Results (Placeholder to Fill After Running Notebook)

### Current status

We implemented the baseline loop and verifier pipeline and are running first experiments on toy circle-packing instances.

### What we will report here

- At least one full run with improvement over iterations.
- Best candidate found and validity confirmation.
- Valid proposal rate and runtime profile.
- One failure case and debugging notes.

### Known limitations

- No reinforcement learning component yet.
- Search strategy is simple and may converge slowly.
- Feedback signal currently limited to scalar score + validity flag.

## 4. Next Steps

### Immediate improvements (Week 5 direction)

- Improve proposal strategy (guided mutations based on constraint violations).
- Add richer verifier feedback to help candidate repair.
- Compare multiple search temperatures/pool sizes via ablation.

### RL extension plan (after baseline is stable)

- Introduce test-time adaptation policy to bias generation toward high-scoring valid candidates.
- Compare against frozen baseline under equal compute budget.
- Measure sample efficiency gains and robustness across seeds.

### Open questions

- Which verifiable benchmark(s) beyond circle packing should be included next?
- What reward design best balances validity and objective quality?
- How to prevent overfitting to one prompt format while still improving performance?

## Self-Critique (Week 4)

### Biggest Technical Risk

The project could stall if we choose tasks where the verifier is strict but the proposal mechanism cannot produce enough valid candidates for learning signal.

### Most Important Assumption

We assume test-time RL adaptation can improve solution quality/sample efficiency on a fixed verifiable task without destabilizing performance.

### Proof-of-Life Experiment (1 Week)

Run a no-RL baseline on circle packing that shows measurable best-score improvement across iterations under a fixed compute budget.