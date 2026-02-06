# Week 4 Report Draft 1

## 1. Problem Statement

### What we are optimizing
We study how to improve AI-agent search loops on strictly verifiable optimization problems.  
Our first benchmark is **2D circle packing** in a bounded square: given a fixed number of circles with equal radius, we want to maximize feasible packing quality under non-overlap and boundary constraints.

### Why this matters
Many scientific and engineering tasks are hard to solve directly but easy to verify once a candidate is proposed.  
If an AI system can reliably improve candidate quality in these settings, this could generalize to broader classes of verifiable discovery problems.

### Success metrics
- Primary: best valid objective score found within a fixed compute budget.
- Secondary: improvement curve over iterations (best-so-far vs iteration).
- Efficiency: valid proposal rate and time-to-best-score.

### Constraints
- Every candidate must pass strict geometric validity checks.
- We use fixed model calls and bounded iteration budgets.
- Week 4 scope focuses on baseline iterative search (no RL updates yet).

### Data requirements
- Synthetic problem instances for circle packing (number of circles, box size, radius).
- Logged candidates and verifier outputs per iteration.
- Compute/runtime logs for reproducibility.

### Failure modes and risks
- Generated candidates may be mostly invalid (low signal to improve).
- Prompt/refinement may plateau quickly without better feedback shaping.
- Claimed gains may come from prompt tuning noise rather than robust method changes.

## 2. Technical Approach

### Mathematical framing
Let each candidate be circle centers \(x_i = (x_i^{(1)}, x_i^{(2)})\), \(i=1,\dots,n\), in square \([0,1]^2\), with fixed radius \(r\).  
Constraints:
- Boundary: \(r \le x_i^{(1)}, x_i^{(2)} \le 1-r\)
- Non-overlap: \(\|x_i - x_j\|_2 \ge 2r\), \(\forall i \ne j\)

Week 4 objective (baseline): maximize a verifier-defined score under these constraints (equivalently minimize violation penalty, then maximize packing quality among valid candidates).

### Algorithmic design (Week 4 baseline)
1. Generate candidate pool (random + mutation/refinement from top candidates).
2. Run deterministic verifier to compute validity and score.
3. Keep top-k valid candidates.
4. Repeat for T iterations and track best-so-far.

### PyTorch/implementation strategy
- Build verifier and scoring code in Python (optionally tensorized with PyTorch for faster batch checks).
- Keep a reproducible random seed and fixed hyperparameters.
- Log per-iteration metrics: best score, valid fraction, average score, runtime.

### Validation plan
- Sanity checks on tiny instances where feasibility is easy to inspect.
- Unit-style checks for boundary and overlap constraints.
- Reproducibility check across multiple seeds.

### Resource requirements
- Local CPU is sufficient for Week 4 baseline.
- No model fine-tuning required this week.
- Estimated runtime: minutes per experiment for small/medium instances.

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

## Elevator Pitch (2 Sentences)
We are optimizing the feedback loop between LLM-based agent search and strictly verifiable optimization tasks, because this can improve AI performance on scientific-style problems where checking is easy but discovering strong solutions is hard.  
Our approach starts with a frozen-model iterative generate-verify-select loop on circle packing, then extends to test-time reinforcement learning to increase the probability of producing high-quality valid solutions.

## Self-Critique (Week 4)

### Biggest Technical Risk
The project could stall if we choose tasks where the verifier is strict but the proposal mechanism cannot produce enough valid candidates for learning signal.

### Most Important Assumption
We assume test-time RL adaptation can improve solution quality/sample efficiency on a fixed verifiable task without destabilizing performance.

### Proof-of-Life Experiment (1 Week)
Run a no-RL baseline on circle packing that shows measurable best-score improvement across iterations under a fixed compute budget.
