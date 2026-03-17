# AirBench Autoresearch Program

You are editing exactly one mutable file: `candidate.py`.
Within that file, you may edit exactly one allowed section at a time.

The loop is simple:
1. Establish the baseline with the current file.
2. Make one experimental edit.
3. Run the experiment.
4. If it improved, keep it and advance.
5. If it did not improve, discard it and revert.

## Objective
Optimize `candidate.py` to train CIFAR-10 on a single `NVIDIA A100-40GB` GPU to at least `94.00%` TTA accuracy as quickly as possible.

Ranking rule:
1. A candidate that reaches `94.00%` beats every candidate that does not.
2. Among candidates that reach `94.00%`, lower `mean_time_seconds` is better.
3. Among candidates that miss `94.00%`, higher `mean_accuracy` is better.

## Hard Constraints
The file must remain a complete standalone Python script.

It must accept these CLI flags:
- `--data-dir`
- `--trials`
- `--warmup-trials`
- `--target-accuracy`
- `--json-only`
- `--preflight`
- `--verbose`

Its final stdout line must be JSON with at least:
- `mean_accuracy`
- `mean_time_seconds`
- `trials`

The runtime environment is fixed:
- Python 3.11
- `torch==2.4.1`
- `torchvision==0.19.1`
- one `A100-40GB`
- CIFAR-10 cached at `/vol/cifar10`

## Fixed Surface
Treat these parts of the file as fixed infrastructure:
- CLI parsing and supported flags
- final JSON output structure
- trial / warmup control flow
- benchmark timing semantics
- whitening front-end dimensions and initialization invariants

Do not casually rewrite the wrapper, the main entrypoint, or the reporting contract.

## Allowed Editable Sections
The loop will only accept one section replacement at a time. The allowed sections are:
- `optimizer_core`: `zeropower_via_newtonschulz5` and `Muon`
- `model_core`: `ConvGroup` and `CifarNet`
- `eval_core`: `infer` and `evaluate`
- `training_loop`: `run_single_trial` and `run_preflight`

All other sections are preserved mechanically by the loop.

## Good Experiment Families
Prefer experiments drawn from a clear family such as:
- optimizer design and hyperparameters
- learning-rate schedule and total training budget
- batch sizes and throughput-related settings
- precision and compile policy
- augmentation and TTA implementation
- model width, depth, or block structure
- data preprocessing and caching

Choose one family at a time and make one coherent experiment within it.

## Editing Guidance
- Simpler is better.
- `candidate.py` is the research surface, but not every line is equally likely to matter. Focus on the training system and evaluation path, not the wrapper.
- Make one coherent experimental revision at a time. A good experiment may require coordinated changes in multiple parts of the file.
- Do not make scattered unrelated tweaks. The edit should reflect a clear technical hypothesis about why the program will become faster or more accurate.
- Preserve the AirBench-style benchmark semantics; do not fake metrics or skip training.
- Keep the program readable.
- A tiny gain that adds ugly complexity is usually not worth it.
- A simplification that preserves quality is valuable.

## Known Failure Modes To Avoid
- Float32 activations fed into half-precision compiled convolutions.
- Removing required CLI flags such as `--verbose`.
- Compiled TTA patterns that trigger CUDAGraph overwrite errors.
- Breaking the final JSON contract.
- Partial whitening/front-end rewrites that change tensor shapes without a coherent end-to-end redesign.

## First Run
- The baseline run is already established externally by the loop.
- Do not waste the first proposal by restating the current file.

## What To Optimize First
- If the incumbent is below 94%, prioritize small accuracy gains.
- If the incumbent is already above 94%, prioritize speed while staying above 94%.
- Robustness improvements are worthwhile if they reduce invalid proposals without adding much complexity.
