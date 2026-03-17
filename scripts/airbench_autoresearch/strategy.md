# Strategy

## Objective
- Keep the incumbent above 94.00% TTA accuracy.
- Once above target, prioritize lower mean_time_seconds.

## Current Direction
- Prefer one coherent experiment family per batch, not unrelated tweaks.
- Focus on training-system changes within one allowed section at a time.

## Allowed Sections
- `optimizer_core`
- `model_core`
- `eval_core`
- `training_loop`

## Good Families To Explore
- Optimizer hyperparameters and parameter grouping
- Learning-rate schedule and total training budget
- Batch size and throughput settings
- Precision and compile policy
- Augmentation and TTA implementation
- Model width, depth, and block structure

## Avoid
- Breaking CLI flags or JSON output
- Casual rewrites of the main wrapper
- Partial whitening/front-end rewrites that do not preserve shape consistency
- Changing boolean toggle flags like `--json-only`, `--preflight`, or `--verbose` into value-taking arguments
- Touching sections outside the allowed list
