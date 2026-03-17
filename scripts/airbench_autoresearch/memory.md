# Memory

## Current Best
- baseline seed reset

## Recent Accepted Changes
- none yet

## Recent Rejections / Failures
- none yet

## Repeated Failure Modes
- Preserve dtype consistency between normalized inputs and half-precision conv weights/biases.
- Preserve CLI flags, especially --verbose.
- Avoid CUDAGraph-unsafe repeated compiled inference patterns in TTA.
