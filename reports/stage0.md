# Stage 0: Running SmolVLA on a CPU

**Outcome: go.** SmolVLA and LIBERO run on an Apple M1 Pro inside arm64 Linux Docker, with closed-loop success in line with published figures. About 2 s per forward pass makes the open-loop work feasible.

## Setup

LeRobot 0.7.0, Python 3.12, PyTorch 2.15, MuJoCo 3.8.1 and robosuite 1.4.0 with OSMesa software rendering. The image is defined in `docker/Dockerfile`. Run `scripts/stage0_run.sh` inside the container to reproduce these numbers.

## Two settings decide feasibility

The checkpoint loads as `bfloat16`, which this CPU runs very slowly. oneDNN's matrix path is also about 2.5× slower than BLAS here.

| Configuration | Seconds per forward pass |
|---|---|
| `bfloat16` (checkpoint default) | 47 |
| `float32` with oneDNN | 6 |
| `float32` without oneDNN | 2.1–2.5 |

Converting to `float32` changed actions by 0.0019 on average in normalized units. Changing the noise seed changed them by 0.076.

## Closed-loop reproduction

The first 5 tasks of each suite, with Wilson 95% intervals:

| Suite | Replan every 10 steps | Replan every step |
|---|---|---|
| LIBERO-Spatial | 8/10 (49–94%) | 4/5 (38–96%) |
| LIBERO-Object | 10/10 (72–100%) | 5/5 (57–100%) |
| Wall-clock per episode | 60–74 s | 374–458 s |

The SmolVLA paper (arXiv 2506.01844, Table 13) reports 89% on Spatial and 94% on Object at both replanning intervals. Our intervals contain those values.

## Timing

Measured on 256×256 renders with synthetic model inputs, on a machine also in light use. Each layout ran only 6 passes, so differences under about 20% are noise.

| Component | Time |
|---|---|
| Environment setup | 9.3 s |
| Environment step with two camera observations | 0.24 s |
| One camera render | 0.10 s |
| Image encoding, each camera | 0.60 s |
| VLM prefill | 0.31 s |
| Expert denoising step (10 per pass) | 0.083 s |

| Workers × threads | Seconds per pass, combined |
|---|---|
| 1×6 | 2.45 |
| 1×8 | 2.09 |
| 2×3 | 2.14 |
| 2×4 | 1.80 |
| 2×5 | 2.82 |
| 3×3 | 2.06 |
| 4×2 | 1.66 |

At 1.7–2.1 s per pass, one suite's 22,000 passes would take 10–13 hours.

## Model anatomy

```
Agentview, wrist ─> SigLIP (12 blocks) ─> connector ─> 64 tokens each ┐
Instruction ─────> word embeddings ───────────────────────────────────┼─> VLM (32 layers, frozen)
Robot state ─────> state projection ──────────────────────────────────┘            │
                                                                     per-layer K/V cache
                                                                                   │
Noise ─> Action expert (32 layers, 10 denoising steps; reads the cache at every layer) ─> 50 actions
```

Details are in `docs/model_notes.md`. The checkpoint has 604.9M parameters and all 32 VLM layers.

## What this shows and what it doesn't

This shows that the pipeline is correct enough to study: the model solves tasks at rates consistent with its paper, and CPU throughput fits the planned open-loop budget. It doesn't establish a precise success rate, because the sample covers 15 episodes per suite on half of each suite's tasks, and intervals are wide. It also doesn't prove that replanning every 10 steps is equivalent to every step. Our data is too small for that, so the choice rests mainly on the paper's result and the 6–7× speedup. Timings come from one laptop and shouldn't be read as portable figures.

**Decision:** replan every 10 steps in closed-loop runs. Stage 5's 720 episodes would then take about 13 hours.
