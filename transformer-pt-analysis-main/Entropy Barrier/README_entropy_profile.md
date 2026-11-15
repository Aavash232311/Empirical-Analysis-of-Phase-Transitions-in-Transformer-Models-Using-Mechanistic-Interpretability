# Entropy at fixed training loss

`entropy_profile.py` estimates the shape of **S(x*, y)**, where

- `x = ln(L_train)` (natural log of the mean training cross-entropy).
- `y = M_diag`, the diagonal fraction of non-DC Fourier power of class-centered logits on **all p² pairs**.
- The parameter measure is uniform Lebesgue measure in a finite box `[-B,B]^N`.

The existing `transformer.py` architecture and `_make_pair_split` are reused. Existing training and analysis scripts are unchanged. The old `wang_landau.py` resamples a finite, nonuniform perturbation cloud; it does not estimate this global parameter-volume measure.

## Start here

From a terminal:

```bash
cd '/home/avash/Documents/research/transformer-pt-analysis-main/Entropy Barrier'
PY=/home/avash/Documents/research/.venv/bin/python
$PY entropy_profile.py inspect
```

Inspect reports both checkpoints' actual training losses and diagonal masses. **Verify `--train-frac 0.3`, `--data-seed 0`, and `--n-heads 4` match how your checkpoints were trained.** These are the current repository defaults, but the old checkpoint format does not store them. Shape-compatible attention heads can still change the model function. No external datasets are required.

Run a short mechanics/pacing pilot:

```bash
$PY entropy_profile.py run --start mem --steps 100 --log-every 10 \
  --output entropy_runs/pilot
```

This will almost certainly report `incomplete_sampling`; it is not a barrier measurement. The default x* is the memorizing checkpoint's freshly measured log training loss. To choose raw CE instead, use `--loss-star 4e-7`; alternatively supply `--x-star -14.7318`. Do not specify both.

Two independent, opposite-phase runs at the same target and prior:

```bash
$PY entropy_profile.py run --start mem --seed 1 --loss-star 4e-7 \
  --half-width 0.1 --steps 100000 --production-steps 50000 \
  --output entropy_runs/mem_seed1
$PY entropy_profile.py run --start gen --seed 2 --loss-star 4e-7 \
  --half-width 0.1 --steps 100000 --production-steps 50000 \
  --output entropy_runs/gen_seed2
```

These are experiment budgets, **not promises of convergence**. At this model size, random walks may need substantially more time or fail to mix even after long runs. The program is a transparent baseline, not the PDF's gradient-driven MD implementation. It avoids saving a full weight vector at every step. The dominant work per move is evaluating the model on all p² pairs, batched with `--batch-size`; CUDA is selected automatically if available. Use `--device cpu` when needed. Try a smaller trained model first for method development.

## What is being estimated?

The hard shell is `|ln L_train - x*| <= delta`, with `delta = --half-width`:

```
g_delta(y) = integral_box 1[|x(theta)-x*| <= delta] delta_D(y-y(theta)) dtheta
S_delta(y) = ln g_delta(y)
```

As delta tends to zero, `S_delta(y) - ln(2*delta)` approaches `S(x*,y)` wherever the density is regular. At finite delta the result is a window-integrated profile, not an exact zero-width slice. The steep density gradient in x can affect its shape: repeat at delta = 0.1, 0.05, 0.025, aligning additive constants, and check the barrier stabilizes. `profile.csv` includes the mean x in each production bin to expose displacement toward a shell boundary.

The sampler changes randomly chosen model coordinates with a fixed-width Gaussian and reflects at the parameter-box boundary. Coordinate selection and reflection are state-independent and symmetric. Out-of-shell or out-of-range proposals are rejected. Inside the shell, acceptance is `min(1, exp(V[current] - V[proposed]))`. The occupied bin is updated even when a proposal is rejected. No training optimizer or weight decay acts during sampling.

The Wang--Landau modification factor is halved only after **every configured y bin** has enough visits and `min(H)/mean(H)` passes `--flatness`. Unvisited bins are never silently removed. Once the requested factor is reached, the bias is frozen, a burn-in is discarded, and production counts give:

```
S_j = V_j + ln(H_production,j) - ln(bin_width) + constant
```

The final profile is shifted so its largest measured value is zero. The estimator uses the original [Wang--Landau approach](https://arxiv.org/abs/cond-mat/0011174) and a frozen-bias histogram correction. Geometric factor reduction is a practical finite-accuracy schedule, not a guarantee of exact convergence. Independent seeds and a smaller final factor remain necessary.

### Matching the initial losses

By default, both endpoint checkpoints are brought to the same training CE by multiplying `out_proj.weight` and `out_proj.bias` by a positive scalar found by bisection. This is an **initialization operation only**, before sampling. All centered logits scale together, so M_diag is invariant. The operation requires strictly correct predictions on every training example and a target below `ln(vocab_size)`. If that condition fails, choose fitted checkpoints, or use `--no-match-loss` when the starting checkpoint already lies in the shell. The final starting loss is always evaluated again.

The shared box half-width is `--box-factor` (default 3) times the largest absolute weight across **both prepared endpoints**. Supply `--box-bound B` for an explicit shared prior. Both runs must use the same checkpoints, target, split, y range, precision and box. The files record checkpoint hashes, split hash, scaling factors and B. This is a bounded flat-prior volume, not the AdamW training distribution.

### Loss and spectrum details

Training CE is averaged only over the reconstructed training split, at the final '=' input position. It includes all p+1 output classes, matching training. CE is accumulated in float64 with `softplus(logsumexp(incorrect_logits - correct_logit))` to retain tiny positive losses. The model uses float32 by default; `--dtype float64` also promotes the model forward pass. The spectral diagnostic uses only the first p output classes, centers across classes, sums FFT power over classes, and excludes the DC mode explicitly. Using held-out-pair loss for x would answer a different question.

## Outputs and interpretation

Each output directory contains:

- `profile.png`: relative entropy, coverage, and conditional mean log loss.
- `profile.csv`: production entropy, adaptive bias, counts, and mean x. Unmeasured entropy entries are NaN, not interpolated.
- `summary.json`: coverage, acceptance, refinement history, box and split metadata, and candidate barrier values.
- `trace.csv`: thinned time trace for checking wandering and persistence.
- `sampler.pt`: current parameters, bias, counters and RNG state for resuming.
- `config.json`: exact command settings.

`--left-phase 0.10 0.30` and `--right-phase 0.90 0.995` define bands containing the phase peaks. They are starting choices for your current checkpoints, not universal phase definitions. The program finds each band's entropy maximum and the intervening minimum, then reports

```
Delta S_mem = S(y_mem_peak) - S(y_trough)
Delta S_gen = S(y_gen_peak) - S(y_trough)
```

These are directional barriers in **nats**. At fixed energy, `Delta F/(k_B*T) = Delta S` in these units. All bins between the peaks must have at least `--min-visits` production visits. Missing coverage gives no barrier number; a monotone measured profile reports `no_resolved_interior_trough`, not a zero global barrier. Peaks/troughs are read from unsmoothed bins, so noisy extrema can overestimate a barrier. Treat them as candidates until stable under binning and replica checks.

The overall status remains `incomplete_sampling` unless the WL schedule finished, every production bin has the minimum visits, and the trajectory completed at least two end-to-end round trips. Round trips count visits to the outermost configured bins across the whole run. Even a passing status only means `candidate_profile_requires_replica_and_window_checks`: flat histograms and correlated visit counts are not proof of ergodicity, independent samples, or a first-order transition.

For uncertainty, run at least three seeds from **each** endpoint, align profiles by subtracting the mean over the same well-sampled bins, compare endpoint groups, and report between-replica spread in the profile and each barrier. No within-run Poisson error bars are supplied: correlated trajectories would make them misleading. Repeat with narrower loss windows, finer y bins, smaller final ln(f), and different shared box bounds. A disconnected shell can trap the sampler permanently; this method cannot assert that an unexplored component has zero entropy.

## Tuning and resuming

- `--step-size`: absolute Gaussian standard deviation per selected weight. Lower it if most proposals leave the loss shell. Increase cautiously if accepted moves barely change y.
- `--block-size`: random coordinates changed per move (default 256; 0 means all). Every coordinate has a chance to move; this is not a fixed low-dimensional subspace.
- `--y-min`, `--y-max`, `--bins`: choose a range containing both phases and their intervening region. Defaults avoid exact y=0 and y=1. Do not shrink away an unvisited valley to manufacture convergence. If you change bins/range, start a new run.
- `--log-f-min`, `--flatness`, `--min-visits`: refinement controls. Flatness alone cannot detect hidden barriers.
- `--save-every`, `--log-every`: checkpoint and progress intervals.

Ctrl-C saves current state. Resume by repeating the original settings and increasing total budgets, for example:

```bash
$PY entropy_profile.py run --start mem --seed 1 --loss-star 4e-7 \
  --half-width 0.1 --steps 300000 --production-steps 50000 \
  --output entropy_runs/mem_seed1 --resume entropy_runs/mem_seed1/sampler.pt
```

`--steps` is a total adaptation budget, not additional steps. A completed WL schedule is never restarted; increase `--production-steps` to extend production. The original numerical settings are validated against the checkpoint. A nonempty output directory is protected unless resuming. GPU/CPU floating-point trajectories need not be identical.

Run the numerical checks with:

```bash
$PY -m unittest -v test_entropy_profile.py
```

The checks cover spectral invariance, tiny training losses, rejection bookkeeping, exact same-device RNG continuation, missing-support barrier handling, and reconstruction of an analytically known nonuniform entropy profile.
