# Running encoding in resumable chunks + a final plot-only pass

Added 2026-07-17 on branch `encoding`. Pattern used to run the slow **lasso**
encoding sweep as several short CLI invocations instead of one monolithic process.

## Why chunk

Lasso encoding is `Lasso(alpha, max_iter=10_000)` coordinate descent
(`analysis/encoding.py:143`), fit per alpha × inner × outer fold × subject × ROI ×
voxel-target. It is **CPU-bound and single-core** (no `n_jobs`), so it is far slower
per model than Ridge's closed form (Ridge ~2–6 min/model; lasso can be many minutes
each). Chunking gives natural checkpoint boundaries, resumability if a chunk dies, and
separate invocations you can watch.

## The mechanism that makes chunking safe: per-model checkpoints + `--resume`

`runner.run_encoding` writes, per model, a `<output_id>_encoding_<reg>.done.json`
manifest plus the two result CSVs (`_algonauts` / `_triple_n_area_label`) under
`results/encoding/`. With `--resume`, a model is **reused only if** the manifest,
both CSVs, the model fingerprint, AND the full config all match
(`_read_encoding_checkpoint`, runner.py:525). `--resume` does NOT mean "skip if
missing" — a not-yet-done model is computed normally.

Key consequences:

- **Chunk by MODEL, never by ROI.** `config['rois']` (and alphas, folds, subjects,
  `triple_n_min_reliability`, data/impl/stimulus hashes) are part of the checkpoint
  identity. Running `--rois early` then `--rois midventral` produces DIFFERENT configs
  → the final all-ROI run won't resume them. Checkpoints are per-model, so the model
  axis is the only safe chunk axis. Hold every other flag identical across all chunks.
- Each run rewrites `encoding_<reg>_all.csv` from only the models in *that*
  invocation (and deletes the run.json + plots at start, runner.py:709). That combined
  CSV is transient; the source of truth is the per-model checkpoints. The final pass
  reassembles the complete CSV + plots from all checkpoints.

## `--models` selection gotcha

`filter_model_specs` (runner.py:193) matches **case-insensitive substring / glob /
exact**. When `--models` is passed, `main()` BYPASSES the `encoding_default` filter,
so an opt-in model (SD-VAE `encoding_default=False`, Pixel baseline) can sneak into a
chunk if your substring hits it. Verify substrings against the label list. The default
set (no `--models`) is exactly: 5 βVAEs + CNN + ResNet50 + VDVAE (+ PCA 50, always
skipped for CV leakage, `encoding_cv_safe=False`). SD-VAE and Pixel are excluded there.

Substrings that select cleanly: `βVAE` → 5 βVAEs; `CNN` → 1; `ResNet50` → 1;
`VDVAE` → 1. None of these hit SD-VAE/Pixel.

## The final "plot-only" pass

Run with `--resume` and **no** `--models` (default set) and **no** `--no-plots`. Every
default model is already checkpointed → all resumed (zero heavy compute) → 
`save_encoding_results` regenerates the heatmap PNGs and the combined CSV. Effectively
plot-only.

## zsh word-split gotcha (bit me 2026-07-17)

The Bash tool runs **zsh**, which does NOT word-split an unquoted `$C`. `uv run
psm-rsa $C ...` passes the whole flag string as ONE argv → `unrecognized arguments`.
Use a zsh array and `"${C[@]}"`, or inline the flags literally.

## Exact commands used (2026-07-17 lasso run)

```zsh
C=(--method encoding --regression lasso --outer-folds 2 --inner-folds 2 \
   --rois early midventral midlateral midparietal ventral lateral parietal \
   --alphas 0.1 1 10 100 1000)
uv run psm-rsa "${C[@]}" --resume --no-plots --models βVAE            # chunk 1: 5 βVAEs
uv run psm-rsa "${C[@]}" --resume --no-plots --models CNN ResNet50   # chunk 2
uv run psm-rsa "${C[@]}" --resume --no-plots --models VDVAE          # chunk 3
uv run psm-rsa "${C[@]}" --resume                                    # chunk 4: plot-only
```

Ridge counterpart (9 alphas 1e-4..1e4, same 2×2 folds) finished cleanly earlier the
same morning; see `results/encoding/encoding_ridge_*`.
