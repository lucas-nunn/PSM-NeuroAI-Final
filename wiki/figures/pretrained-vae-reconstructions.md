# Pretrained-VAE reconstruction figures

Added 2026-07-17 on branch `encoding`. Committed `e1560b1`.

## What / where

`scripts/reconstruct_pretrained_vaes.py` reconstructs a fixed batch of the shared
stimuli through the two **pretrained** VAE arms and saves originals-vs-reconstruction
grids in the SAME two-row layout the beta-VAE training loop uses
(`models/beta_vae.py:190 save_reconstruction` — originals on top row,
reconstructions below, `make_grid(nrow=n)` → `save_image`).

Outputs (tracked; `figures/` is not gitignored, `results/` is):
- `figures/recon_sd_vae.png`      (512px, ~5.5 MB)
- `figures/recon_vdvae.png`       (64px, full-hierarchy, near-lossless)
- `figures/recon_vdvae_res4.png`  (64px, res≤4 embedding only — see below)

Run: `uv run python scripts/reconstruct_pretrained_vaes.py`
(`--which {all,sd,vdvae,vdvae-top}`, `--top-temperature` for the res≤4 variant.)

## Inputs

First 8 **digit-named** shared stimuli `StimuliNNN/0001..0008.bmp`. The digit-only
filter matters: the dir also holds 72 `MFOB*.bmp` (1072 total, 1000 numbered). This
mirrors `ModelAnalysisBase.features()` (model.py:96, `^\d+$` filter), so the grid
shows the same images the encoding/RSA arms embed.

## Per-model reconstruction paths

- **SD-VAE** (`AutoencoderKL`, `stabilityai/sd-vae-ft-mse`): its analyzer transform
  (512×512, ToTensor) → `x*2-1` → `encode(x).latent_dist.mean` → `decode(z).sample` →
  rescale `/2+0.5`. Uses the posterior MEAN (matches the embedding the encoding arm
  uses). Per-pixel MSE ≈ 0.0007 — faithful, with the mild detail-softening expected
  from a 48× spatial compressor. A genuine, informative reconstruction.
- **VDVAE** (imagenet64): `preprocess` (NHWC, imagenet64 stats) →
  `forward_get_latents(deterministic=True)` → decode ALL 75 groups from their
  posterior means via `forward_samples_set_latents`. 64px, MSE ≈ 0.0000.

## CAVEAT — VDVAE full reconstruction is near-lossless and is NOT the arm's embedding

The full-hierarchy reconstruction looks almost identical to the original (MSE≈0). That
is correct, not a bug: at 64px the full latent stack (~1.45M dims across 75 groups)
vastly over-covers the 64×64×3 = 12 288 pixels, so VDVAE reconstructs near-losslessly.

But the encoding/RSA arm only uses the **res≤4** truncation (top 6 groups, 1 056-d;
see `models/vdvae.py TOP_LATENT_RESOLUTION`). So `recon_vdvae.png` shows what the
*model* can do, not what the *embedding* preserves.

`recon_vdvae_res4.png` fixes that: `reconstruct_vdvae_top` keeps ONLY the res≤4
posterior means and lets `forward_samples_set_latents` fill the deeper ~69 groups from
the prior (`zip_longest` pads with `lvs=None`). Temperature enters as
`pv + log(t)` → σ·t (`sample_uncond`), so `--top-temperature 1e-6` collapses the deeper
groups to their prior MEAN — a deterministic "single most likely completion given the
coarse latents." Result (MSE≈0.030 vs ≈0.000 full): global colour + coarse layout
survive (ocean blue with a horizon, green field, golden coffee, orange food) while all
object detail is gone. This is the visual confirmation that res≤4 is the global /
scene-level end of the hierarchy (matches `wiki/models/vdvae.md`). Raise
`--top-temperature` toward 1.0 for stochastic natural-looking completions instead.

By contrast the beta-VAE recon and the SD-VAE recon both decode from the same latent
the analysis arm uses, so those two are directly "the same as beta-vae training";
VDVAE's full recon is the one with the truncation asymmetry.

See also `wiki/models/vdvae.md`, `wiki/operations/chunked-encoding-runs.md`.
