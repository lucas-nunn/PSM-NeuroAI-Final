# VDVAE arm (pretrained hierarchical VAE)

Status: implemented, tested, wired into the RSA runner. Legacy schema-2 encoding
runs were active on 2026-07-16, but no schema-3 area-label + noise-normalized result
has been reviewed or accepted yet.
Added 2026-07-15 on branch `encoding`. Runs a SINGLE res≤4 arm: a res≤4/8/16 layer
sweep was added 2026-07-16 and turned off the same day (user chose res≤4 only) — see
`## The cut`.

**CONFLICT / STATUS CORRECTION:** the earlier statement that encoding had "not yet
run" became stale once the parallel schema-2 jobs were launched. Those already-running
Python processes keep their old imported code and are not valid schema-3 normalization
runs; they were not interrupted by the 2026-07-16 implementation change.

## Why it exists

Replaces the β-VAE as the project's generative arm. The β-VAE
(`models/beta_vae.py`) cannot represent MS COCO: 4 conv layers at 64px through a
single `Linear` to a 512-d latent. **The limit is capacity and latent structure —
not the VAE objective, and not resolution.** VDVAE runs at the *same* 64px but
spreads ~1.45M latent dims over 75 stochastic groups (coarse 1x1 → fine 64x64) and
reconstructs natural scenes near-losslessly.

Keeping the arm a *real* KL-regularised VAE is deliberate. The arm exists to test
Higgins et al. 2021 (Nat. Commun. s41467-021-26751-5): an unsupervised
*disentangling* objective yields IT-like representations. Swapping in SD-VAE
(`AutoencoderKL` kl-f8, KL weight ~1e-6 — an autoencoder in all but name) would
reconstruct better but delete the hypothesis. See `## Alternatives considered`.

## Layout

| Path | Role |
|---|---|
| `src/psm_final/vendor/vdvae/` | inference-only copy of openai/vdvae (`vae.py`, `vae_helpers.py`) + README of patches |
| `src/psm_final/models/vdvae.py` | hps, checkpoint download, preprocess, `top_latents` |
| `src/psm_final/analysis/vdvae_analysis.py` | `VDVAEAnalysis(ModelAnalysisBase)` |
| `tests/test_vdvae.py` | 15 tests (8 architecture-only, 7 need the checkpoint) |
| `results/vdvae/imagenet64-iter-1600000-model-ema.th` | checkpoint, 500,977,841 bytes (gitignored) |

Get the checkpoint: `python -m psm_final.models.vdvae` (or `psm-vdvae`). ~500 MB.
`discover()` returns `[]` when it's absent — it never auto-downloads inside a sweep.

## Latent hierarchy (imagenet64, zdim=16, 75 groups)

`dec_blocks = "1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12"`.
One decoder block == one latent group. Strictly coarse → fine, so **a resolution cut
is a depth cut** (this is what `n_top_groups` relies on).

| max_resolution | groups | dims | what survives |
|---|---|---|---|
| 1 | 2 | 32 | nothing usable |
| 4 | 6 | 1,056 | global layout + colour only |
| 8 | 14 | 9,248 | object shape/pose appear |
| 16 | 30 | 74,784 | identity clear |
| 32 | 62 | 599,072 | near-perfect |
| 64 | 75 | 1,451,040 | lossless |

Cross-check: the first **31** groups sum to **91,168** dims — exactly the figure
Brain-Diffuser publishes for this checkpoint. `tests/test_vdvae.py` asserts it; it
validates our block parsing / zdim / group ordering against an independent user of
the same model.

## GOTCHAS (all cost real debugging time)

1. **The hierarchy is autoregressive — taking `qm` is NOT enough.** Each block
   derives `qm` from decoder state `x`, which shallower blocks built via
   `x = x + z_fn(z)` using their *sampled* `z`. So `qm` is deterministic only for
   group 0; every deeper group inherits upstream sampling noise. Measured: **70% of
   `res≤4` embedding dims changed between two identical calls.** Fixed by the
   vendored `deterministic=True` path (propagates posterior means through the whole
   chain). `test_sampled_pass_is_stochastic_beyond_the_first_group` characterises it.
2. **Input is NHWC**, not NCHW — the encoder permutes `(0,3,1,2)` itself. Feeding
   NCHW silently "works" on square images while treating channels as spatial rows.
3. **Normalisation is `(x_uint8 - 115.92961967) / 69.37404`**, NOT the `[0,1]`
   scaling the β-VAE arm uses. From openai/vdvae `data.py`, dataset `imagenet64`.
4. **Checkpoint URL host is `very-deep-vaes-assets`**, not `very-deep-vaes`. The
   latter 404s. (openai/vdvae's own README has the wrong one.)
5. `hps.py` is not vendored; `IMAGENET64_HPS` is declared by hand. It's validated by
   `load_vdvae`'s `strict=True` load — a typo there fails loudly, not silently.
6. `parse_layer_string` lives in `vae.py`, not `vae_helpers.py`.

## The cut: res≤4 only (sweep turned off 2026-07-16)

`TOP_LATENT_RESOLUTION = 4` — 6/75 groups, 1,056-d, the global / scene-level end of the
hierarchy and ≈ the β-VAE's 512-d scale. `VDVAEAnalysis.discover()` returns a SINGLE arm
at this cut, labelled `VDVAE res≤4`.

Decision history (both same day):
- The 2026-07-16 res≤4/8/16 layer sweep (`SWEEP_RESOLUTIONS`, one arm per cut) was
  **removed** later 2026-07-16 — user: "turn off the layer sweep … just do res≤4".
- That sweep had itself superseded the original 2026-07-15 "res≤4, don't change it"
  decision. Net effect: back to a single res≤4 arm, which is where it started.

The cut is still parameterised (`max_resolution` on `top_latents`/`VDVAEAnalysis`,
`--max-resolution` on `main()`), so a deeper cut is a one-liner if ever wanted, but the
runner ships res≤4 only. Because the stack is coarse→fine, a deeper cut would pull in
finer/local detail and push the arm *toward EVC*, not IT.

Counter-evidence on record (not acted on): in the reconstruction check res≤4 holds layout
and colour but not object identity (a stop sign is a featureless red ellipse); res≤8 is
the first cut where shape appears. Caveat — those panels draw the remaining groups from
the prior, so some mushiness is the decoder having nothing to work with rather than the
embedding being empty; the latents may carry more than the picture implies.

## Expected result (prediction, not yet measured)

Brain-Diffuser uses this checkpoint as its explicitly **low-level** branch (shape,
texture, layout) and needs CLIP-conditioned diffusion for semantics. So expect the
VDVAE arm to track **EVC > IT**. A strong IT correlation should be treated as
suspicious rather than celebrated. Compare against [rsa-cross-model-result].

## Alternatives considered

- **SD-VAE (`AutoencoderKL` kl-f8)** — reconstructs COCO near-perfectly, but KL weight
  ~1e-6 and a 4x64x64 *spatial* latent: it preserves local texture rather than
  compressing semantically, so its RDM ≈ a pixel/Gabor RDM. Not a stronger β-VAE; a
  different animal. Still the best *low-level anchor* if a second arm is ever wanted —
  the VDVAE-vs-SD-VAE contrast isolates hierarchy from reconstruction fidelity
  (Takagi & Nishimoto CVPR 2023 map SD's z from early visual cortex).
- **DiffAE** — semantic latent, but pretrained checkpoints are FFHQ/LSUN, not scenes.
- **VA-VAE** — latents aligned to DINOv2; circular, since DINOv2 would be doing the
  representational work.
