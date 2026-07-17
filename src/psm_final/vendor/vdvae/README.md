# Vendored: OpenAI VDVAE (inference only)

Source: https://github.com/openai/vdvae (branch `main`), from the paper
*Very Deep VAEs Generalize Autoregressive Models and Can Outperform Them on Images*
(Child, ICLR 2021 — https://arxiv.org/abs/2011.10650).

Vendored rather than pip-installed because upstream ships as a training research
repo, not a package: it has no `setup.py`, and its `train.py` / `utils.py` pull in
`mpi4py` + `torch.distributed`. Only `vae.py` and `vae_helpers.py` are needed to
run the encoder, and those import nothing beyond torch/numpy — so we take those
two files and leave the training stack behind.

## Files

| File | Upstream | Modified |
|---|---|---|
| `vae.py` | `vae.py` | yes — see below |
| `vae_helpers.py` | `vae_helpers.py` | no (byte-identical) |

## Local patches to `vae.py`

Both are marked inline with `# PATCHED:`.

1. **Import made package-relative.** `from vae_helpers import ...` →
   `from .vae_helpers import ...`. Upstream assumes flat sys.path.

2. **`DecBlock` exposes the posterior `q(z|x)`.** Upstream's
   `forward(..., get_latents=True)` returns only the sample `z`, drawn via
   `draw_gaussian_diag_samples(qm, qv)`. A sample is stochastic, so an embedding
   built from it changes on every call and the resulting RDM is not reproducible.
   `sample()` now additionally returns `qm, qv`, and `forward()` puts them in the
   stats dict, so callers can use the deterministic posterior mean `qm`.
   The same change is made for the same reason in Brain-Diffuser
   (https://github.com/ozcelikfu/brain-diffuser).

   This does not alter the forward computation — `z` is still the sample, and the
   decoder still consumes `z`. Only the reported stats gain two keys.

3. **Optional `deterministic` path** through `DecBlock.sample` / `DecBlock.forward` /
   `Decoder.forward` / `VAE.forward_get_latents` (default `False` = upstream
   behaviour, bit-for-bit).

   Patch 2 alone is *not* enough to get a reproducible embedding, which is worth
   spelling out because it is silent and counterintuitive. VDVAE's hierarchy is
   autoregressive: `DecBlock.sample` derives `qm` from the decoder state `x`, and
   shallower blocks build `x` up via `x = x + z_fn(z)` using their *sampled* `z`. So
   reading `qm` off an ordinary pass is deterministic only for the very first latent
   group; every deeper group inherits the upstream sampling noise. Measured on the
   default res≤4 cut, ~70% of embedding dims changed between two identical calls.

   With `deterministic=True`, each block sets `z = qm`, so the conditioning path
   carries posterior means and the returned latents are a pure function of the input.
   This mirrors what the project's beta-VAE arm does (it embeds `mu`, never a draw).

`hps.py` is deliberately NOT vendored; the imagenet64 hyperparameters are declared
explicitly in `psm_final/models/vdvae.py` and validated by a `strict=True`
state-dict load against the released checkpoint.
