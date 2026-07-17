"""Reconstruct a fixed batch of shared stimuli through the two *pretrained* VAE arms
and save originals-vs-reconstruction grids.

Mirrors how the beta-VAE training loop visualises progress
(`psm_final.models.beta_vae.save_reconstruction`): a single image with the originals
on the top row and their reconstructions directly below, laid out with
``torchvision.utils.make_grid`` / ``save_image``. The beta-VAE is trained in-house so
it emits one grid per epoch; these two arms are frozen released checkpoints, so there
is a single reconstruction per model.

The two arms decode differently, so each is handled with its own analyzer's exact
preprocessing (so the "original" row is literally what the model was fed):

* SD-VAE (`AutoencoderKL`, ``stabilityai/sd-vae-ft-mse``): 512x512 RGB scaled to
  [-1, 1] -> 4x64x64 latent. Reconstruction is ``decode(encode(x).mean)`` (posterior
  MEAN, matching the embedding the encoding/RSA arm uses), rescaled back to [0, 1].
* VDVAE (imagenet64): 64px NHWC input. Reconstruction runs the full hierarchy with
  every group pinned to its posterior mean (``deterministic=True``), i.e. all 75
  latent groups, not just the res<=4 groups the encoding embedding truncates to -- so
  this shows the model's true reconstruction quality, not the coarse embedding.

Usage:
    uv run python scripts/reconstruct_pretrained_vaes.py
"""

import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from psm_final.analysis.autoencoderKL import AutoKL
from psm_final.analysis.vdvae_analysis import VDVAEAnalysis
from psm_final.models.vdvae import (
    TOP_LATENT_RESOLUTION,
    checkpoint_path,
    preprocess,
)

REPO = Path(__file__).resolve().parents[1]


def resolve_triple_n_dir(explicit=None):
    """Triple-N root: --triple-n-dir, then $TRIPLE_N_DIR, then the repo .env line."""
    if explicit:
        return explicit
    env = os.environ.get("TRIPLE_N_DIR")
    if env:
        return env
    dotenv = REPO / ".env"
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            line = line.strip()
            if line.startswith("TRIPLE_N_DIR="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("could not resolve TRIPLE_N_DIR (pass --triple-n-dir)")


def load_stimuli(triple_n_dir, n):
    """First ``n`` shared stimuli (StimuliNNN/0001.bmp ...) as RGB PIL images.

    Matches ModelAnalysisBase.features(): keep only the digit-named stimuli
    (0001..1000, the shared set the models embed) and sort them numerically, so
    images[i] is the same stimulus the encoding/RSA arms see at position i.
    """
    stim_dir = Path(triple_n_dir) / "others" / "StimuliNNN"
    numeric = [p for p in stim_dir.glob("*.bmp") if p.stem.isdigit()]
    paths = sorted(numeric, key=lambda p: int(p.stem))[:n]
    if len(paths) < n:
        raise SystemExit(f"found only {len(paths)} stimuli under {stim_dir}")
    print(f"[stimuli] {', '.join(p.name for p in paths)}")
    return [Image.open(p).convert("RGB") for p in paths]


@torch.no_grad()
def reconstruct_sd_vae(analyzer, images):
    """Return (originals, reconstructions) in [0, 1], NCHW, at 512x512."""
    dev = analyzer.device
    orig = torch.stack([analyzer.transform(im) for im in images])       # [0,1] (N,3,512,512)
    recon = torch.empty_like(orig)
    for i in range(orig.shape[0]):                                       # per image: keeps VRAM low
        x = (orig[i:i + 1].to(dev) * 2.0) - 1.0                          # [0,1] -> [-1,1]
        z = analyzer.model.encode(x).latent_dist.mean                   # posterior mean
        out = analyzer.model.decode(z).sample                           # [-1,1]
        recon[i] = ((out / 2.0) + 0.5).clamp(0, 1).squeeze(0).cpu()
    return orig, recon


@torch.no_grad()
def reconstruct_vdvae(analyzer, images):
    """Return (originals, reconstructions) in [0, 1], NCHW, at 64x64.

    Full deterministic reconstruction: encode with posterior means, then decode every
    latent group from those means (`forward_samples_set_latents`).
    """
    dev = analyzer.device
    batch = torch.cat([preprocess(im, device=dev) for im in images], dim=0)  # (N,64,64,3) NHWC
    stats = analyzer.model.forward_get_latents(batch, deterministic=True)
    latents = [s["z"] for s in stats]                                   # posterior means (all groups)
    recon_u8 = analyzer.model.forward_samples_set_latents(batch.shape[0], latents)  # (N,64,64,3) uint8
    recon = torch.from_numpy(recon_u8).float().div(255.0).permute(0, 3, 1, 2)       # (N,3,64,64) [0,1]

    # Originals at the SAME 64px framing the encoder saw (resize short edge + centre crop).
    disp = transforms.Compose([
        transforms.Resize(64, interpolation=Image.BICUBIC),
        transforms.CenterCrop(64),
        transforms.PILToTensor(),
    ])
    orig = torch.stack([disp(im).float().div(255.0) for im in images])
    return orig, recon


def save_grid(orig, recon, path, n):
    """Originals on top row, reconstructions below -- beta-VAE save_reconstruction layout."""
    comparison = torch.cat([orig[:n], recon[:n]])
    grid = make_grid(comparison, nrow=n)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, path)
    mse = torch.mean((orig[:n] - recon[:n]) ** 2).item()
    print(f"[wrote] {path}  (per-pixel recon MSE={mse:.4f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--triple-n-dir", default=None)
    ap.add_argument("--n", type=int, default=8, help="images in the grid (default 8)")
    ap.add_argument("--out-dir", default=str(REPO / "figures"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    triple_n_dir = resolve_triple_n_dir(args.triple_n_dir)
    out_dir = Path(args.out_dir)
    images = load_stimuli(triple_n_dir, args.n)

    print("[SD-VAE] loading stabilityai/sd-vae-ft-mse ...")
    sd = AutoKL(triple_n_path=triple_n_dir, device=args.device)
    orig, recon = reconstruct_sd_vae(sd, images)
    save_grid(orig, recon, out_dir / "recon_sd_vae.png", args.n)
    del sd
    torch.cuda.empty_cache()

    print(f"[VDVAE] loading checkpoint (res<={TOP_LATENT_RESOLUTION} arm, full-hierarchy decode) ...")
    vd = VDVAEAnalysis(
        triple_n_path=triple_n_dir,
        model_path=str(checkpoint_path(REPO)),
        device=args.device,
    )
    orig, recon = reconstruct_vdvae(vd, images)
    save_grid(orig, recon, out_dir / "recon_vdvae.png", args.n)


if __name__ == "__main__":
    main()
