"""
vdvae.py

VDVAE (Very Deep VAE, Child ICLR 2021) -- the pretrained hierarchical-VAE arm.

Why this model. The project's own beta-VAE (`beta_vae.py`) cannot represent MS COCO:
four conv layers at 64px funnelled through a single Linear into a 512-d latent. The
limit is capacity and latent *structure*, not the VAE objective and not resolution --
VDVAE runs at the same 64px but spreads ~100k latent dims across a stack of 75
stochastic groups, coarse (1x1) to fine (64x64), and reconstructs natural scenes
cleanly. It is therefore a like-for-like replacement that keeps the arm a real
KL-regularised VAE, preserving the hypothesis the arm exists to test (Higgins et al.
2021, Nat. Commun., https://www.nature.com/articles/s41467-021-26751-5: a
disentangling objective yields IT-like representations).

Precedent on this exact data: Brain-Diffuser (Ozcelik & VanRullen 2023,
https://arxiv.org/abs/2303.05334) regresses NSD fMRI onto this same imagenet64
checkpoint's latents. Algonauts 2023 *is* NSD, and Triple-N stimuli carry an NSD
crosswalk, so the stimulus alignment is already in place.

Two upstream conventions that are easy to get wrong, both handled by `preprocess`:
  * the encoder takes **NHWC** input (it permutes to NCHW itself), not NCHW;
  * images are normalised as ``(x_uint8 + SHIFT) * SCALE`` with imagenet64's dataset
    statistics -- NOT the [0, 1] scaling the beta-VAE arm uses.
"""

from pathlib import Path

import numpy as np
import torch

from psm_final.vendor.vdvae.vae import VAE, parse_layer_string

# --------------------------------------------------------------------------- #
# Checkpoint
# --------------------------------------------------------------------------- #
# The released imagenet64 model: 125M params, 1.6M iters (~2.5 weeks on 32 V100s).
# The EMA weights are the ones upstream evaluates with, so we use those.
CHECKPOINT_URL = (
    "https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/"
    "vdvae-assets-2/imagenet64-iter-1600000-model-ema.th"
)
CHECKPOINT_NAME = "imagenet64-iter-1600000-model-ema.th"
# Lives under results/ (gitignored, like the beta-VAE checkpoints) so the RSA
# runner's --checkpoints-root default finds it with no extra configuration.
CHECKPOINT_SUBDIR = Path("results") / "vdvae"

# --------------------------------------------------------------------------- #
# Hyperparameters -- openai/vdvae hps.py, `i64` (+ parser defaults it inherits)
# --------------------------------------------------------------------------- #
# Declared here rather than vendoring hps.py (which is an argparse harness). A typo
# here cannot pass silently: `load_vdvae` loads the released checkpoint with
# strict=True, so any mismatch in width/zdim/blocks fails loudly at load time.
IMAGENET64_HPS = dict(
    dataset="imagenet64",
    image_size=64,
    image_channels=3,
    width=512,
    zdim=16,
    custom_width_str="",
    bottleneck_multiple=0.25,
    no_bias_above=64,
    num_mixtures=10,
    dec_blocks="1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12",
    enc_blocks="64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5",
)

# openai/vdvae data.py, set_up_data() for dataset == 'imagenet64'.
SHIFT = -115.92961967
SCALE = 1.0 / 69.37404

# Keep latent groups at spatial resolution <= this. VDVAE's stack runs coarse ->
# fine, so a resolution cut IS a depth cut: the 1x1 and 4x4 groups are the global /
# scene-level end of the hierarchy, and 64x64 groups carry local texture.
#
# Cumulative cost of each cut (zdim=16, dims = zdim * res^2 per group):
#
#     max_resolution  groups  embedding dims
#            1            2              32
#            4            6           1,056   <- default
#            8           14           9,248
#           16           30          74,784
#           32           62         599,072
#           64           75       1,451,168
#
# Default 4 keeps the embedding (1,056-d) the same order as the beta-VAE's 512-d
# latent, so the two arms stay comparable. For reference, Brain-Diffuser cuts at 31
# groups / 91,168 dims -- the first 30 groups (res <= 16) plus the first 32x32 group.
TOP_LATENT_RESOLUTION = 4


class Hyperparams(dict):
    """Attribute-access dict. VDVAE's modules read config as ``H.width`` etc.;
    upstream uses the identical shim in hps.py."""

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError as exc:
            raise AttributeError(attr) from exc

    def __setattr__(self, attr, value):
        self[attr] = value


def latent_resolutions(dec_blocks=None):
    """Spatial resolution of each latent group, in decoder order (coarse -> fine).

    One VDVAE latent group == one decoder block, so the resolutions are derived from
    ``dec_blocks`` rather than hardcoded. For imagenet64 this is 75 groups running
    1,1,4,4,4,4,8,... up to 64.
    """
    dec_blocks = IMAGENET64_HPS["dec_blocks"] if dec_blocks is None else dec_blocks
    return [res for res, _mixin in parse_layer_string(dec_blocks)]


def latent_group_sizes(hps=None):
    """Flattened dimensionality (zdim * res^2) of each latent group, decoder order."""
    hps = IMAGENET64_HPS if hps is None else hps
    return [hps["zdim"] * res * res for res in latent_resolutions(hps["dec_blocks"])]


def n_top_groups(max_resolution=TOP_LATENT_RESOLUTION, hps=None):
    """How many leading latent groups sit at resolution <= ``max_resolution``.

    Relies on VDVAE's stack being monotonically coarse -> fine, so the groups at or
    below a resolution are exactly a prefix of the hierarchy.
    """
    resolutions = latent_resolutions((hps or IMAGENET64_HPS)["dec_blocks"])
    return sum(1 for res in resolutions if res <= max_resolution)


def embedding_dim(max_resolution=TOP_LATENT_RESOLUTION, hps=None):
    """Length of the vector `top_latents` returns for one image."""
    hps = IMAGENET64_HPS if hps is None else hps
    return sum(latent_group_sizes(hps)[: n_top_groups(max_resolution, hps)])


def checkpoint_path(root):
    """Where the checkpoint is cached, given a checkpoints root (repo root)."""
    return Path(root) / CHECKPOINT_SUBDIR / CHECKPOINT_NAME


def download_checkpoint(root, url=CHECKPOINT_URL):
    """Fetch the pretrained imagenet64 checkpoint (~500 MB) if not already cached.

    Returns the local path. Downloading is never triggered implicitly by the RSA
    runner -- `VDVAEAnalysis.discover` just skips the arm when the file is absent --
    so a sweep can't silently pull half a gigabyte. Run this module as a script to
    fetch it: ``python -m psm_final.models.vdvae``.
    """
    dest = checkpoint_path(root)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.hub.download_url_to_file(url, str(dest), progress=True)
    return dest


def build_vdvae(hps=None):
    """Instantiate the VDVAE architecture (random weights)."""
    return VAE(Hyperparams(**(IMAGENET64_HPS if hps is None else hps)))


def load_vdvae(path, device=None, hps=None):
    """Build the architecture and load the released weights into it, in eval mode.

    ``strict=True`` is load-bearing: it is what validates IMAGENET64_HPS against the
    real checkpoint, so a wrong width/zdim/block string fails here instead of quietly
    producing a differently-shaped embedding.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_vdvae(hps)
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def preprocess(image, image_size=None, device=None):
    """PIL image -> normalised NHWC float tensor (1, S, S, 3) ready for the encoder.

    Aspect-preserving resize + centre crop, mirroring the beta-VAE arm's transform so
    the two models see identically framed stimuli and any RSA difference is about the
    model, not the crop. Normalisation and the NHWC layout follow VDVAE's own
    conventions (see module docstring).
    """
    from PIL import Image
    from torchvision import transforms

    image_size = IMAGENET64_HPS["image_size"] if image_size is None else image_size
    # Triple-N stimuli are .bmp and may be grayscale; the encoder wants 3 channels.
    image = image.convert("RGB")
    resize = transforms.Compose([
        transforms.Resize(image_size, interpolation=Image.BICUBIC),
        transforms.CenterCrop(image_size),
    ])
    arr = np.asarray(resize(image), dtype=np.float32)          # (S, S, 3), 0..255
    arr = (arr + SHIFT) * SCALE
    tensor = torch.from_numpy(arr).unsqueeze(0)                # (1, S, S, 3), NHWC
    return tensor if device is None else tensor.to(device)


@torch.no_grad()
def top_latents(model, batch, max_resolution=TOP_LATENT_RESOLUTION):
    """Posterior means of the coarsest latent groups, flattened to (N, D).

    Takes the posterior mean q(z|x) rather than a sample, mirroring the beta-VAE arm
    (which embeds `mu`, not a draw) so the two are comparable.

    `deterministic=True` matters and is easy to miss: VDVAE's hierarchy is
    autoregressive, so group k's qm is computed from a decoder state built out of the
    z's of groups < k. Reading qm off a *sampled* pass would still be stochastic for
    every group but the first -- roughly 70% of the default embedding's dims moved
    between calls before this was fixed. Propagating the mean makes the whole path
    reproducible, so one image always gives one embedding.
    """
    stats = model.forward_get_latents(batch, deterministic=True)
    keep = stats[: n_top_groups(max_resolution)]
    return torch.cat([s["qm"].flatten(start_dim=1) for s in keep], dim=1)


def main():
    """``python -m psm_final.models.vdvae`` -- fetch the checkpoint and self-check."""
    import argparse

    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    parser.add_argument("--checkpoints-root", default=str(repo_root))
    parser.add_argument("--max-resolution", type=int, default=TOP_LATENT_RESOLUTION)
    args = parser.parse_args()

    path = download_checkpoint(args.checkpoints_root)
    print(f"checkpoint: {path} ({path.stat().st_size / 1e6:.0f} MB)")

    groups = latent_resolutions()
    print(f"latent groups: {len(groups)} (resolutions {groups[0]}..{groups[-1]})")
    print(f"cut at res<={args.max_resolution}: "
          f"{n_top_groups(args.max_resolution)} groups, "
          f"{embedding_dim(args.max_resolution)} dims")

    model = load_vdvae(path)
    print(f"loaded OK on {next(model.parameters()).device} "
          f"({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)")


if __name__ == "__main__":
    main()
