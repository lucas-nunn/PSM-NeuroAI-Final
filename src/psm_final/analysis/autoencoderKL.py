"""RSA / encoding analyzer for a Stable-Diffusion latent-diffusion VAE.

The generative "diffuser" arm: it embeds each stimulus with the encoder of the
KL-regularised autoencoder that Stable Diffusion diffuses in. Mirrors the other
generative arms (`beta_vae_analysis.py`, `vdvae_analysis.py`) so all three are
directly comparable -- same base class, same shared stimuli, same deterministic
posterior-mean embedding.

The checkpoint is ``stabilityai/sd-vae-ft-mse``: the standalone, MSE-fine-tuned
SD1.x VAE. It is a drop-in for the original ``runwayml/stable-diffusion-v1-5``
VAE (same 4x64x64 latent) but is still hosted on the Hub -- the runwayml repo was
removed in 2024 -- and downloads only the VAE rather than the whole pipeline.
"""

import numpy as np
import torch
import torchvision.transforms as transforms
from diffusers import AutoencoderKL

from psm_final.analysis.model import ModelAnalysisBase

# Standalone MSE-fine-tuned SD1.x VAE. 512x512 RGB in [-1, 1] -> 4x64x64 latent.
_HF_MODEL = "stabilityai/sd-vae-ft-mse"
_LATENT_DIM = 4 * 64 * 64  # 16,384


class AutoKL(ModelAnalysisBase):
    # Like VDVAE, this is a single released checkpoint rather than a set of local
    # training runs, so there is nothing to glob under ``checkpoints_root``.
    @classmethod
    def discover(cls, *, triple_n_path, checkpoints_root, device=None):
        """Return the single pretrained SD-VAE for the runner to compare.

        The ~335 MB checkpoint downloads (and is Hub-cached) the first time the
        factory is actually invoked, so ``discover`` itself stays cheap.
        """
        label = f"SD-VAE z={_LATENT_DIM}"
        return [(label, lambda: cls(triple_n_path=triple_n_path, device=device))]

    def __init__(self, triple_n_path, device=None):
        # Lets the base class find the Triple-N stimulus images later.
        super().__init__(triple_n_path)

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = AutoencoderKL.from_pretrained(_HF_MODEL)
        self.model.to(self.device).eval()

        # SD's VAE expects 512x512 images; scaled to [-1, 1] in `embedding`.
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

    def embedding(self, image):
        # Base class hands one PIL image at a time. Force RGB (some stimuli are
        # grayscale), preprocess, and add a batch dimension.
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        x = x * 2.0 - 1.0  # [0, 1] -> [-1, 1]

        with torch.no_grad():
            # Posterior MEAN, not a sample: a deterministic, reproducible embedding
            # (matching beta-VAE `mu` and VDVAE's means). Sampling would make the
            # RDM jitter run-to-run.
            latents = self.model.encode(x).latent_dist.mean

        # Flatten the 4x64x64 latent to a 16,384-d vector for this one image.
        return latents.squeeze(0).cpu().to(torch.float32).numpy().ravel()
