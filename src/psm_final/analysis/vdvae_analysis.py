"""RSA / encoding analyzer for the pretrained VDVAE arm.

Mirrors `beta_vae_analysis.py` so the two generative arms are directly comparable:
same base class, same shared stimuli, same crop. The embedding is the concatenated
posterior means of VDVAE's coarsest latent groups -- the global / scene-level end of
its hierarchy, and the closest analogue to the beta-VAE's single 512-d latent.

See `psm_final.models.vdvae` for why VDVAE replaces the beta-VAE and how the
resolution cut maps onto embedding size.
"""

from pathlib import Path

import torch

from psm_final.analysis.model import ModelAnalysisBase
from psm_final.models.vdvae import (
    TOP_LATENT_RESOLUTION,
    checkpoint_path,
    embedding_dim,
    load_vdvae,
    n_top_groups,
    preprocess,
    top_latents,
)


class VDVAEAnalysis(ModelAnalysisBase):
    # Unlike the beta-VAE arm there are no local training runs to sweep: VDVAE is a
    # single released checkpoint, cached at <checkpoints_root>/results/vdvae/. A single
    # arm at TOP_LATENT_RESOLUTION (the res<=4/8/16 layer sweep was turned off
    # 2026-07-16 -- res<=4 only; see psm_final.models.vdvae).
    @classmethod
    def discover(cls, *, triple_n_path, checkpoints_root, device=None):
        """Return the single pretrained VDVAE, or nothing if it isn't downloaded.

        Deliberately does NOT auto-download: the checkpoint is ~500 MB and `discover`
        runs on every RSA sweep. When it's absent the runner already reports the arm
        as having no checkpoints; fetch it with
        ``python -m psm_final.models.vdvae``.
        """
        path = checkpoint_path(checkpoints_root)
        if not path.exists():
            return []
        label = (f"VDVAE res≤{TOP_LATENT_RESOLUTION} "
                 f"z={embedding_dim(TOP_LATENT_RESOLUTION)}")
        return [(label, lambda: cls(triple_n_path=triple_n_path,
                                    model_path=str(path), device=device))]

    def __init__(self, triple_n_path, model_path=None, device=None,
                 max_resolution=TOP_LATENT_RESOLUTION):
        super().__init__(triple_n_path)

        self.device = torch.device(
            device or ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.max_resolution = max_resolution

        if model_path is None:                    # notebook convenience
            model_path = checkpoint_path(Path(__file__).resolve().parents[3])
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"VDVAE checkpoint not found at {model_path}. Download it with:\n"
                f"    python -m psm_final.models.vdvae"
            )

        self.model = load_vdvae(model_path, device=self.device)
        self.n_groups = n_top_groups(max_resolution)
        self.embedding_dim = embedding_dim(max_resolution)

    def embedding(self, image):
        batch = preprocess(image, device=self.device)
        latents = top_latents(self.model, batch, max_resolution=self.max_resolution)
        return latents.squeeze(0).cpu().numpy()   # (embedding_dim,) for this image
