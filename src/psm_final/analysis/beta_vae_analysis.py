
import re
from pathlib import Path

import torch
import torchvision.transforms as transforms

from psm_final.analysis.model import ModelAnalysisBase
from psm_final.models.beta_vae import BetaVAE


class BetaVAEAnalysis(ModelAnalysisBase):
    # Trained checkpoints live at ``<checkpoints_root>/results/beta_vae/<run>/vae.pth``
    # (matching where the training scripts save under ``results/``); the RSA runner
    # globs this to discover every model.
    CHECKPOINT_GLOB = "results/beta_vae/*/vae.pth"

    @classmethod
    def discover(cls, *, triple_n_path, checkpoints_root, device=None):
        """Find every trained β-VAE checkpoint under ``checkpoints_root`` and return
        one ``(label, factory)`` per model, sorted by (beta, latent).

        Run directories are named like ``latent_512_beta_2.0_epochs_50_seed_42``; the
        latent size and beta are parsed from that name for a short, unique label.
        """
        specs = []
        for ckpt in Path(checkpoints_root).glob(cls.CHECKPOINT_GLOB):
            name = ckpt.parent.name
            beta_match = re.search(r"beta_([0-9.]+)", name)
            latent_match = re.search(r"latent_(\d+)", name)
            beta = float(beta_match.group(1)) if beta_match else None
            latent = int(latent_match.group(1)) if latent_match else None
            label = "βVAE"
            if latent is not None:
                label += f" z={latent}"
            if beta is not None:
                label += f" β={beta:g}"
            specs.append((beta if beta is not None else float("inf"),
                          latent if latent is not None else 0, label, str(ckpt)))
        specs.sort(key=lambda spec: (spec[0], spec[1]))
        # `path=path` binds the current checkpoint into each factory (avoids the
        # classic late-binding closure bug where all factories share the last path).
        return [
            (label, lambda path=path: cls(triple_n_path=triple_n_path,
                                          model_path=path, device=device))
            for _, _, label, path in specs
        ]

    def __init__(self, triple_n_path, model_path, latent_dim=None, device=None):
        super().__init__(triple_n_path)

        self.device = torch.device(
            device or ('cuda' if torch.cuda.is_available() else 'cpu')
        )

        # Training saves a state_dict (torch.save(vae.state_dict(), ...)), not the
        # whole model, so rebuild the architecture and load the weights into it.
        state_dict = torch.load(model_path, map_location=self.device)
        if latent_dim is None:
            # The final encoder Linear emits latent_dim*2 units (mu and log_var),
            # so recover latent_dim straight from the checkpoint.
            latent_dim = state_dict['encoder.9.weight'].shape[0] // 2

        self.model = BetaVAE(latent_dim=latent_dim)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Match the training transform exactly (same resize + [0, 1] scaling).
        self.transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
        ])

    def embedding(self, image):
        # The encoder expects 3 channels; Triple-N stimuli are .bmp and may be
        # grayscale, so force RGB to match training.
        image_tensor = self.transform(image.convert('RGB')).unsqueeze(0).to(self.device)

        with torch.no_grad():
            mu, log_var = self.model.encode(image_tensor)

        return mu.squeeze(0).cpu().numpy()  # (latent_dim,) embedding for this image
