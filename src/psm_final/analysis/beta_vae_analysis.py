
import torch
import torchvision.transforms as transforms

from psm_final.analysis.model import ModelAnalysisBase
from psm_final.models.beta_vae import BetaVAE


class BetaVAEAnalysis(ModelAnalysisBase):
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
