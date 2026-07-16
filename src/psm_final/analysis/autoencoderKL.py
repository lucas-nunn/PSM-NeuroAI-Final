# write the class

import numpy as np
import torch
import torchvision.transforms as transforms

# L: Hit "pip install diffusers transformers" in the Terminal.
# L: Hit "uv add diffusers" in Terminal to instell it here?
from diffusers import AutoencoderKL

from psm_final.analysis.model import ModelAnalysisBase


class AutoKL(ModelAnalysisBase):
    def __init__(self, triple_n_path, device=None):
        # This line lets the base class find the Triple-N stimulus images later.
        super().__init__(triple_n_path)

        # Use the GPU if there is one, otherwise the CPU.
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # (1) DOWNLOAD your pretrained model from the internet.
        #     torch.hub downloads the weights the first time and caches them.
        #     Swap this line for whatever AE / VAE / diffusion model you chose.
        self.model = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
        self.model.to(self.device).eval()

        # (2) HOOKS - Stable Diffusion's VAE does not need a hook.

        # (3) How to turn a PIL image into the tensor your model expects.
        #     Match whatever preprocessing your chosen model was trained with.
        # L: SD's VAE expects 512x512 images scaled to [-1, 1].
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

    def embedding(self, image):
        # The base class hands you one PIL image at a time. Convert to RGB in case
        # a stimulus is grayscale, preprocess it, and add a fake "batch" dimension.
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        # Scale [0, 1] -> [-1, 1]
        x = x * 2.0 - 1.0

        with torch.no_grad():          # don't waste memory tracking gradients
            # Extract the latent representation from the encoder
            moments = self.model.encode(x).latent_dist
            latents = moments.sample()  # We define 'latents' here!

        # Return a flat 1-D numpy vector: this is the embedding for this one image.
        # This takes the 4x64x64 latent tensor and flattens it to a clean 16,384-dimensional vector!
        return latents.squeeze().cpu().to(torch.float32).numpy().ravel()