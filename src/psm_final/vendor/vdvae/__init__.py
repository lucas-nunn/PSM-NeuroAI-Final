"""Inference-only vendored copy of OpenAI's VDVAE (https://github.com/openai/vdvae).

See README.md in this directory for provenance and the list of local patches.
"""

from .vae import VAE

__all__ = ["VAE"]
