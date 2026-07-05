import argparse
import csv
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image


class BetaVAE(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, latent_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(512, 4, 4)),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=-1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, mu, log_var

    def loss_function(self, recon_x, x, mu, log_var, beta=1.0):
        """Per-image beta-VAE loss (averaged over the batch).

        Normalising by batch size keeps the loss scale -- and hence the meaning
        of ``beta`` and the learning rate -- independent of ``batch_size``. This
        is the canonical beta-VAE convention: ``beta=1`` is a vanilla VAE, and
        ``beta>1`` puts more pressure on the latent (more structured/disentangled
        code, blurrier reconstructions).

        Returns ``(loss, BCE, KLD, kld_per_dim)`` where ``kld_per_dim`` is the
        batch-averaged KL contribution of each latent dimension (nats/image),
        used to count active units.
        """
        batch_size = x.size(0)
        # Reconstruction: summed over pixels, averaged over the batch -> nats/image.
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') / batch_size
        # Per-latent-dim KL, averaged over the batch -> nats/image for each dim.
        kld_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).mean(dim=0)
        KLD = kld_per_dim.sum()
        return BCE + beta * KLD, BCE, KLD, kld_per_dim


# Spatial size the BetaVAE encoder/decoder are hard-wired to (see BetaVAE).
IMAGE_SIZE = 64


def _image_transform(image_size=IMAGE_SIZE):
    """Aspect-preserving resize + centre crop, kept as uint8 [0, 255] (CHW)."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.PILToTensor(),
    ])


def build_image_cache(root_dir, cache_path, image_size=IMAGE_SIZE):
    """Decode + resize every image once into a single uint8 tensor on disk.

    Training otherwise re-reads and re-decodes every JPEG on every epoch, which
    is the I/O + CPU bottleneck. Pre-transforming to one uint8 tensor (~1.5 GB
    for COCO unlabeled2017 at 64px) lets the training loop read pre-sized images
    straight from RAM instead. This is a one-time cost the first time a dataset
    is used.
    """
    root_dir = Path(root_dir)
    image_paths = sorted(root_dir.glob('*.jpg'))
    if not image_paths:
        raise FileNotFoundError(f'no .jpg images found under {root_dir}')

    transform = _image_transform(image_size)
    cache = torch.empty((len(image_paths), 3, image_size, image_size), dtype=torch.uint8)
    for i, path in enumerate(image_paths):
        with Image.open(path) as image:
            cache[i] = transform(image.convert('RGB'))
        if (i + 1) % 5000 == 0:
            print(f'  pre-transformed {i + 1}/{len(image_paths)} images')

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    size_gb = cache.element_size() * cache.nelement() / 1e9
    print(f'wrote image cache {cache_path} (shape {tuple(cache.shape)}, {size_gb:.2f} GB)')
    return cache


class COCODataset(Dataset):
    """COCO images pre-transformed to a cached uint8 tensor held in RAM.

    On first use every image is decoded, resized and cropped once and written to
    ``cache_dir``; later runs load that cache and skip JPEG decoding entirely,
    removing the per-epoch I/O bottleneck. Delete the cache file if the image set
    or ``image_size`` changes (the filename encodes both).
    """
    def __init__(self, root_dir, cache_dir='./cache', image_size=IMAGE_SIZE):
        cache_path = Path(cache_dir) / f'{Path(root_dir).name}_{image_size}px.pt'
        if cache_path.exists():
            print(f'loading cached images from {cache_path}')
            self.images = torch.load(cache_path)
        else:
            print(f'no cache at {cache_path}; pre-transforming images once...')
            self.images = build_image_cache(root_dir, cache_path, image_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # uint8 [0, 255] -> float [0, 1], matching the previous ToTensor() scaling.
        return self.images[idx].float().div_(255.0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_dim', type=int, default=512, help='Dimensionality of the latent space')
    parser.add_argument('--beta', type=float, default=1.0, help='Weight of the KL divergence term in the loss function')
    parser.add_argument('--coco_root', type=str, required=True, help='Path to the COCO dataset images')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--cache_dir', type=str, default='./cache', help='Directory for the pre-transformed image tensor cache')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader worker processes (with the cache, a few is plenty)')
    return parser.parse_args()


def set_seed(seed):
    """Seed every RNG so a run is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic (slower, but repeatable convolutions).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """Give each DataLoader worker a deterministic, distinct seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@torch.no_grad()
def save_reconstruction(vae, images, epoch, out_dir, n=8):
    """Save the fixed batch and its reconstruction side by side for this epoch."""
    was_training = vae.training
    vae.eval()
    recon_x, _, _ = vae(images[:n])
    # Two rows: originals on top, reconstructions below.
    comparison = torch.cat([images[:n], recon_x])
    grid = make_grid(comparison, nrow=n)
    save_image(grid, out_dir / f'recon_epoch_{epoch + 1:03d}.png')
    if was_training:
        vae.train()


def main():
    args = parse_args()
    set_seed(args.seed)

    vae = BetaVAE(latent_dim=args.latent_dim)
    coco = COCODataset(root_dir=args.coco_root, cache_dir=args.cache_dir)
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    # A dedicated generator + seed_worker make shuffling and workers reproducible.
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    dataloader = DataLoader(
        coco, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
        pin_memory=True, worker_init_fn=seed_worker, generator=generator,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae.to(device)

    out_dir = Path(f'./results/beta_vae/latent_{args.latent_dim}_beta_{args.beta}_epochs_{args.num_epochs}_seed_{args.seed}')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fix one batch up front so every epoch's reconstruction shows the same images.
    fixed_batch = next(iter(dataloader)).to(device)

    # A latent dim counts as "active" when its mean KL exceeds this many nats;
    # near-zero KL means the dim carries no information (collapsed).
    active_kl_threshold = 1e-2

    # Log per-epoch loss components to CSV so the run can be inspected/plotted
    # later. Flushed every epoch, so it stays readable even if the run is killed.
    metrics_path = out_dir / 'metrics.csv'
    with open(metrics_path, 'w', newline='') as metrics_file:
        metrics_writer = csv.writer(metrics_file)
        metrics_writer.writerow(['epoch', 'loss', 'bce', 'kld', 'active_units', 'total_units'])

        for epoch in range(args.num_epochs):
            running_loss = running_bce = running_kld = 0.0
            kld_per_dim_sum = torch.zeros(args.latent_dim, device=device)
            n_batches = 0
            for batch in dataloader:
                batch = batch.to(device)
                recon_x, mu, log_var = vae(batch)
                loss, bce, kld, kld_per_dim = vae.loss_function(recon_x, batch, mu, log_var, beta=args.beta)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Accumulate per-image quantities so the epoch log is an average, not
                # a single noisy last-batch reading.
                running_loss += loss.item()
                running_bce += bce.item()
                running_kld += kld.item()
                kld_per_dim_sum += kld_per_dim.detach()
                n_batches += 1
            avg_loss = running_loss / n_batches
            avg_bce = running_bce / n_batches
            avg_kld = running_kld / n_batches
            kld_per_dim_mean = kld_per_dim_sum / n_batches
            active_units = int((kld_per_dim_mean > active_kl_threshold).sum().item())
            print(f'Epoch [{epoch + 1}/{args.num_epochs}], Loss: {avg_loss:.2f}, '
                  f'BCE: {avg_bce:.2f}, KLD: {avg_kld:.2f}, '
                  f'active_units: {active_units}/{args.latent_dim}')
            metrics_writer.writerow(
                [epoch + 1, f'{avg_loss:.4f}', f'{avg_bce:.4f}', f'{avg_kld:.4f}',
                 active_units, args.latent_dim]
            )
            metrics_file.flush()
            save_reconstruction(vae, fixed_batch, epoch, out_dir)

    torch.save(vae.state_dict(), out_dir / 'vae.pth')


if __name__ == "__main__":
    main()