import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class BetaVAE(nn.Module):
    def __init__(self, latent_dim=50):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, latent_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(256, 4, 4)),
            nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1),
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
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + beta * KLD    


class COCODataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.image_paths = list(self.root_dir.glob('*.jpg'))
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_dim', type=int, default=50, help='Dimensionality of the latent space')
    parser.add_argument('--beta', type=float, default=1.0, help='Weight of the KL divergence term in the loss function')
    parser.add_argument('--coco_root', type=str, required=True, help='Path to the COCO dataset images')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--save_path', type=str, default='vae.pth', help='Path to save the trained model')
    return parser.parse_args()


def main():
    args = parse_args()

    vae = BetaVAE(latent_dim=args.latent_dim)
    coco = COCODataset(root_dir=args.coco_root)
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)
    dataloader = DataLoader(coco, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae.to(device)

    for epoch in range(args.num_epochs):
        for batch in dataloader:
            batch = batch.to(device)
            recon_x, mu, log_var = vae(batch)
            loss = vae.loss_function(recon_x, batch, mu, log_var, beta=args.beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{args.num_epochs}], Loss: {loss.item():.4f}')

    torch.save(vae.state_dict(), args.save_path)


if __name__ == "__main__":
    main()