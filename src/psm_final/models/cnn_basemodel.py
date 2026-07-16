"""
cnn_basemodel.py

Self-contained CNN classifier for COCO images -- model, dataset, and CLI
all in one file, matching beta_vae.py's pattern (no separate dataset module).

Requires COCO's annotation file (not just the image zips) -- download
annotations_trainval2017.zip from https://cocodataset.org/#download,
unzip it, and point --annotation_path at annotations/instances_train2017.json.
Parsed with Python's built-in json module -- no extra annotation library needed.
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np 
import pandas as pd
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms


from psm_final.models.beta_vae import _image_transform, IMAGE_SIZE
from psm_final.helpers.training_testing import test, train_and_test  

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # First convolutional layer:
        # Input: 3 channels (RGB), Output: 32 feature maps
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)

        # Second convolutional layer:
        # Input: 32 feature maps, Output: 64 feature maps
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)

        # Third convolutional layer:
        # Input: 64 feature maps, Output: 128 feature maps
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)

        # Max pooling layer: reduces spatial size by a factor of 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Activation function
        self.relu = nn.ReLU()

        # Adaptive average pooling: reduces each feature map to size 1x1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 1 * 1, 256)
        # num_classes is set dynamically to match however many categories
        # exist in the annotation file -- see main().
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x, return_features=False):
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = self.relu(self.conv3(x))

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        features = self.relu(self.fc1(x))
        x = self.fc2(features)

        if return_features:
            return x, features
        return x

def load_shared_stimuli_coco_ids(crosswalk_path):
    crosswalk = pd.read_csv(crosswalk_path)
    scenes = crosswalk[crosswalk['kind'] == 'scene']
    return set(scenes['coco_id'].dropna().astype(int))

def build_coco_classification_cache(annotation_path, image_dir, cache_path, image_size=IMAGE_SIZE,
                                     exclude_coco_ids=None):
    """
    Turn COCO's multi-object annotations into single-label classification data,
    using every category found in the annotation file (no restriction/subset).

    Each image's label is the category of its LARGEST (by bounding-box area)
    annotated object -- a simple, deterministic way to collapse "multiple
    objects per image" into "one label per image". Images with no annotations
    at all are skipped.
    """
    image_dir = Path(image_dir)
    exclude_coco_ids = set(exclude_coco_ids) if exclude_coco_ids else set()

    # Loads the whole ~450MB file into memory once -- fine for a one-time
    # cache-building pass, and the result feeds straight into the loop below.
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)

    # "categories": [{"id": 1, "name": "person", "supercategory": "person"}, ...]
    # Use every category present, ordered by id, so label indices are stable
    # and predictable (index 0 = lowest category id, etc.).
    ordered_categories = sorted(coco_data['categories'], key=lambda c: c['id'])
    class_names = [cat['name'] for cat in ordered_categories]
    cat_id_to_label = {cat['id']: i for i, cat in enumerate(ordered_categories)}

    # "images": [{"id": 391895, "file_name": "000000391895.jpg", ...}, ...]
    image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}

    # "annotations": one entry per labeled OBJECT (many per image). Walk the
    # list once, tracking the largest-area object seen per image so far.
    best_area = {}
    best_label = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id in exclude_coco_ids:
            continue
        area = ann['area']
        if image_id not in best_area or area > best_area[image_id]:
            best_area[image_id] = area
            best_label[image_id] = cat_id_to_label[ann['category_id']]

    kept_image_ids = sorted(best_label.keys())
    kept_labels = [best_label[image_id] for image_id in kept_image_ids]

    if not kept_image_ids:
        raise RuntimeError('no images have any annotations in this file')

    print(f'{len(kept_image_ids)} images matched across {len(class_names)} categories '
          f'(out of {len(coco_data["images"])} total in the annotation file, '
          f'{len(exclude_coco_ids)} excluded)')

    transform = _image_transform(image_size)
    cache = torch.empty((len(kept_image_ids), 3, image_size, image_size), dtype=torch.uint8)
    for i, image_id in enumerate(kept_image_ids):
        file_name = image_id_to_filename[image_id]
        with Image.open(image_dir / file_name) as image:
            cache[i] = transform(image.convert('RGB'))
        if (i + 1) % 5000 == 0:
            print(f'  pre-transformed {i + 1}/{len(kept_image_ids)} images')

    labels = torch.tensor(kept_labels, dtype=torch.long)

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'images': cache, 'labels': labels, 'classes': class_names}, cache_path)
    size_gb = cache.element_size() * cache.nelement() / 1e9
    print(f'wrote image cache {cache_path} (shape {tuple(cache.shape)}, {size_gb:.2f} GB)')
    return cache, labels, class_names


class COCOClassificationDataset(Dataset):
    """Single-label COCO images, using every category in the annotation file,
    cached as a uint8 tensor for fast repeated epochs.

    Same load-cache-or-build-it pattern as beta_vae.py's COCODataset, but
    returns (image, label) pairs instead of just images. Delete the cache
    file if the image set or image_size changes.
    """
    def __init__(self, annotation_path, image_dir, cache_dir='./cache', image_size=IMAGE_SIZE,
                 max_images=None, exclude_coco_ids=None):
        tag = 'excl' if exclude_coco_ids else 'all'
        cache_path = Path(cache_dir) / f'{Path(image_dir).name}_{image_size}px_{tag}cls.pt'
        if cache_path.exists():
            print(f'loading cached images from {cache_path}')
            data = torch.load(cache_path)
            self.images, self.labels, self.classes = data['images'], data['labels'], data['classes']
        else:
            print(f'no cache at {cache_path}; building from COCO annotations...')
            self.images, self.labels, self.classes = build_coco_classification_cache(
                annotation_path, image_dir, cache_path, image_size, exclude_coco_ids=exclude_coco_ids,
            )

        # Optionally keep only a random subset to shrink the in-RAM footprint
        # (the full 224px cache is ~12.5 GB and both train+val are resident at
        # once). Fancy-indexing returns a fresh, smaller tensor, so reassigning
        # here drops the reference to the full tensor and frees that memory.
        # Deterministic (fixed seed) so the same subset is used across runs.
        if max_images is not None and 0 < max_images < len(self.images):
            n_total = len(self.images)
            keep = torch.randperm(n_total, generator=torch.Generator().manual_seed(0))[:max_images]
            self.images = self.images[keep].contiguous()
            self.labels = self.labels[keep].contiguous()
            print(f'subsampled to {len(self.images)} of {n_total} images (--max_train_images)')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # uint8 [0, 255] -> float [0, 1]
        image = self.images[idx].float().div_(255.0)
        label = self.labels[idx]
        return image, label


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_root', type=str, required=True,
                         help='Path to train2017/ folder of images')
    parser.add_argument('--annotation_path', type=str, required=True,
                         help='Path to instances_train2017.json')
    parser.add_argument('--val_coco_root', type=str, required=True,
                     help='Path to val2017/ folder of images')
    parser.add_argument('--val_annotation_path', type=str, required=True,
                        help='Path to instances_val2017.json')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                         help='Directory for the pre-transformed image tensor cache')
    parser.add_argument('--image_size', type=int, default=IMAGE_SIZE,
                         help='Square resolution (px) images are resized to before training. '
                              'Changing it builds a separate cache (the size is part of the cache filename).')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader worker processes')
    parser.add_argument('--max_train_images', type=int, default=0,
                         help='If >0, randomly subsample the training set to this many images '
                              '(deterministic). Shrinks the in-RAM cache; 0 = use all images.')
    parser.add_argument('--k_folds', type=int, default=5,
                     help='Number of CV folds to run on the training set before the final model')
    parser.add_argument('--exclude_crosswalk', type=str, default=None,
                         help='Path to triple_n_crosswalk.csv -- excludes its shared-1000 '
                              'RSA/encoding stimuli from training')
    return parser

def k_fold_indices(n_samples, k, seed=42):
    """Split n_samples indices into k roughly-equal, shuffled folds."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    return np.array_split(indices, k)

def parse_args():
    return build_parser().parse_args()

def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    exclude_coco_ids = (
        load_shared_stimuli_coco_ids(args.exclude_crosswalk) if args.exclude_crosswalk else None
    )
    if exclude_coco_ids:
        print(f'excluding {len(exclude_coco_ids)} shared-stimuli images from training')

    coco_train = COCOClassificationDataset(
        annotation_path=args.annotation_path,
        image_dir=args.coco_root,
        cache_dir=args.cache_dir,
        image_size=args.image_size,
        max_images=args.max_train_images or None,
        exclude_coco_ids=exclude_coco_ids,
    )

    coco_dataloader_train = DataLoader(
        coco_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    
    coco_test = COCOClassificationDataset(
        annotation_path=args.val_annotation_path,  
        image_dir=args.val_coco_root,             
        cache_dir=args.cache_dir,
        image_size=args.image_size,
    )
        
    coco_dataloader_test = DataLoader(
        coco_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    assert coco_train.classes == coco_test.classes, "class order mismatch between train and val!"

    folds = k_fold_indices(len(coco_train), args.k_folds, seed=args.seed)
    fold_val_losses, fold_val_accuracies = [], []

    for fold_idx in range(args.k_folds):
        print(f'\n=== CV fold {fold_idx + 1}/{args.k_folds} ===')

        val_indices = folds[fold_idx]
        train_indices = np.concatenate([folds[i] for i in range(args.k_folds) if i != fold_idx])

        fold_train_loader = DataLoader(
            Subset(coco_train, train_indices), batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers,
        )
        fold_val_loader = DataLoader(
            Subset(coco_train, val_indices), batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers,
        )

        fold_cnn = SimpleCNN(num_classes=len(coco_train.classes)).to(device)
        fold_criterion = nn.CrossEntropyLoss()
        fold_optimizer = torch.optim.Adam(fold_cnn.parameters(), lr=args.learning_rate)

        train_and_test(fold_cnn, fold_train_loader, fold_val_loader, args.num_epochs,
                       fold_criterion, fold_optimizer, device)
        val_loss, val_accuracy = test(fold_cnn, fold_val_loader, fold_criterion, device)
        fold_val_losses.append(val_loss)
        fold_val_accuracies.append(val_accuracy)

    print(f'\n=== CV summary ({args.k_folds} folds) ===')
    print(f'val loss     -- mean: {np.mean(fold_val_losses):.4f}  std: {np.std(fold_val_losses):.4f}')
    print(f'val accuracy -- mean: {np.mean(fold_val_accuracies):.2f}%  std: {np.std(fold_val_accuracies):.2f}%')

    cnn = SimpleCNN(num_classes=len(coco_train.classes))
    cnn = cnn.to(device)

    cnn_criterion = nn.CrossEntropyLoss()
    cnn_optimizer = torch.optim.Adam(cnn.parameters(), lr=args.learning_rate)
        
    history = train_and_test(cnn, coco_dataloader_train, coco_dataloader_test, args.num_epochs, cnn_criterion, cnn_optimizer, device)
    print(history)
    
    test(cnn, coco_dataloader_test, cnn_criterion, device)
    
    out_dir = Path('./results/cnn_basemodel')
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / 'train_history_simple_cnn.csv', history, delimiter=',', header='batch_loss', comments='')
    torch.save({
        'model_state_dict': cnn.state_dict(),
        'classes': coco_train.classes,
        'num_classes': len(coco_train.classes),
        'image_size': args.image_size,
    }, out_dir / 'cnn.pth')
    print(f'saved model to {out_dir / "cnn.pth"}')
    
if __name__ == '__main__':
    main()