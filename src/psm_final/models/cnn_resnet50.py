"""
cnn_resnet50.py

ResNet50 classifier for COCO images, fine-tuned from ImageNet-pretrained
weights. Mirrors cnn_basemodel.py's structure (dataset class reused from
there, same train/val split, same train_and_test/test helpers, same
metadata-wrapped checkpoint format) so downstream code -- e.g. cnn_analysis.py's
RSA analyzers -- can treat both models identically.
"""

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet50_Weights

from psm_final.helpers.training_testing import train_and_test, test
from psm_final.models.cnn_basemodel import COCOClassificationDataset, build_parser, parse_args


def build_resnet_model(num_classes):
    # weights=ResNet50_Weights.DEFAULT is the current torchvision API (replaces
    # the deprecated pretrained=True) -- loads ImageNet-pretrained weights for
    # every layer except the final one, which gets swapped out below.
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    coco_train = COCOClassificationDataset(
        annotation_path=args.annotation_path,
        image_dir=args.coco_root,
        cache_dir=args.cache_dir,
        image_size=args.image_size,
        max_images=args.max_train_images or None,
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

    # sanity check -- both datasets must use the same category order, or label
    # index 12 could mean "dog" in one and something else in the other.
    assert coco_train.classes == coco_test.classes, "class order mismatch between train and val!"

    # create an instance of the model
    resnet50 = build_resnet_model(num_classes=len(coco_train.classes))

    # Move the model to the desired device
    resnet50 = resnet50.to(device)

    # Criterion and optimizer
    resnet_criterion = nn.CrossEntropyLoss()  # Standardly used for classification
    resnet_optimizer = torch.optim.Adam(resnet50.parameters(), lr=args.learning_rate)

    history, val_history = train_and_test(
        resnet50, coco_dataloader_train, coco_dataloader_test,
        args.num_epochs, resnet_criterion, resnet_optimizer, device,
    )

    # train_and_test() now returns the per-epoch val loss/accuracy it evaluates
    # each epoch; one explicit test() call still confirms the final numbers.
    val_loss, val_accuracy = test(resnet50, coco_dataloader_test, resnet_criterion, device)

    out_dir = Path('./results/cnn_resnet50')
    out_dir.mkdir(parents=True, exist_ok=True)
    # Per-epoch validation curve (columns: epoch, val_loss, val_accuracy_pct) --
    # the file plot_cnn_training.py reads for the ResNet-50 panels.
    pd.DataFrame(val_history).to_csv(out_dir / 'resnet50_val_history.csv', index=False)

    # Same metadata-wrapped save format as cnn_basemodel.py, so cnn_analysis.py
    # style loading code (checkpoint['model_state_dict'], checkpoint['classes'])
    # works identically for both models.
    torch.save({
        'model_state_dict': resnet50.state_dict(),
        'classes': coco_train.classes,
        'num_classes': len(coco_train.classes),
    }, out_dir / 'resnet50.pth')
    print(f'saved model to {out_dir / "resnet50.pth"}')


if __name__ == '__main__':
    main()