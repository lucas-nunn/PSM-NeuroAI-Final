import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models


from psm_final.models.beta_vae import _image_transform
from psm_final.helpers.training_testing import train
from psm_final.models.cnn_basemodel import COCOClassificationDataset, build_parser , parse_args 

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes):
    model = models.resnet50(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    coco_train = COCOClassificationDataset(
        annotation_path=args.annotation_path,
        image_dir=args.coco_root,
        cache_dir=args.cache_dir,
    )

    coco_dataloader_train = DataLoader(
        coco_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # create an instance of the model 
    resnet50 = build_model(num_classes=len(coco_train.classes))

    # Move the model to the desired device
    resnet50 = resnet50.to(device)

    #Criterion and optimizer
    cnn_criterion = nn.CrossEntropyLoss()  # Standardly used for classification
    cnn_optimizer = torch.optim.Adam(resnet50.parameters(), lr=args.learning_rate)
    
    train(resnet50, coco_dataloader_train, 50, cnn_criterion, cnn_optimizer, device)

if __name__ == '__main__':
    main()