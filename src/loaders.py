import json
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

ROOT = "data"
IMG_SIZE = 160
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def get_loaders(batch_size=32, num_workers=0):
    with open("splits.json") as f:
        splits = json.load(f)

    train_full = OxfordIIITPet(root=ROOT, split="trainval", target_types="category", transform=train_tf)
    val_full = OxfordIIITPet(root=ROOT, split="trainval", target_types="category", transform=eval_tf)
    test_ds = OxfordIIITPet(root=ROOT, split="test", target_types="category", transform=eval_tf)

    train_ds = Subset(train_full, splits["train"])
    val_ds = Subset(val_full, splits["val"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, train_full.classes