import json
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import OxfordIIITPet

ROOT = "data"
SEED = 42


def make_splits():
    trainval = OxfordIIITPet(root=ROOT, split="trainval", target_types="category")
    test = OxfordIIITPet(root=ROOT, split="test", target_types="category")

    labels = np.array(trainval._labels)
    idx = np.arange(len(trainval))
    train_idx, val_idx = train_test_split(
        idx, test_size=0.2, stratify=labels, random_state=SEED
    )

    splits = {
        "train": sorted(train_idx.tolist()),
        "val": sorted(val_idx.tolist()),
        "test_size": len(test),
    }
    with open("splits.json", "w") as f:
        json.dump(splits, f)

    print("train:", len(train_idx))
    print("val:", len(val_idx))
    print("test (untouched):", len(test))
    print("train per class: min", np.bincount(labels[train_idx]).min(),
          "max", np.bincount(labels[train_idx]).max())
    print("val per class: min", np.bincount(labels[val_idx]).min(),
          "max", np.bincount(labels[val_idx]).max())


if __name__ == "__main__":
    make_splits()