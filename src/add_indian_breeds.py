import copy
import json
import os
import random
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.datasets import OxfordIIITPet
from src.loaders import eval_tf, ROOT

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

NEW_DIR = os.path.join(ROOT, "new_breeds")


def discover():
    found = {}
    for folder in sorted(os.listdir(NEW_DIR)):
        d = os.path.join(NEW_DIR, folder)
        if not os.path.isdir(d):
            continue
        n = len([f for f in os.listdir(d) if f.lower().endswith(EXTS)])
        if n < MIN_PER_BREED:
            print(f"skipping {folder}: {n} photos (need {MIN_PER_BREED})")
        else:
            found[folder] = folder.replace("_", " ").title()
    return found
MIN_PER_BREED = 20
EXTS = (".jpg", ".jpeg", ".png")
OUT_DIR = "space_v2"


class PathDataset(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        with Image.open(path) as im:
            img = im.convert("RGB")
        return self.tf(img), y


def extract(model, ds, batch_size=64):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    feats, labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            feats.append(model(x))
            labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def main():
    oxford_classes = OxfordIIITPet(root=ROOT, split="test", target_types="category").classes
    BREEDS = discover()
    if not BREEDS:
        raise SystemExit('no breed folder has enough photos yet')
    indian_names = list(BREEDS.values())
    classes = list(oxford_classes) + indian_names
    n_old, n_all = len(oxford_classes), len(classes)

    # 1. collect and split Indian photos per breed (70/15/15)
    split = {"train": [], "val": [], "test": []}
    for k, folder in enumerate(BREEDS):
        d = os.path.join(NEW_DIR, folder)
        files = sorted(f for f in os.listdir(d) if f.lower().endswith(EXTS))
        if len(files) < MIN_PER_BREED:
            raise SystemExit(f"{folder}: only {len(files)} photos, need at least {MIN_PER_BREED}")
        random.Random(SEED).shuffle(files)
        n_eval = max(5, round(0.15 * len(files)))
        label = n_old + k
        for name in files[:n_eval]:
            split["test"].append((os.path.join(d, name), label))
        for name in files[n_eval:2 * n_eval]:
            split["val"].append((os.path.join(d, name), label))
        for name in files[2 * n_eval:]:
            split["train"].append((os.path.join(d, name), label))
        print(f"{folder}: {len(files)} photos")
    with open("indian_splits.json", "w") as f:
        json.dump(split, f)

    # 2. features
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    ox = torch.load("resnet18_features.pt")
    if os.path.exists("resnet18_test_features.pt"):
        t = torch.load("resnet18_test_features.pt")
        Xte_ox, yte_ox = t["X"], t["y"]
    else:
        print("extracting Oxford test features (a few minutes on CPU)...")
        test_ds = OxfordIIITPet(root=ROOT, split="test", target_types="category", transform=eval_tf)
        Xte_ox, yte_ox = extract(backbone, test_ds)
        torch.save({"X": Xte_ox, "y": yte_ox}, "resnet18_test_features.pt")
    print("extracting Indian breed features...")
    ind = {k: extract(backbone, PathDataset(v, eval_tf)) for k, v in split.items()}

    Xtr = torch.cat([ox["Xtr"], ind["train"][0]])
    ytr = torch.cat([ox["ytr"], ind["train"][1]])
    Xva = torch.cat([ox["Xva"], ind["val"][0]])
    yva = torch.cat([ox["yva"], ind["val"][1]])

    # 3. train a new head (same recipe as resnet_frozen.py)
    head = nn.Linear(512, n_all)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(torch.utils.data.TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)
    best_val, best_state = 0.0, None
    for epoch in range(1, 31):
        head.train()
        for x, y in loader:
            opt.zero_grad()
            loss_fn(head(x), y).backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            val_acc = (head(Xva).argmax(1) == yva).float().mean().item()
        if val_acc > best_val:
            best_val, best_state = val_acc, copy.deepcopy(head.state_dict())
        if epoch % 5 == 0:
            print(f"epoch {epoch:2d} | val acc {val_acc:.3f}")
    head.load_state_dict(best_state)
    head.eval()
    print("best val acc:", round(best_val, 3))

    # 4. evaluate on held-out test sets
    with torch.no_grad():
        p_ox = head(Xte_ox).argmax(1)
        Xte_in, yte_in = ind["test"]
        p_in = head(Xte_in).argmax(1)
    print(f"\nOxford test acc ({n_old} original breeds, {n_all} outputs): {(p_ox == yte_ox).float().mean().item():.3f}")
    print(f"Oxford test images wrongly sent to an Indian breed: {(p_ox >= n_old).sum().item()} of {len(yte_ox)}")
    print(f"Indian test acc: {(p_in == yte_in).float().mean().item():.3f} on {len(yte_in)} photos")
    for k, name in enumerate(indian_names):
        m = yte_in == (n_old + k)
        print(f"  {name}: {(p_in[m] == yte_in[m]).float().mean().item():.3f} ({int(m.sum())} test photos)")

    # 5. export to a NEW folder (the live demo folder is untouched)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "classes.json"), "w") as f:
        json.dump(classes, f)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, n_all)
    model.fc.load_state_dict(head.state_dict())
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "resnet18_frozen.pt"))
    print(f"\nsaved {OUT_DIR}/classes.json and {OUT_DIR}/resnet18_frozen.pt")


if __name__ == "__main__":
    main()
