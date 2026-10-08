import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import models
from torchvision.datasets import OxfordIIITPet
import json
from src.loaders import eval_tf, ROOT

SEED = 42
torch.manual_seed(SEED)


def extract(model, ds, indices=None, batch_size=64):
    if indices is not None:
        ds = Subset(ds, indices)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    feats, labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            feats.append(model(x))
            labels.append(y)
    return torch.cat(feats), torch.cat(labels)


if __name__ == "__main__":
    with open("splits.json") as f:
        splits = json.load(f)

    weights = models.ResNet18_Weights.IMAGENET1K_V1
    backbone = models.resnet18(weights=weights)
    backbone.fc = nn.Identity()

    trainval = OxfordIIITPet(root=ROOT, split="trainval", target_types="category", transform=eval_tf)

    start = time.time()
    Xtr, ytr = extract(backbone, trainval, splits["train"])
    Xva, yva = extract(backbone, trainval, splits["val"])
    print("feature extraction:", round(time.time() - start), "seconds")
    print("train features:", tuple(Xtr.shape), "val features:", tuple(Xva.shape))
    torch.save({"Xtr": Xtr, "ytr": ytr, "Xva": Xva, "yva": yva}, "resnet18_features.pt")

    head = nn.Linear(512, 37)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)

    best_val = 0.0
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
            best_val = val_acc
            torch.save(head.state_dict(), "resnet18_head.pt")
        if epoch % 5 == 0:
            print(f"epoch {epoch:2d} | val acc {val_acc:.3f}")
    print("best val acc:", round(best_val, 3))