import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from torchvision.datasets import OxfordIIITPet
from src.loaders import eval_tf, ROOT

SEEDS = [42, 1, 2]

data = torch.load("resnet18_features.pt")
Xtr, ytr, Xva, yva = data["Xtr"], data["ytr"], data["Xva"], data["yva"]

# extract test features once (about 3 to 5 minutes on CPU)
backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
backbone.fc = nn.Identity()
backbone.eval()
test_ds = OxfordIIITPet(root=ROOT, split="test", target_types="category", transform=eval_tf)
feats, labels = [], []
with torch.no_grad():
    for x, y in DataLoader(test_ds, batch_size=64, shuffle=False):
        feats.append(backbone(x))
        labels.append(y)
Xte, yte = torch.cat(feats), torch.cat(labels)
print("test features:", tuple(Xte.shape))

results = {}
for seed in SEEDS:
    torch.manual_seed(seed)
    head = nn.Linear(512, 37)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)
    best_val, best_state = 0.0, None
    for epoch in range(30):
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
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        test_acc = (head(Xte).argmax(1) == yte).float().mean().item()
    results[str(seed)] = {"val_acc": best_val, "test_acc": test_acc}
    print(f"seed {seed}: val {best_val:.4f} | test {test_acc:.4f}")

accs = np.array([r["test_acc"] for r in results.values()])
print(f"mean {accs.mean():.4f} | std {accs.std(ddof=1):.4f}")

with open("frozen_seed_test_results.json", "w") as f:
    json.dump(results, f)