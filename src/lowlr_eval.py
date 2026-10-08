import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from src.loaders import get_loaders

FILES = {
    42: "resnet18_finetuned_lr1e-5.pt",
    1: "resnet18_finetuned_lr1e-5_seed1.pt",
    2: "resnet18_finetuned_lr1e-5_seed2.pt",
}

_, _, test_loader, classes = get_loaders(batch_size=64)

results = {}
for seed, path in FILES.items():
    model = models.resnet18()
    model.fc = nn.Linear(512, len(classes))
    model.load_state_dict(torch.load(path))
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            correct += (model(x).argmax(1) == y).sum().item()
    acc = correct / len(test_loader.dataset)
    results[str(seed)] = acc
    print(f"seed {seed}: test acc {acc:.4f}")

accs = np.array(list(results.values()))
print(f"mean {accs.mean():.4f} | std {accs.std(ddof=1):.4f}")

with open("lowlr_seed_test_results.json", "w") as f:
    json.dump(results, f)