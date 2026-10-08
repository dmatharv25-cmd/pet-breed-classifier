import json
import statistics
import torch
import torch.nn as nn
from torchvision import models
from src.loaders import get_loaders
from src.resnet_finetune import evaluate

if __name__ == "__main__":
    _, _, test_loader, classes = get_loaders()
    files = {
        42: "resnet18_finetuned.pt",
        1: "resnet18_finetuned_seed1.pt",
        2: "resnet18_finetuned_seed2.pt",
    }
    results = {}
    for seed, path in files.items():
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(512, len(classes))
        model.load_state_dict(torch.load(path))
        acc = evaluate(model, test_loader)
        results[seed] = acc
        print(f"seed {seed}: test acc {acc:.4f}")
    vals = list(results.values())
    print(f"mean {statistics.mean(vals):.4f} | std {statistics.stdev(vals):.4f}")
    with open("finetune_seed_test_results.json", "w") as f:
        json.dump(results, f)
