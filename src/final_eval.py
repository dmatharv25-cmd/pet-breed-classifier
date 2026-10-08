import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from src.loaders import get_loaders
from src.small_cnn import SmallCNN


def predict(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x).argmax(1))
            labels.append(y)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


def bootstrap_ci(correct, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(correct)
    accs = [correct[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    return np.percentile(accs, [2.5, 97.5])


if __name__ == "__main__":
    _, _, test_loader, classes = get_loaders(batch_size=64)

    # 1. small CNN
    small = SmallCNN(len(classes))
    small.load_state_dict(torch.load("small_cnn.pt"))

    # 2. frozen ResNet18 (backbone + trained head)
    frozen = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    frozen.fc = nn.Linear(512, len(classes))
    frozen.fc.load_state_dict(torch.load("resnet18_head.pt"))

    # 3. fine-tuned ResNet18
    ft = models.resnet18()
    ft.fc = nn.Linear(512, len(classes))
    ft.load_state_dict(torch.load("resnet18_finetuned.pt"))

    results = {}
    for name, model in [("small_cnn", small), ("resnet18_frozen", frozen), ("resnet18_finetuned", ft)]:
        preds, labels = predict(model, test_loader)
        correct = (preds == labels).astype(float)
        lo, hi = bootstrap_ci(correct)
        print(f"{name}: test acc {correct.mean():.3f} (95% interval {lo:.3f} to {hi:.3f})")
        results[name] = {"preds": preds.tolist(), "correct": correct.tolist()}
        if name == "resnet18_finetuned":
            ft_preds, ft_labels = preds, labels

    # paired comparison: fine-tuned minus frozen
    c_ft = np.array(results["resnet18_finetuned"]["correct"])
    c_fr = np.array(results["resnet18_frozen"]["correct"])
    rng = np.random.default_rng(1)
    n = len(c_ft)
    diffs = []
    for _ in range(2000):
        i = rng.integers(0, n, n)
        diffs.append(c_ft[i].mean() - c_fr[i].mean())
    d_lo, d_hi = np.percentile(diffs, [2.5, 97.5])
    print(f"fine-tuned minus frozen: {c_ft.mean() - c_fr.mean():+.3f} (95% interval {d_lo:+.3f} to {d_hi:+.3f})")

    np.save("test_preds_finetuned.npy", ft_preds)
    np.save("test_labels.npy", ft_labels)
    with open("test_results.json", "w") as f:
        json.dump({k: v["correct"] for k, v in results.items()}, f)
    print("saved test_preds_finetuned.npy, test_labels.npy, test_results.json")