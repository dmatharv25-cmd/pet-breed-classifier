import time
import json
import torch
import torch.nn as nn
from torchvision import models
from src.loaders import get_loaders

SEED = 42
EPOCHS = 5
torch.manual_seed(SEED)


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(1) == y).sum().item()
            total += len(y)
    return correct / total


if __name__ == "__main__":
    train_loader, val_loader, _, classes = get_loaders()

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, len(classes))

    head_params = list(model.fc.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]

    opt = torch.optim.Adam([
        {"params": backbone_params, "lr": 1e-4},
        {"params": head_params, "lr": 1e-3},
    ])
    loss_fn = nn.CrossEntropyLoss()

    best_val = 0.0
    history = []
    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        model.train()
        run_loss = 0.0
        for x, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(y)
        train_loss = run_loss / len(train_loader.dataset)
        val_acc = evaluate(model, val_loader)
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), "resnet18_finetuned.pt")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc})
        print(f"epoch {epoch} | train loss {train_loss:.3f} | val acc {val_acc:.3f} | {time.time() - start:.0f}s")

    with open("resnet18_finetune_history.json", "w") as f:
        json.dump(history, f)
    print("best val acc:", round(best_val, 3))