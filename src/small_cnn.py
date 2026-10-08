import time
import json
import torch
import torch.nn as nn
from src.loaders import get_loaders

SEED = 42
EPOCHS = 12
torch.manual_seed(SEED)


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    def __init__(self, n_classes=37):
        super().__init__()
        self.features = nn.Sequential(
            block(3, 32), block(32, 64), block(64, 128), block(128, 256)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.fc(self.drop(x))


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
    model = SmallCNN(len(classes))
    n_params = sum(p.numel() for p in model.parameters())
    print("parameters:", n_params)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
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
            torch.save(model.state_dict(), "small_cnn.pt")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc})
        print(f"epoch {epoch:2d} | train loss {train_loss:.3f} | val acc {val_acc:.3f} | {time.time() - start:.0f}s")

    with open("small_cnn_history.json", "w") as f:
        json.dump(history, f)
    print("best val acc:", round(best_val, 3))