import time
from src.loaders import get_loaders

train_loader, val_loader, test_loader, classes = get_loaders()

print("train batches:", len(train_loader))
print("val batches:", len(val_loader))
print("test batches:", len(test_loader))

x, y = next(iter(train_loader))
print("batch shape:", tuple(x.shape))
print("labels shape:", tuple(y.shape))
print("pixel min/max:", round(x.min().item(), 2), round(x.max().item(), 2))

start = time.time()
for i, (x, y) in enumerate(train_loader):
    if i == 9:
        break
print("time for 10 batches:", round(time.time() - start, 1), "seconds")