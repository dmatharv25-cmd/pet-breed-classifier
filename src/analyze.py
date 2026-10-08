import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix
from torchvision.datasets import OxfordIIITPet

preds = np.load("test_preds_finetuned.npy")
labels = np.load("test_labels.npy")
classes = OxfordIIITPet(root="data", split="test", target_types="category").classes

# per-class accuracy
per_class = []
for c in range(len(classes)):
    mask = labels == c
    per_class.append((classes[c], (preds[mask] == c).mean(), int(mask.sum())))
per_class.sort(key=lambda t: t[1])

print("5 hardest breeds:")
for name, acc, n in per_class[:5]:
    print(f"  {name}: {acc:.3f} ({n} test images)")
print("5 easiest breeds:")
for name, acc, n in per_class[-5:]:
    print(f"  {name}: {acc:.3f} ({n} test images)")

# most common confusions
pairs = Counter()
for t, p in zip(labels, preds):
    if t != p:
        pairs[(classes[t], classes[p])] += 1
print("most common mistakes (true -> predicted):")
for (t, p), n in pairs.most_common(8):
    print(f"  {t} -> {p}: {n}")

# confusion matrix image
cm = confusion_matrix(labels, preds)
plt.figure(figsize=(12, 10))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.title("Confusion matrix, fine-tuned ResNet18 (test)")
plt.xlabel("predicted class index")
plt.ylabel("true class index")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=100)
print("saved confusion_matrix.png")