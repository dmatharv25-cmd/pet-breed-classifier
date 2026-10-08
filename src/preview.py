import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.datasets import OxfordIIITPet

ds = OxfordIIITPet(root="data", split="trainval", target_types="category")

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for ax, i in zip(axes.flat, range(0, 3680, 460)):
    img, label = ds[i]
    ax.imshow(img)
    ax.set_title(ds.classes[label], fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig("samples.png", dpi=100)
print("saved samples.png")