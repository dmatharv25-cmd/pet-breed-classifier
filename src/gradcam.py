import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import models
from torchvision.datasets import OxfordIIITPet
from src.loaders import eval_tf, IMG_SIZE

preds = np.load("test_preds_finetuned.npy")
labels = np.load("test_labels.npy")

raw = OxfordIIITPet(root="data", split="test", target_types="category")
ds = OxfordIIITPet(root="data", split="test", target_types="category", transform=eval_tf)
classes = ds.classes

model = models.resnet18()
model.fc = nn.Linear(512, len(classes))
model.load_state_dict(torch.load("resnet18_finetuned.pt"))
model.eval()

store = {}
model.layer4.register_forward_hook(lambda m, i, o: store.update(act=o))
model.layer4.register_full_backward_hook(lambda m, gi, go: store.update(grad=go[0]))


def cam_for(idx):
    x, _ = ds[idx]
    out = model(x.unsqueeze(0))
    cls = out.argmax(1).item()
    model.zero_grad()
    out[0, cls].backward()
    w = store["grad"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * store["act"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().numpy()
    return cam / (cam.max() + 1e-8)


rng = np.random.default_rng(0)
correct_idx = rng.choice(np.where(preds == labels)[0], 4, replace=False)
wrong_idx = rng.choice(np.where(preds != labels)[0], 4, replace=False)

fig, axes = plt.subplots(2, 8, figsize=(20, 6))
for col, idx in enumerate(list(correct_idx) + list(wrong_idx)):
    img = raw[idx][0].resize((IMG_SIZE, IMG_SIZE))
    cam = cam_for(idx)
    ok = preds[idx] == labels[idx]
    axes[0, col].imshow(img)
    axes[0, col].set_title(f"true: {classes[labels[idx]]}\npred: {classes[preds[idx]]}",
                           fontsize=7, color="green" if ok else "red")
    axes[1, col].imshow(img)
    axes[1, col].imshow(cam, cmap="jet", alpha=0.45)
    axes[0, col].axis("off")
    axes[1, col].axis("off")
plt.tight_layout()
plt.savefig("gradcam.png", dpi=100)
print("saved gradcam.png")
print("correct examples:", [int(i) for i in correct_idx])
print("wrong examples:", [int(i) for i in wrong_idx])
