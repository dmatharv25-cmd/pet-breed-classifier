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
    prob = torch.softmax(out, dim=1)[0, cls].item()
    model.zero_grad()
    out[0, cls].backward()
    w = store["grad"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * store["act"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().numpy()
    return cam / (cam.max() + 1e-8), prob


PAIRS = [
    ("American Pit Bull Terrier", "Staffordshire Bull Terrier"),
    ("Beagle", "Basset Hound"),
    ("Egyptian Mau", "Bengal"),
    ("Ragdoll", "Birman"),
]

fig, axes = plt.subplots(4, 4, figsize=(14, 14))
for row, (true_name, pred_name) in enumerate(PAIRS):
    t, p = classes.index(true_name), classes.index(pred_name)
    idxs = np.where((labels == t) & (preds == p))[0][:2]
    print(f"{true_name} -> {pred_name}: {len(np.where((labels == t) & (preds == p))[0])} mistakes, using indices {[int(i) for i in idxs]}")
    for k, idx in enumerate(idxs):
        img = raw[idx][0].resize((IMG_SIZE, IMG_SIZE))
        cam, prob = cam_for(idx)
        a_img, a_cam = axes[row, 2 * k], axes[row, 2 * k + 1]
        a_img.imshow(img)
        a_img.set_title(f"true: {true_name}\npred: {pred_name} (p={prob:.2f})", fontsize=8, color="red")
        a_cam.imshow(img)
        a_cam.imshow(cam, cmap="jet", alpha=0.45)
        a_cam.set_title("heat for predicted class", fontsize=8)
        a_img.axis("off")
        a_cam.axis("off")
plt.tight_layout()
plt.savefig("gradcam_pairs.png", dpi=110)
print("saved gradcam_pairs.png")