import os
from collections import Counter
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from src.loaders import eval_tf

classes = json_classes = None
from torchvision.datasets import OxfordIIITPet
from src.loaders import ROOT
classes = OxfordIIITPet(root=ROOT, split="test", target_types="category").classes

backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
backbone.fc = nn.Identity()
backbone.eval()
sd = torch.load("resnet18_head.pt")
sd = {k.replace("fc.", ""): v for k, v in sd.items()}
head = nn.Linear(512, 37)
head.load_state_dict(sd)
head.eval()

d = "data/non_pet"
files = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png")))
rows = []
for f in files:
    try:
        with Image.open(os.path.join(d, f)) as im:
            x = eval_tf(im.convert("RGB")).unsqueeze(0)
    except Exception as e:
        print("skipped", f, e)
        continue
    with torch.no_grad():
        p = head(backbone(x)).softmax(1)[0]
    c, i = p.max(0)
    rows.append((c.item(), f, classes[i.item()]))

n = len(rows)
print(f"{n} non-pet images")
for th in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
    hi = sum(1 for c, _, _ in rows if c >= th)
    print(f"confidence >= {th:.1f}: {hi} of {n} ({hi / n * 100:.1f}%) get no warning")
print("\nmost common predicted breeds:", Counter(b for _, _, b in rows).most_common(5))
print("\nmost confident images (check these by eye):")
for c, f, b in sorted(rows, reverse=True)[:5]:
    print(f"  {f}: {b} {c * 100:.0f}%")
