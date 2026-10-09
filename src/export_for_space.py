import json
import os
import torch
import torch.nn as nn
from torchvision import models
from torchvision.datasets import OxfordIIITPet

classes = OxfordIIITPet(root="data", split="test", target_types="category").classes
os.makedirs("space", exist_ok=True)

with open("space/classes.json", "w") as f:
    json.dump(classes, f)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(512, len(classes))
model.fc.load_state_dict(torch.load("resnet18_head.pt"))
torch.save(model.state_dict(), "space/resnet18_frozen.pt")
print("saved space/classes.json and space/resnet18_frozen.pt")