from collections import Counter
from torchvision.datasets import OxfordIIITPet

ROOT = "data"

trainval = OxfordIIITPet(root=ROOT, split="trainval", target_types="category", download=True)
test = OxfordIIITPet(root=ROOT, split="test", target_types="category", download=True)

counts = Counter(trainval._labels)

print("trainval images:", len(trainval))
print("test images:", len(test))
print("number of classes:", len(trainval.classes))
print("first 5 classes:", trainval.classes[:5])
print("images per class in trainval: min", min(counts.values()), "max", max(counts.values()))
print("image size of first sample:", trainval[0][0].size)