import torch
import torch.nn as nn

t = torch.load("resnet18_test_features.pt")
X, y = t["X"], t["y"]
sd = torch.load("resnet18_head.pt")
sd = {k.replace("fc.", ""): v for k, v in sd.items()}
head = nn.Linear(512, 37)
head.load_state_dict(sd)
head.eval()
with torch.no_grad():
    p = head(X).softmax(1)
conf, pred = p.max(1)
ok = (pred == y).float()
n = len(y)
print(f"test acc {ok.mean().item():.3f} on {n} images (should be about 0.84)")
print(f"mean confidence {conf.mean().item():.3f}")

print("\nthreshold | share of images at/above | acc at/above | acc below | wrong preds flagged")
wrong_total = (1 - ok).sum().item()
for th in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    hi = conf >= th
    lo = ~hi
    acc_hi = ok[hi].mean().item() if hi.any() else float("nan")
    acc_lo = ok[lo].mean().item() if lo.any() else float("nan")
    flagged = ((1 - ok)[lo]).sum().item() / wrong_total
    print(f"   {th:.1f}    |        {hi.float().mean().item()*100:5.1f}%          |    {acc_hi*100:5.1f}%    |  {acc_lo*100:5.1f}%  |   {flagged*100:5.1f}%")

ece = 0.0
edges = torch.linspace(0, 1, 11)
for a, b in zip(edges[:-1], edges[1:]):
    m = (conf > a) & (conf <= b)
    if m.any():
        ece += m.float().mean().item() * abs(ok[m].mean().item() - conf[m].mean().item())
print(f"\nexpected calibration error (10 bins): {ece:.3f}")
