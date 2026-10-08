import torch
import torch.nn as nn
import gradio as gr
from torchvision import models
from torchvision.datasets import OxfordIIITPet
from src.loaders import eval_tf

classes = OxfordIIITPet(root="data", split="test", target_types="category").classes

model = models.resnet18()
model.fc = nn.Linear(512, len(classes))
model.load_state_dict(torch.load("resnet18_finetuned.pt", map_location="cpu"))
model.eval()


def predict(img):
    if img is None:
        return {}
    x = eval_tf(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    top = torch.topk(probs, 3)
    return {classes[i]: float(p) for p, i in zip(top.values, top.indices)}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a cat or dog photo"),
    outputs=gr.Label(num_top_classes=3, label="Top 3 breeds"),
    title="Pet Breed Classifier (ResNet18, 37 breeds)",
    description="Fine-tuned ResNet18 on Oxford-IIIT Pet. About 83% test accuracy. Look-alike breeds, such as Pit Bull and Staffordshire Bull Terrier, are often confused.",
)

if __name__ == "__main__":
    demo.launch()
