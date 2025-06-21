# app.py
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

class DigitGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(10 + 100, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 784),
            torch.nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.cat([noise, labels], dim=1)
        return self.fc(x).view(-1, 1, 28, 28)

def one_hot(labels, num_classes=10):
    return torch.eye(num_classes)[labels]

def generate_images(model, digit, n=5):
    model.eval()
    noise = torch.randn(n, 100)
    labels = one_hot(torch.tensor([digit] * n))
    with torch.no_grad():
        imgs = model(noise, labels)
    return imgs

# Load model
model = DigitGenerator()
model.load_state_dict(torch.load("digit_generator.pth", map_location="cpu"))

# Web UI
st.title("Handwritten Digit Generator")
digit = st.selectbox("Select a digit to generate:", list(range(10)))
if st.button("Generate Images"):
    imgs = generate_images(model, digit)
    grid = make_grid(imgs, nrow=5, normalize=True)
    npimg = grid.numpy().transpose((1, 2, 0))
    st.image(npimg, caption=f"Generated Digit: {digit}", width=300)
