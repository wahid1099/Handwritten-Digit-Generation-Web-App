# app.py
import streamlit as st
import torch
import numpy as np
from PIL import Image
import io

# Load the generator model
class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Linear(100 + 10, 256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(256, 512),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(512, 1024),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(1024, 784),
            torch.nn.Tanh()
        )
    
    def forward(self, z, labels):
        labels_embed = torch.zeros((labels.size(0), 10))
        labels_embed.scatter_(1, labels.unsqueeze(1), 1)
        z = torch.cat([z, labels_embed], dim=1)
        output = self.main(z)
        return output.view(-1, 1, 28, 28)

# Initialize and load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G = Generator().to(device)
G.load_state_dict(torch.load('cvae_mnist.pth', map_location=device))
G.eval()

# Streamlit app
st.title('Handwritten Digit Generator')

# Digit selection
digit = st.selectbox('Select a digit to generate:', list(range(10)))

if st.button('Generate Digits'):
    with st.spinner('Generating...'):
        # Generate 5 images
        z = torch.randn(5, 100).to(device)
        labels = torch.full((5,), digit, dtype=torch.long).to(device)
        
        generated_images = G(z, labels).detach().cpu()
        
        # Display images
        cols = st.columns(5)
        for i in range(5):
            img = generated_images[i].squeeze().numpy()
            img = (img + 1) / 2  # Scale from [-1, 1] to [0, 1]
            img = (img * 255).astype(np.uint8)
            
            pil_img = Image.fromarray(img)
            cols[i].image(pil_img, caption=f'Digit {digit}', use_column_width=True)