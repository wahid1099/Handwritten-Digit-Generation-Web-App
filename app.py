# app.py
import streamlit as st, torch, torch.nn as nn
from torchvision.utils import make_grid

class Decoder(nn.Module):
    def __init__(self,z_dim=20):
        super().__init__()
        self.fc = nn.Linear(z_dim+10,64*7*7)
        self.deconv = nn.Sequential(nn.ConvTranspose2d(64,32,3,2,1,1),nn.ReLU(),
                                    nn.ConvTranspose2d(32,1,3,2,1,1),nn.Sigmoid())
    def forward(self,z,y):
        z = torch.cat([z, torch.eye(10)[y]],1)
        h = self.fc(z).view(-1,64,7,7)
        return self.deconv(h)

dec = Decoder(); dec.load_state_dict(torch.load('cvae_mnist.pth',map_location='cpu'),strict=False)
dec.eval()

st.title("MNIST Digit Generator")
d = st.selectbox("Digit",list(range(10)))
if st.button("Generate"):
    with torch.no_grad():
        z = torch.randn(5,20)
        lbl = torch.tensor([d]*5)
        imgs = dec(z,lbl).cpu()
    grid = make_grid(imgs,nrow=5,padding=5,normalize=True)
    st.image(grid.permute(1,2,0).numpy(),width=400)
