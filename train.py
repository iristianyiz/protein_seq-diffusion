import torch
from torch.utils.data import DataLoader
from utils.protein_dataset import ProteinDataset
from models.unet1d import Simple1DUNet
from diffusion.gaussian_diffusion import GaussianDiffusion

# Hyperparameters
BATCH_SIZE = 8
SEQ_LEN = 64
VOCAB_SIZE = 21  # 20 amino acids + 1 pad
EPOCHS = 10
TIMESTEPS = 1000
LR = 1e-4

# Dataset
dataset = ProteinDataset("data/protein_sequences.txt", seq_len=SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
model = Simple1DUNet(seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE)
diffusion = GaussianDiffusion(model, seq_length=SEQ_LEN, timesteps=TIMESTEPS)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
diffusion = diffusion.to(device)

# Training loop
for epoch in range(EPOCHS):
    for batch in dataloader:
        batch = batch.to(device)

        t = torch.randint(0, TIMESTEPS, (batch.size(0),), device=device)
        loss = diffusion(batch, t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")