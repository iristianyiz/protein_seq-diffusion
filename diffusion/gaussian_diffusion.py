import torch
import torch.nn.functional as F
import math

class GaussianDiffusion(torch.nn.Module):
    def __init__(self, model, seq_length, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        """
        model: your denoising model (e.g., 1D UNet)
        seq_length: length of tokenized sequence
        timesteps: number of diffusion steps
        """
        super().__init__()
        self.model = model
        self.seq_length = seq_length
        self.timesteps = timesteps

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps)

        alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def forward(self, x_0, t):
        """
        Forward pass: take x_0 and timestep t, return loss.
        """
        # Embed tokens to get [B, seq_len, embed_dim]
        x_0_emb = self.model.embedding(x_0)  # [B, seq_len, dim]
        noise = torch.randn_like(x_0_emb, dtype=torch.float32)  # [B, seq_len, dim]

        # Reshape for broadcasting
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)

        # Noised version
        x_t = sqrt_alpha * x_0_emb + sqrt_one_minus_alpha * noise

        # Model predicts noise from noised embedding
        pred_noise = self.model(x_t, t, is_embedded=True)  # <-- see below

        return F.mse_loss(pred_noise, noise)

    def sample(self, batch_size, device):
        """
        Sample from noise using denoising process.
        """
        x_t = torch.randn(batch_size, self.seq_length, self.model.output_proj.out_features).to(device)

        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((batch_size,), t, dtype=torch.long, device=device)
            pred_noise = self.model(x_t.argmax(dim=-1), t_batch)

            alpha = self.alphas_cumprod[t]
            sqrt_alpha = torch.sqrt(alpha)
            sqrt_one_minus_alpha = torch.sqrt(1 - alpha)

            x_t = sqrt_alpha * x_t + sqrt_one_minus_alpha * pred_noise

        return x_t.argmax(dim=-1)

# notes:
	# •	Works with 1D sequences
	# •	Uses token embeddings
	# •	Uses MSE loss between predicted and actual noise
	# •	Performs greedy decoding for simplicity