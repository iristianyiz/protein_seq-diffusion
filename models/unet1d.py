import torch
import torch.nn as nn

class Simple1DUNet(nn.Module):
    def __init__(self, seq_len=128, vocab_size=21, dim=128):
        """
        A simple 1D UNet-like model for denoising amino acid sequences.

        Args:
            seq_len (int): Length of input sequences.
            vocab_size (int): Number of tokens (20 amino acids + 1 padding).
            dim (int): Internal hidden dimensionality.
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dim)

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.pool = nn.MaxPool1d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.dec1 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Final projection back to token logits
        self.output_proj = nn.Linear(dim, vocab_size)

    def forward(self, x, t=None):
        """
        Forward pass for diffusion training.
        Args:
            x (Tensor): [batch_size, seq_len], integer tokens.
            t (Tensor): [batch_size], timestep embeddings (unused for now).
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        # Shape: [batch, seq_len, dim]
        x = self.embedding(x)
        x = x.transpose(1, 2)  # -> [batch, dim, seq_len]

        # Encode
        x = self.enc1(x)
        x = self.pool(x)  # downsample

        # Bottleneck
        x = self.bottleneck(x)

        # Decode
        x = self.upsample(x)
        x = self.dec1(x)

        # Back to [batch, seq_len, dim]
        x = x.transpose(1, 2)

        # Project to vocab logits
        return self.output_proj(x)