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

        # Final projection back to embedding dim (not vocab size)
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, x, t=None, is_embedded=False):
        """
        Forward pass for diffusion training.
        Args:
            x (Tensor): [batch_size, seq_len] (tokens) or [batch_size, seq_len, dim] (embedded)
            t (Tensor): [batch_size], timestep embeddings (unused for now).
            is_embedded (bool): If True, x is already embedded.
        Returns:
            logits: [batch_size, seq_len, dim]
        """
        if not is_embedded:
            x = self.embedding(x)
        x = x.transpose(1, 2)  # [batch, dim, seq_len]

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

        # Project to embedding dim
        return self.output_proj(x)

# notes:
	# • This is a U-Net–inspired architecture, but simplified.
	# • Token Embedding turns integer tokens into vector representations.
	# • Conv1d layers slide across the sequence to learn local patterns (like motifs).
	# • Could add skip connections later for a full U-Net flavor.
