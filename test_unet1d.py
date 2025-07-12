import torch
from models.unet1d import Simple1DUNet

model = Simple1DUNet(seq_len=64)
dummy_input = torch.randint(1, 21, (4, 64))  # [batch, seq_len]

logits = model(dummy_input)
print("Output shape:", logits.shape)  # Should be [4, 64, 21] 

# output:
# Output shape: torch.Size([4, 64, 21])

# 1D UNet is successfully:
    # •	Accepting a batch of 4 protein sequences,
	# •	Each with 64 positions (amino acids),
	# •	And predicting a distribution over 21 tokens (20 amino acids + 1 for padding/unknown) for each position.