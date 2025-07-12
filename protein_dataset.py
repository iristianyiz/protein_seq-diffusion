import torch
from torch.utils.data import Dataset

class ProteinDataset(Dataset):
    def __init__(self, filepath, seq_len=128):
        self.aa_vocab = 'ACDEFGHIKLMNPQRSTVWY'  # 20 amino acids
        self.char2idx = {c: i + 1 for i, c in enumerate(self.aa_vocab)}  # 0 reserved for padding
        self.idx2char = {i: c for c, i in self.char2idx.items()}

        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if len(line.strip()) >= seq_len]

        self.sequences = [seq[:seq_len] for seq in lines]
        self.seq_len = seq_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        token_ids = [self.char2idx.get(char, 0) for char in seq]
        return torch.tensor(token_ids, dtype=torch.long)

    def decode(self, tensor):
        return ''.join([self.idx2char.get(int(i), '-') for i in tensor])