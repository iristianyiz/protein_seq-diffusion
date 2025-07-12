import torch
from torch.utils.data import Dataset

class ProteinDataset(Dataset):
    def __init__(self, filepath, seq_len=64):
        self.aa_vocab = 'ACDEFGHIKLMNPQRSTVWY'
        self.char2idx = {c: i + 1 for i, c in enumerate(self.aa_vocab)}  # 0 = padding
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        self.seq_len = seq_len

        with open(filepath, 'r') as f:
            lines = [line.strip().upper() for line in f if line.strip()]

        # Clean and pad all sequences
        self.sequences = []
        for line in lines:
            cleaned = ''.join([c if c in self.aa_vocab else 'A' for c in line[:seq_len]])
            padded = cleaned.ljust(seq_len, 'A')
            self.sequences.append(padded)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        token_ids = [self.char2idx.get(c, 0) for c in seq]
        return torch.tensor(token_ids, dtype=torch.long)

    def decode(self, tensor):
        return ''.join([self.idx2char.get(int(i), '-') for i in tensor])