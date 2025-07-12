from utils.protein_dataset import ProteinDataset


ds = ProteinDataset('data/protein_sequences.txt', seq_len=50) # short seq 
print("Number of sequences:", len(ds))
print("Sample tokenized:", ds[0])
print("Decoded:", ds.decode(ds[0]))


# output:

# Number of sequences: 3
# Sample tokenized: tensor([11, 18,  9, 18, 20,  1, 13,  1, 16, 16,  1, 12, 11, 16, 18,  6,  5,  3,
#         18, 10,  6,  1,  1, 18, 17, 13, 18,  3,  6,  1, 10, 10,  6,  3, 18, 18,
#         17, 18,  4,  1,  1,  4, 17,  5, 16, 10, 12, 12, 10,  6])
# Decoded: MVKVYAPASSANMSVGFDVLGAAVTPVDGALLGDVVTVEAAETFSLNNLG