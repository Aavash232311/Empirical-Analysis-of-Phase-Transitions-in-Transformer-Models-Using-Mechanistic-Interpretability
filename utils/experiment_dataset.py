import torch
import random
from torch.utils.data import Dataset

def get_seed_split(mod=97, train_frac=0.3):
    all_seeds = [(i, j) for i in range(mod) for j in range(mod)]
    random.shuffle(all_seeds)

    split_idx = int(train_frac * len(all_seeds))
    return all_seeds[:split_idx], all_seeds[split_idx:]

class FibonacciModDataset(Dataset):
    def __init__(self, seeds, seq_len=10, mod=97, num_samples=10000):
        self.mod = mod
        self.samples = []

        for _ in range(num_samples):
            a, b = random.choice(seeds)

            seq = self.generate_fib_sequence(a, b, seq_len + 1, mod)

            x = torch.tensor(seq[:-1], dtype=torch.long)
            y = torch.tensor(seq[1:], dtype=torch.long)

            self.samples.append((x, y))

    def generate_fib_sequence(self, a, b, length, mod):
        seq = [a, b]
        while len(seq) < length:
            seq.append((seq[-1] + seq[-2]) % mod)
        return seq

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
vocab_size = 53
seq_len = 20
train_seeds, val_seeds = get_seed_split(mod=vocab_size, train_frac=0.3)

train_ds = FibonacciModDataset(train_seeds, seq_len=seq_len, mod=vocab_size, num_samples=20000)
val_ds   = FibonacciModDataset(val_seeds,   seq_len=seq_len, mod=vocab_size, num_samples=5000)
