import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import os
import sys
from pathlib import Path
import analysis as a
import matplotlib.pyplot as plt

PROJECT_ROOT = Path.cwd().parent
sys.path.insert(0, str(PROJECT_ROOT))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

''''
Experimental Model 

 '''
class FibonacciModDataset(Dataset):
    def __init__(self, seq_len=10, mod=10, num_samples=10000):
        self.mod = mod
        self.seq_len = seq_len
        self.global_seq = self.generate_fib_sequence(mod)
        self.samples = []
        """for _ in range(num_samples):
            start_idx = torch.randint(0, len(self.global_seq) - seq_len - 1, (1,)).item()
            seq = self.global_seq[start_idx:start_idx + seq_len + 1]
            x = torch.tensor(seq[:-1], dtype=torch.long)
            y = torch.tensor(seq[1:], dtype=torch.long)
            self.samples.append((x, y))"""
        ''' 
            samples is just list of list, 
            each list of list is a pair generated
        '''

        for i in range(len(self.global_seq)):
            arr = self.global_seq[i]
            x = torch.tensor(arr[:-1], dtype=torch.long)
            y = torch.tensor(arr[1:], dtype=torch.long)
            self.samples.append((x, y))

    """def generate_fib_sequence(self, length, mod):
        seq = [1, 1]
        while len(seq) < length:
            seq.append((seq[-1] + seq[-2]) % mod)
        return seq"""

    def generate_fib_sequence(self, mod):
        all_pairs = [(a, b) for a in range(mod) for b in range(mod)]
        random.shuffle(all_pairs)
        train_pairs = all_pairs[:int(0.75 * len(all_pairs))]

        sequences = []

        for a, b in train_pairs:

            s = [a, b]
            for _ in range(self.seq_len - 1):
                s.append((s[-1] + s[-2]) % mod)

            # # track pairs this sequence adds
            # for i in range(len(s) - 1): # that generated mesh also accouts for seen pairs.
            #     seen_pairs.add((s[i], s[i+1]))

            sequences.append(s)

        return sequences

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class GenerateEvulatePairs(Dataset):

    def __init__(self, dataset, mod, num_samples=1000):
        self.dataset = dataset
        self.mod = mod
        pair_counters = set()
        seq_len = len(self.dataset[0][0])

        for a, b in self.dataset:
            for i in range(seq_len-1):
                x = a[i].item()
                y = a[i+1].item()
                pair_counters.add((x, y))

        all_pairs = {(a, b) for a in range(self.mod) for b in range(self.mod)}
        unseen = list(all_pairs - pair_counters)

        seq_len = len(self.dataset[0][0])
        self.samples = []


        """for a, b in unseen:
            seq = [a, b]
            while len(seq) < seq_len + 1:
                seq.insert(0, (seq[1] - seq[0]) % self.mod)  # since it is a backward loop

            x = torch.tensor(seq[:-1], dtype=torch.long)
            y = torch.tensor(seq[1:], dtype=torch.long)
            self.samples.append((x, y))"""

        for _ in range(num_samples):
            idx = torch.randint(0, len(unseen)-1, (1,)).item()
            a, b = unseen[idx]
            seq = [a, b]
            while len(seq) <= seq_len:
                seq.append((seq[-1] + seq[-2]) % self.mod)

            x = torch.tensor(seq[:-1], dtype=torch.long)
            y = torch.tensor(seq[1:], dtype=torch.long)
            #print(x)
            #print(y)
            self.samples.append((x, y))



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class MLP(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, x):
        return self.net(x)


class MinimalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=4, n_heads=1, num_layers=3, max_seq_len=20, hidden_MLP=128):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        #self.dropout = nn.Dropout(p=0.01) # regulation paramaters
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.0)
            for _ in range(num_layers)
        ])
        self.mlps = nn.ModuleList([
            MLP(d_model, hidden_MLP)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, tokens):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)
        x = self.token_embed(tokens) + self.pos_embed(pos).unsqueeze(0)
        #x = self.dropout(x) # random activations zeroed out
        attn_mask = torch.triu(torch.ones(T, T, device=tokens.device) * float('-inf'), diagonal=1)
        for attn, mlp in zip(self.layers, self.mlps):
            attn_out, _ = attn(x, x, x, attn_mask=attn_mask)
            x = x + attn_out
            x = x + mlp(x)
        return self.out_proj(x)

    def get_embeddings(self):
        return self.pos_embed + self.token_embed


train_plot = []
eval_plot = []


def train_model(model, dataloader, test_loader, epochs=100, lr=0.001, weight_decay=0.01):
    global history, vocab_size, seq_len, batch_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    start_time = time.time()
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            #print(torch.max(F.softmax(logits, dim=-1)))
            loss = loss_fn(logits[:, 1:].reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        """if epoch % monitor_every == 0:
            obs = a.monitor_spectral_observables(
                    model=model,
                    n=vocab_size,
                    seq_len=seq_len,
                    device=device,
                    batch_size=batch_size,
            )
            history["epoch"].append(epoch)
            history["diag_mass"].append(obs["diagonal_spectral_mass"])
            history["xi"].append(obs["correlation_length"])
            history["peak_kx"].append(obs["peak_index"][0])
            history["peak_ky"].append(obs["peak_index"][1])
            print(
            f"step={epoch} "
            f"diag_mass={obs['diagonal_spectral_mass']:.6f} "
            f"xi={obs['correlation_length']:.4f} "
            f"peak={obs['peak_index']}")"""

        avg_loss = total_loss / len(dataloader)
        train_plot.append({'loss': avg_loss, 'epoch': epoch})

        model.eval()
        with torch.no_grad():
            _, val_loss = evaluate_model(model, test_loader)
            eval_plot.append(val_loss)

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss (Training): {avg_loss:.4f} Loss(val): {val_loss:.4f}")

    end_time = time.time()
    print(f"Total Training Time: {end_time - start_time:.2f} seconds")


def evaluate_model(model, dataloader):
    correct, total = 0, 0
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=-1)
            #print(torch.max(F.softmax(logits)))
            loss = loss_fn(logits[:, 1:].reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1))
            total_loss += loss.item()
            correct += (pred[:, 1:] == y[:, 1:]).sum().item()
            total += y[:, 1:].numel()

    return 100 * correct / total, total_loss/len(dataloader)


def plot_history(history):
    steps = history["epoch"]
    plt.figure(figsize=(7, 4))
    plt.plot(steps, history["diag_mass"], marker="o")
    plt.xlabel("training step")
    plt.ylabel("diagonal spectral mass")
    plt.title("Diagonal spectral mass during training")
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(7, 4))
    plt.plot(steps, history["xi"], marker="o")
    plt.xlabel("training step")
    plt.ylabel("correlation length estimator")
    plt.title("Estimated correlation length during training")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    vocab_size = 113
    batch_size = 1000
    d_model = 128
    seq_len = 2
    num_layers = 1
    num_head = 4
    hidden_MLP = 512

    lr = 0.001
    weight_decay = 0.1

    monitor_every = 5
    history = {
        "epoch": [],
        "diag_mass": [],
        "xi": [],
        "peak_kx": [],
        "peak_ky": [],
    }

    train_ds = FibonacciModDataset(num_samples=1000, mod=vocab_size, seq_len=seq_len)

    #train_size = int(0.8 * len(generated_ds))
    #test_size = len(generated_ds) - train_size
    #train_ds, test_ds = random_split(generated_ds, [train_size, test_size])
    test_ds = GenerateEvulatePairs(train_ds, vocab_size, num_samples=500)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)

    model = MinimalTransformer(vocab_size=vocab_size, d_model=d_model, n_heads=num_head,
                               num_layers=num_layers, max_seq_len=seq_len, hidden_MLP=hidden_MLP).to(device)

    #checkpoint_dir = 'checkpoints'
    #file_name = 'dim6_layer3_head2_batch16_seq20_voc10.pth'
    #full_path = os.path.join(checkpoint_dir, file_name)

    epoch = 2000
    try:
        train_model(model, train_loader, epochs=epoch, test_loader=test_loader, lr=lr, weight_decay=weight_decay)
    except KeyboardInterrupt:
        pass

    acc_eval, _ = evaluate_model(model, test_loader)
    acc_train, _ = evaluate_model(model, train_loader)

    print("Accuracy (test set): ", str(acc_eval) + "%")
    print("Accuracy (train set): ", str(acc_train) + "%")

    #plot_history(history)

   # if not os.path.exists(checkpoint_dir):
        #os.makedirs(checkpoint_dir)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'train_loss_history': train_plot,
        'eval_loss_history': eval_plot,
        'epoch': epoch
    }
    #torch.save(checkpoint, full_path)
    #print(f"Successfully saved to: {full_path}")

