import os
import sys
import time
import torch
import random
import torch.nn as nn
from pathlib import Path
from script.analysis.analysis_dft import Analysis
from torch.utils.data import Dataset, DataLoader, random_split

PROJECT_ROOT = Path.cwd().parent
sys.path.insert(0, str(PROJECT_ROOT))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
''''
Experimental Model 

 '''


class FibonacciModDataset(Dataset):
    def __init__(self, seq_len=10, mod=10, num_samples=10000):
        self.mod = mod
        self.global_seq = self.generate_fib_sequence(30000, mod)
        self.samples = []
        for _ in range(num_samples):
            start_idx = torch.randint(0, len(self.global_seq) - seq_len - 1, (1,)).item()
            seq = self.global_seq[start_idx:start_idx + seq_len + 1]
            x = torch.tensor(seq[:-1], dtype=torch.long)
            y = torch.tensor(seq[1:], dtype=torch.long)
            self.samples.append((x, y))

    def generate_fib_sequence(self, length, mod):
        seq = [1, 1]
        while len(seq) < length:
            seq.append((seq[-1] + seq[-2]) % mod)
        return seq

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
''' This one is a smaller shample, not 'smaller' but will see more unique paris '''
class FibonacciModDatasetSmallShample(Dataset):
    def __init__(self, seq_len=10, mod=10, num_samples=10000, seeds=None):
        self.mod = mod
        self.samples = []

        if seeds is None:
            seeds = [(i, j) for i in range(mod) for j in range(mod)]

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


class MLP(nn.Module):
    def __init__(self, d_model, expansion=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Linear(d_model * expansion, d_model),
        )

    def forward(self, x):
        return self.net(x)


class MinimalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=2, num_layers=2, max_seq_len=20):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.01)
            for _ in range(num_layers)
        ])
        self.mlps = nn.ModuleList([
            MLP(d_model)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, tokens):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)
        x = self.token_embed(tokens) + self.pos_embed(pos).unsqueeze(0)

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
epoch_d_masses = []
epoch_masses = []

def train_model(model, dataloader, test_loader, epochs=12, lr=0.001, decay=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    loss_fn = nn.CrossEntropyLoss()
    start_time = time.time()

    ''' For diagonal spectral mass let's analyse that class here '''
    analysis = Analysis(vocab_size=model.vocab_size) 

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_d_masses = [] 
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits[:, 1:].reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            ''' Diagonal spectral mass '''
            d_mass = analysis.diagonal_sperectal_mass(preferred_logits=logits, x=x)
            batch_d_masses.append(d_mass)
   
        epoch_masses.append(sum(batch_d_masses) / len(batch_d_masses))
        avg_loss = total_loss / len(dataloader)
        train_plot.append({'loss': avg_loss, 'epoch': epoch})

        model.eval()
        with torch.no_grad():
            ''' Wish we had something like 'out' like in c#, or I am not sure here '''
            val_loss, total_acc = evaluate_model(model, test_loader, show_accuracy=False) 
            eval_plot.append(val_loss)

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss (Training): {avg_loss:.4f} Loss(val): {val_loss:.4f}")

    end_time = time.time()
    print(f"Total Training Time: {end_time - start_time:.2f} seconds")


def evaluate_model(model, dataloader, show_accuracy=False):
    correct, total = 0, 0
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=-1)
            loss = loss_fn(logits[:, 1:].reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1))
            total_loss += loss.item()
            correct += (pred[:, 1:] == y[:, 1:]).sum().item()
            total += y[:, 1:].numel()

    total_accuracy = f"{correct / total:.2%}"
    if show_accuracy:
        print(f"Accuracy (eval mode): {total_accuracy}")

    avg_loss = total_loss / len(dataloader)
    total_accuracy = (correct / total) * 100
    return avg_loss, total_accuracy

vocab_size = 10
batch_size = 128
seq_len = 20
generated_ds = FibonacciModDatasetSmallShample(num_samples=25000, mod=vocab_size, seq_len=seq_len)


train_size = int(0.8 * len(generated_ds))
test_size = len(generated_ds) - train_size
train_ds, test_ds = random_split(generated_ds, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)

model = MinimalTransformer(vocab_size=vocab_size).to(device)


if __name__ == "__main__":

    file_name = 'dim_4_layer_3.pth'
    if isinstance(generated_ds, FibonacciModDatasetSmallShample):
        file_name = "dim_4_layer_3_alterset.pth"

    checkpoint_dir = 'checkpoints'
    full_path = os.path.join(checkpoint_dir, file_name)

    epoch = 22
    total_accuray = 0 # this is for total evulate accuracy
    try:
        train_model(model, train_loader, epochs=epoch, test_loader=test_loader)
        avg_loss, eval_accuracy  = evaluate_model(model, test_loader, show_accuracy=True)
        total_accuray = eval_accuracy
    except KeyboardInterrupt:
        pass
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'train_loss_history': train_plot,
        'eval_loss_history': eval_plot,
        'epoch': epoch,
        'total_accuracy': total_accuray, # Just to keep track of what we are evulating
        'd_model': model.d_model,
        'd_mass': epoch_d_masses
    }
    torch.save(checkpoint, full_path)
    print(f"Successfully saved to: {full_path}")
