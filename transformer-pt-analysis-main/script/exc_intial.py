import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

''' 
This is for analysis on inital embedding on same hyperparamasters that we did on shell.py

'''


class FibonacciModDataset(Dataset):
    def __init__(self, seq_len=10, mod=10, num_samples=10000):
        self.mod = mod

        self.global_seq = self.generate_fib_sequence(1000, mod)
        self.samples = []
        for _ in range(num_samples):
            start_idx = torch.randint(0, len(self.global_seq) - seq_len - 1, (1,)).item()
            seq = self.global_seq[start_idx:start_idx + seq_len + 1]
            x = torch.tensor(seq[:-1], dtype=torch.long)
            y = torch.tensor(seq[1:], dtype=torch.long)
            self.samples.append((x, y))

    def generate_fib_sequence(self, length, mod):
        seq = [1, 1] # these are the starting values
        while len(seq) < length: # this prevents overlap
            seq.append((seq[-1] + seq[-2]) % mod)
        return seq

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class MinimalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=8, n_heads=2, num_layers=1, max_seq_len=20):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, tokens):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)
        x = self.token_embed(tokens) + self.pos_embed(pos).unsqueeze(0)
        attn_mask = torch.triu(torch.ones(T, T, device=tokens.device) * float('-inf'), diagonal=1)
        for attn in self.layers:
            attn_out, _ = attn(x, x, x, attn_mask=attn_mask)
            x = x + attn_out
        return self.out_proj(x)
    
    def get_embeddings(self):
        return self.pos_embed + self.token_embed
    


train_plot = []
eval_plot = []
def train_model(model, dataloader, test_loader, epochs=12, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train() # because we have that eval mode after each epoch
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            loss = loss_fn(logits[:, 1:].reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        train_plot.append({'loss': avg_loss, 'epoch': epoch})  
        
        model.eval()
        with torch.no_grad():
            val_loss = evaluate_model(model, test_loader)
            eval_plot.append(val_loss)
    
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss (Training): {avg_loss:.4f} Loss(val): {val_loss:.4f}")
    
    end_time = time.time()
    print(f"Total Training Time: {end_time - start_time:.2f} seconds")


def evaluate_model(model, dataloader, show_accuracy = False):
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

    if show_accuracy:
      print(f"Accuracy (eval mode): {correct / total:.2%}")

    avg_loss = total_loss / len(dataloader) 
    return avg_loss

if __name__ == "__main__":
    vocab_size = 10
    batch_size = 128
    generated_ds = FibonacciModDataset(num_samples=25000, mod=vocab_size, seq_len=20)



    train_size = int(0.8 * len(generated_ds))
    test_size = len(generated_ds) - train_size
    train_ds, test_ds = random_split(generated_ds, [train_size, test_size]) 

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)


    model = MinimalTransformer(vocab_size=vocab_size).to(device)

    checkpoint_dir = 'checkpoints'
    file_name = 'v1_initial.pth'
    full_path  = os.path.join(checkpoint_dir, file_name)

    if not os.path.exists(checkpoint_dir): # if this does not exists for some reason then create one
        os.makedirs(checkpoint_dir)

    torch.save(model.state_dict(), full_path) # save it right there
    print(f"Successfully saved to: {full_path}")
