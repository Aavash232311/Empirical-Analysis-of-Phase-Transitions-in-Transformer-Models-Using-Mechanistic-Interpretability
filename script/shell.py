import os
import time
import random
import torch
import torch.nn as nn
from .analysis.analysis_dft import Analysis
from torch.utils.data import Dataset, DataLoader
# from utils.load_pipeline import GenerateEvulatePairs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ''' The hyperparamaters in which we got perfect accuracy '''
# class FibonacciModDataset(Dataset):
#     def __init__(self, seq_len=10, mod=10):
#         self.mod = mod
#         self.seq_len = seq_len
#         self.global_seq = self.generate_fib_sequence(mod=mod)

#         self.samples = []
#         ''' 
#             samples is just list of list, 
#             each list of list is a pair generated
#         '''

#         for _ in range(len(self.global_seq)):
#             arr = self.global_seq[_]
#             x = torch.tensor(arr[:-1], dtype=torch.long)
#             y = torch.tensor(arr[1:], dtype=torch.long)
#             self.samples.append((x, y))

#     def generate_fib_sequence(self, mod):
#         all_pairs = [(a, b) for a in range(mod) for b in range(mod)]

#         train_pairs = all_pairs[:int(0.2 * len(all_pairs))] 
    
#         sequences = []

#         for a, b in train_pairs:
                
#             s = [a, b]
#             for _ in range(self.seq_len - 1):
#                 elem = (s[-1] + s[-2]) % mod
#                 s.append(elem)
            
#             sequences.append(s)

#         return sequences


#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         return self.samples[idx]


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

        print(f"Unseen pairs {len(unseen)}")
        print(f"Seen pairs {len(pair_counters)}")



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# old- with GELU
# class MLP(nn.Module):
#     def __init__(self, d_model, hidden_layer):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(d_model, hidden_layer),
#             nn.GELU(),
#             nn.Linear(hidden_layer, hidden_layer),
#             nn.GELU(),
#             nn.Linear(hidden_layer, d_model)
#         )

#     def forward(self, x):
#         return self.net(x)


class MLP(nn.Module):
    def __init__(self, d_model, hidden_layer):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, d_model),
        )

    def forward(self, x):
        return self.net(x)

class MinimalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, num_layers=1, max_seq_len=20):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            for _ in range(num_layers)
        ])

        self.mlps = nn.ModuleList([
            MLP(d_model, hidden_layer=512)
            for _ in range(num_layers)
        ])
  

        self.out_proj = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size
        # self.input_proj = nn.Linear(2, d_model)
        self.d_model = d_model

    def forward(self, tokens):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)              
        x = self.token_embed(tokens) + self.pos_embed(pos).unsqueeze(0)   

        attn_mask = torch.triu(torch.ones(T, T, device=tokens.device) * float('-inf'), diagonal=1)
        # Here this is supposed to be without MLP to experiment
        for attn, mlp in zip(self.layers, self.mlps):
            attn_out, _ = attn(x, x, x, attn_mask=attn_mask)
            x = x + attn_out
            x = x + mlp(x)
        return self.out_proj(x)
    
    def get_embeddings(self):
        return self.pos_embed + self.token_embed
    


train_plot = []
eval_plot = []
epoch_masses = []

train_accuracy = []
test_accuracy = []

checkpoint_dir = 'checkpoints/bin'
file_name = f'batch_full_dimension_128_head_4_wd_0.09_with_mlp.pth'
full_path  = os.path.join(checkpoint_dir, file_name)

def train_model(model, dataloader, test_loader, epochs=12, lr=0.001, weight_decay=0.07):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = nn.CrossEntropyLoss()
    start_time = time.time()

    analysis = Analysis(vocab_size=model.vocab_size) 

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        correct = 0
        total = 0
        batch_d_masses = []

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits[:, 1:].reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1)) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            ''' For training accuracy. '''
            pred = logits.argmax(dim=-1)

            correct += (pred[:, 1:] == y[:, 1:]).sum().item() 
            total += y[:, 1:].numel() 
            ''' Diagonal spectral mass '''
            d_mass = analysis.diagonal_sperectal_mass(preferred_logits=logits, x=x)
            batch_d_masses.append(d_mass)

        avg_loss = total_loss / len(dataloader)
        train_plot.append({'loss': avg_loss, 'epoch': epoch})
        epoch_masses.append(sum(batch_d_masses) / len(batch_d_masses))

        model.eval()
        with torch.no_grad():
            ''' evulate_model returns the loss and accuracy per epoch. '''
            val_loss, eval_accuracy = evaluate_model(model, test_loader, show_accuracy=False) 

            
            ''' Here we have accuray per ecpoh so we can append that in train accuracy '''
            test_accuracy.append({"epoch": epoch, "test_accuracy": eval_accuracy}) # for testing accuracy.
            eval_plot.append(val_loss)

        model.train()
        avg_loss = total_loss / len(dataloader)
        train_accuracy.append((correct / total) * 100)

        if (epoch + 1) % 50 == 0:
            if not os.path.exists(checkpoint_dir): 
                os.makedirs(checkpoint_dir)

            checkpoint = {
                'model_state_dict': model.state_dict(),
                'train_loss_history': train_plot,
                'eval_loss_history': eval_plot,
                'epoch': epoch,
                'd_model': model.d_model,
                'test_accuracy': test_accuracy,
                'train_accuracy': train_accuracy,
                 "d_mass": epoch_masses,
                'vocab_size': model.vocab_size,
                'weight_decay': weight_decay
            }
            torch.save(checkpoint, full_path) 
            print(f"Successfully saved to: {full_path}", "- " * 20, "Checkpoint saved")
            print("10 interval")

        current_wd = optimizer.param_groups[0]['weight_decay']
        current_lr = optimizer.param_groups[0]['lr']


        # if avg_loss < 0.05 and current_wd == 0.0:  # let it memorize first.
        #     for g in optimizer.param_groups:
        #         g['weight_decay'] = 1.9
        #         g['lr'] = 0.015
        #     print("Switched to weight decay phase")

        # if epoch >= 500:
        #     for g in optimizer.param_groups:
        #         g['lr'] = 0.00001

        
        print(f"Epoch {epoch+1}, Loss (Training): {avg_loss:.4f} Loss(val): {val_loss:.4f} Learning rate(eta): {current_lr:.10f} weight decay {current_wd} Eval accuracy: {eval_accuracy:.2f}%")


    end_time = time.time()
    print(f"Total Training Time: {end_time - start_time:.2f} seconds")


def evaluate_model(model, dataloader, show_accuracy=False):
    correct, total = 0, 0
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device, dtype=torch.long), y.to(device, dtype=torch.long)
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



def execute():
    vocab_size = 113
    epoch = int(input("Enter number of epoch: "))

    total_accuray = 0
    # generated_ds = FibonacciModDataset(mod=vocab_size, seq_len=2)
    # eval_ds = GenerateEvulatePairs(generated_ds, mod=vocab_size)

    # train_loader = DataLoader(
    #     generated_ds, 
    #     batch_size=len(generated_ds),  # Full batch loading.
    #     shuffle=False, 
    #     num_workers=0,   
    #     pin_memory=True
    # )

    # test_loader = DataLoader(
    #     eval_ds, 
    #     batch_size=len(eval_ds),     
    #     shuffle=False, 
    #     num_workers=0,   
    #     pin_memory=True
    # )


    seq_len=2
    train_ds = FibonacciModDataset(num_samples=1000, mod=vocab_size, seq_len=seq_len)

    #train_size = int(0.8 * len(generated_ds))
    #test_size = len(generated_ds) - train_size
    #train_ds, test_ds = random_split(generated_ds, [train_size, test_size])
    test_ds = GenerateEvulatePairs(train_ds, vocab_size, num_samples=500)


    train_loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
   
    model = MinimalTransformer(vocab_size=vocab_size).to(device)

    # ''' 
    # We have made the model get high accuracy on limited possible resources, now we need to save the checkpoint in order to save time.
    # '''
    weight_decay = 0.09

    # def unique_pairs(dataset):
    #     pair_counters = set()
    #     seq_len = len(dataset[0][0])

    #     for a, b in dataset:
    #         for i in range(seq_len-1):
    #             x = a[i].item()
    #             y = a[i+1].item()
    #             pair_counters.add((x, y))   

    #     return pair_counters


    # train_pairs = unique_pairs(generated_ds)
    # evulate_pairs = unique_pairs(eval_ds)

    # common_factor = train_pairs & evulate_pairs # if common factor is zero, then it is a binary task, there force it is mathematically impossible to have non-repetance on the fib sequence we are generating.
    # print(f"Common factor {len(common_factor)}")


    train_model(model=model, dataloader=train_loader, epochs=epoch, test_loader=test_loader, weight_decay=weight_decay) 
    evaluate_model(model=model, dataloader=test_loader, show_accuracy=True)


    if not os.path.exists(checkpoint_dir): # if this does not exists for some reason then create one
        os.makedirs(checkpoint_dir)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'train_loss_history': train_plot,
        'eval_loss_history': eval_plot,
        'epoch': epoch,
        'total_accuracy': total_accuray,
        'd_model': model.d_model,
        "d_mass": epoch_masses,
        'test_accuracy': test_accuracy,
        'train_accuracy': train_accuracy,
        'vocab_size': vocab_size,
        'weight_decay': weight_decay

    }
    torch.save(checkpoint, full_path) 
    print(f"Successfully saved to: {full_path}")

if __name__ == "__main__":
    execute()
    
