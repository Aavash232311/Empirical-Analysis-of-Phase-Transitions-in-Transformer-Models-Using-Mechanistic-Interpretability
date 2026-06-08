import torch
from torch.utils.data import Dataset, DataLoader



''' Use this if you have the pairs, to load it into a dataset '''
class AdhocTestDataset(Dataset):
    def __init__(self, sequences):
        self.samples = sequences  
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

'''  From x, y torch array for input output to data loader object.'''
def get_dataset(seq, batch_size, shuffle): # leave this fasle in eval.
    ds = AdhocTestDataset(seq)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def build_sequence(a, b, mod=10, seq_len=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq = [a, b] 

    while len(seq) < seq_len + 1:
        seq.append((seq[-1] + seq[-2]) % mod)

    # problem with this slicing. 
    x = torch.tensor(seq[:-1], dtype=torch.long).to(device=device)
    y = torch.tensor(seq[1:], dtype=torch.long).to(device=device)

    return x, y

''' from (a, b) array pairs retruns x, and y '''
def pairs(arr, mod, seq_len):
    res = []
    for a, b in arr:
        x, y = build_sequence(a, b, mod=mod, seq_len=seq_len)
        res.append((x, y))
    return res


''' This get's the missing pairs out of total possible combinations. '''
def get_missing_pairs(mod=10, seq_length=1000):
    seq = [1, 1]
    while len(seq) < seq_length: # here just normal stuff, of generating fib
        seq.append((seq[-1] + seq[-2]) % mod)
    
    found = set()
    for i in range(len(seq) - 1): # here we do not one dublicates in set.
        found.add((seq[i], seq[i+1]))
    
    all_pairs = {(a, b) for a in range(mod) for b in range(mod)} # here we have all the possible combinations that could be found
    missing = all_pairs - found
    
    print(f"missing pairs {len(missing)} total")
    print(f"non missing pairs: {len(found)}")
    for p in sorted(missing):
        print(f"  {p}")
    return found, missing


def generate_pairs_start_point(x, y, mod=10, seq_length=20):

    seq = [x, y]

    while len(seq) < seq_length:
        seq.append((seq[-1] - seq[-2]) % mod)
    
    points = []
    for i in range(0, len(seq)):
        fa = seq[i]

        if i + 1 >= len(seq):
            break

        fb = seq[i + 1]
        points.append((fa, fb))

        if fa == seq[0] and fb == seq[1] and i > 0:
            break # end of the loop

    return points


''' We need to get the dataset with the unqiue pairs build backwards.
    Unseen pairs should be 40 no matter the datalength,

 '''
class GenerateEvulatePairs(Dataset):

    def __init__(self, dataset, mod):
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
        print(f"Total {len(all_pairs)}")
        print(f"seen {len(pair_counters)}")
        print(f"Unseen {len(unseen)}")

        seq_len =  len(self.dataset[0][0])
        self.samples = [] # this returns the unseen pairs. 
        print(f"Seq len {seq_len}")


        for a, b in unseen:
            seq = [a, b]
            while len(seq) < seq_len + 1:
                seq.append((seq[-1] + seq[-2]) % self.mod)
            
            x = torch.tensor(seq[:-1], dtype=torch.long)
            y = torch.tensor(seq[1:],  dtype=torch.long)
            self.samples.append((x, y))


    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

