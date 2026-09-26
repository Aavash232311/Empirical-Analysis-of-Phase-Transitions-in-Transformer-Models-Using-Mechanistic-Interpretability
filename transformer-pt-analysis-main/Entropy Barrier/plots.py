import os
import torch
import transformer as t


path = os.path.join("Skeletons", "memorization_phase.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = t.TrainConfig

model = t.MinimalTransformer(
    vocab_size=cfg.vocab_size, d_model=cfg.d_model, n_heads=cfg.n_heads,
    num_layers=cfg.num_layers, max_seq_len=cfg.seq_len,
    hidden_mlp=cfg.hidden_mlp,
).to(device)

skeletons = torch.load(path, map_location=device)

history = skeletons['history']

xi = [pt['xi'] for pt in history['spectral']]
peak = [pt['peak'] for pt in history['spectral']]

t.plot_training_curves(history, title=f"d={cfg.d_model} P={cfg.p}")
t.plot_spectral_history(history, title=f"Diagonal spectral mass during training seed")


