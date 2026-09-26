import os
import torch
import transformer as t

'''
Captures Phase and saves the skeletons
'''
class Train:

    def __init__(self, dir: str = "Skeletons", model: t.MinimalTransformer = None):
        self.dir = dir
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = t.TrainConfig
        self.cfg.memorization_check_point_release = True
        self.cfg.memorization_check_point_release_epoch = 3500

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        self.x_star = None


    def start(self, epochs:int = 8500, capture_epoch_mem: int = 3000, capture_epoch_gen: int = 8000):

        torch.manual_seed(self.cfg.torch_seed)

        train_ds = t.FibonacciTrainDataset(
            mod=self.cfg.p, seq_len=self.cfg.seq_len, train_frac=self.cfg.train_frac,
            seed=self.cfg.data_seed,
        )
        val_ds = t.FibonacciValDataset(train_ds, num_samples=500)

        n_unseen = sum(
            1 for a in range(self.cfg.p) for b in range(self.cfg.p)
            if (a, b) not in train_ds.seen_pairs
        )

        self.cfg.batch_size = len(train_ds)
        self.cfg.epochs = epochs

        train_loader = t.DataLoader(
            train_ds, batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=4, pin_memory=True, persistent_workers=True,
        )
        val_loader = t.DataLoader(
            val_ds, batch_size=self.cfg.batch_size,
            num_workers=4, pin_memory=True, persistent_workers=True,
        )

        model = t.MinimalTransformer(
            vocab_size=self.cfg.vocab_size, d_model=self.cfg.d_model, n_heads=self.cfg.n_heads,
            num_layers=self.cfg.num_layers, max_seq_len=self.cfg.seq_len,
            hidden_mlp=self.cfg.hidden_mlp,
        ).to(self.device)

        self.cfg.weight_decay = 1
        self.cfg.checkpoint_path = os.path.join(self.dir, f'training_checkpoint.pth')


        try:
            history = t.train_model(model, train_loader, val_loader, self.cfg)
        except KeyboardInterrupt:
            history = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc", "spectral")}

        val_loss, val_acc = t.evaluate(model, val_loader)
        train_loss, train_acc = t.evaluate(model, train_loader)

        print(f"\nFinal — train acc: {train_acc:.3f}  val acc: {val_acc:.3f}")

        if self.cfg.checkpoint_path:
            t.save_checkpoint(model, history, self.cfg.checkpoint_path)


capture = Train()
capture.start()
