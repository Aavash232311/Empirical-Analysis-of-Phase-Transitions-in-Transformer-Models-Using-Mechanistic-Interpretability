import math
import time
import os.path
import random
import torch
import threading
import analysis as a
import transformer as t
from torch.nn.utils import vector_to_parameters
from concurrent.futures import ThreadPoolExecutor

'''
Trade off notes: There is a forward pass on a transformer after nudging the value of parameters
by a small amount. Here the loss stays fixed and diagonal spectral mass stays changed.

'''
class WangLandau:
    def __init__(self, skeleton_path_grokked: str, skeleton_path_memorized: str, binary_d_mass_path:str="Binary") -> None:
        self.skeleton_path_grokked = skeleton_path_grokked
        self.skeleton_path_memorized = skeleton_path_memorized

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.skeletons_grokked = torch.load(self.skeleton_path_grokked, map_location=self.device)
        self.skeleton_memorized = torch.load(self.skeleton_path_memorized, map_location=self.device)

        self.theta_grokked = []
        self.theta_mem = []

        self.cfg = t.TrainConfig

        self.model = t.MinimalTransformer(
            vocab_size=self.cfg.vocab_size, d_model=self.cfg.d_model, n_heads=self.cfg.n_heads,
            num_layers=self.cfg.num_layers, max_seq_len=self.cfg.seq_len,
            hidden_mlp=self.cfg.hidden_mlp,
        ).to(self.device)

        self.rng = torch.Generator()
        self.rng.manual_seed(42)

        self.binary_d_mass_path = binary_d_mass_path

        if not os.path.exists(binary_d_mass_path):
            os.makedirs(binary_d_mass_path)


    def load_model(self):
        state_dict = self.skeletons_grokked['model_state_dict']
        self.theta_grokked = torch.cat([p.flatten() for p in state_dict.values()]).detach().to(self.device)

        state_dict = self.skeleton_memorized['model_state_dict']
        self.theta_mem = torch.cat([p.flatten() for p in state_dict.values()]).detach().to(self.device)

        d_mass, loss, new_theta = wang.precompute_d_mass_landscape(50, 50, self.theta_grokked)

        d_mass.numpy().tofile(os.path.join(self.binary_d_mass_path, "grokked_diagonal_mass.bin"))
        loss.numpy().tofile(os.path.join(self.binary_d_mass_path, "grokked_loss.bin"))
        new_theta.cpu().numpy().tofile(os.path.join(self.binary_d_mass_path, "nudged_theta.bin"))

    def _nudge_theta(self, parameters: torch.Tensor) -> torch.Tensor:
        mean = 0.0
        std = 0.5
        noise = mean + std * torch.randn_like(parameters)   
        return parameters + noise
    

    def precompute_d_mass_landscape(self, sweeps: int, step: int, parameters, num_threads: int = 8):
        start_time = time.perf_counter()
        fc = sweeps * step
        d_mass_list = torch.zeros(fc)
        loss_list = torch.zeros(fc)

        thread_local = threading.local()
        streams = [torch.cuda.Stream(device=self.device) for _ in range(num_threads)]

        def get_thread_model():
            # each thread builds its own model ONCE, reused across all its assigned iterations
            # careful this is where race condition can happen
            if not hasattr(thread_local, "model"):
                thread_local.model = t.MinimalTransformer(
                    vocab_size=self.cfg.vocab_size, d_model=self.cfg.d_model, n_heads=self.cfg.n_heads,
                    num_layers=self.cfg.num_layers, max_seq_len=self.cfg.seq_len,
                    hidden_mlp=self.cfg.hidden_mlp,
                ).to(self.device)
            return thread_local.model

        def worker(i: int):
            model = get_thread_model()
            stream = streams[i % num_threads]  # round-robin stream assignment

            new_theta = self._nudge_theta(parameters)
            vector_to_parameters(new_theta, model.parameters())

            with torch.cuda.stream(stream):
                with torch.no_grad():

                    L, loss = a.collect_state_logits(
                        model=model, n=self.cfg.p, device=self.device,
                        batch_size=self.cfg.batch_size, eq_token=self.cfg.p,
                        seq_len=self.cfg.seq_len
                    )
                    S, _, _ = a.structure_factor_from_logits(L)
                    m_diag = a.diagonal_spectral_mass(S)

            stream.synchronize()  # sync
            return i, m_diag.item(), loss.item(), new_theta

        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            for count, (i, val, loss, new_theta) in enumerate(pool.map(worker, range(fc)), start=1):
                d_mass_list[i] = val
                loss_list[i] = loss
                if count % 25 == 0 or count == fc:
                    print(f"[{count}/{fc}] Diagonal Spectral Mass: {val:.4f}")

        end_time = time.perf_counter()

        print(f"Execution time: {(end_time - start_time) / 60:.2f} Minutes")
        return d_mass_list, loss_list, new_theta




    def wang_landau(self):

        ''' WangLandau configurations '''

        n_bins = 2

        g = torch.zeros(n_bins)
        h = torch.ones(n_bins)

        x_star  = 0.04   
        window = 1e-4 
        sweeps = 50
        steps_per_sweep = 50


        mod_factor = math.e  

        with open(os.path.join(self.binary_d_mass_path, "grokked_diagonal_mass.bin"), "rb") as fh:
            d_mass_bin = fh.read()
        with open(os.path.join(self.binary_d_mass_path, "nudged_theta.bin"), "rb") as fh:
            nudged_theta_bin = fh.read()
        with open(os.path.join(self.binary_d_mass_path, "grokked_loss.bin"), "rb") as fh:   
            loss_bin = fh.read()

        d_mass_list = torch.frombuffer(bytearray(d_mass_bin), dtype=torch.float32)
        nudged_theta = torch.frombuffer(bytearray(nudged_theta_bin), dtype=torch.float32)
        loss_list = torch.frombuffer(bytearray(loss_bin), dtype=torch.float32)

        def bin_of(y_value: float, n_bins: int) -> int:
            b = int(y_value * n_bins)          # scale [0,1] into n_bins buckets, THEN truncate
            return max(0, min(b, n_bins - 1))

        rng = random.Random(12345)
        pool_size = d_mass_list.shape[0]

        for sweep in range(sweeps):
            for step in range(steps_per_sweep):
                flat_step = sweep * steps_per_sweep + step

                proposed_idx = (flat_step) % pool_size

                # In the smae phase of loss in both the phases what is the diag-m we are only taking if we have certain range of loss.
                # If CUDA, filter this outside the loop. Then thrink the iteration.
                # here mg_diag is the energy domain

                if abs(loss_list[proposed_idx].item() - x_star) > window:
                    continue

                y_new = d_mass_list[proposed_idx].item()
                x_new = bin_of(y_new, n_bins)

                acc = min(1.0, math.exp(g[x] - g[x_new]))
                if rng.random() < acc:
                    current_idx, x = proposed_idx, x_new

                g[x] += mod_factor
                h[x] += 1

wang = WangLandau("Skeletons/training_checkpoint.pth", "Skeletons/memorization_phase.pth")
# wang.load_model()
wang.wang_landau()
