import json
import time
import shutil
import contextlib
import threading

import os
import torch
from torch.nn.utils import vector_to_parameters, parameters_to_vector

import math
import random
import analysis as a
import transformer as t
import numpy as np
from concurrent.futures import ThreadPoolExecutor


class WangLandau:

    def __init__(
        self,
        skeleton_path_grokked: str,
        skeleton_path_memorized: str,
        save_path_sk: str = "/media/avash/New Volume/Binary_dump_phase"

    ) -> None:
        self.skeleton_path_grokked = skeleton_path_grokked
        self.skeleton_path_memorized = skeleton_path_memorized

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cfg = t.TrainConfig

        # Skeletons of grokked models
        self.skeletons_grokked = torch.load(self.skeleton_path_grokked, map_location=self.device)
        self.skeleton_memorized = torch.load(self.skeleton_path_memorized, map_location=self.device)

        # Instance of transformer
        self.model = t.MinimalTransformer(
            vocab_size=self.cfg.vocab_size, d_model=self.cfg.d_model, n_heads=self.cfg.n_heads,
            num_layers=self.cfg.num_layers, max_seq_len=self.cfg.seq_len,
            hidden_mlp=self.cfg.hidden_mlp,
        ).to(self.device)
        self.model.eval()

        # Paramaters of model, between two phases
        # FIX: theta in model.parameters() order (no buffers, no duplicated tied weights),
        # so it lines up exactly with vector_to_parameters.
        self.model.load_state_dict(self.skeletons_grokked['model_state_dict'])
        self.theta_grokked = parameters_to_vector(self.model.parameters()).detach().clone()

        self.model.load_state_dict(self.skeleton_memorized['model_state_dict'])
        self.theta_memorized = parameters_to_vector(self.model.parameters()).detach().clone()


        self.seed = 42
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(self.seed)

        self.binary_d_mass_path = save_path_sk

        # FIX: fail NOW (not hours later) if the volume is missing or not writable.
        os.makedirs(self.binary_d_mass_path, exist_ok=True)
        print(f"output folder: {self.binary_d_mass_path} | "
              f"free: {shutil.disk_usage(self.binary_d_mass_path).free / 1e9:.1f} GB | "
              f"n_params: {self.theta_grokked.numel()}")

        self.step = None
        self.sweeps = None


    def _nudge_theta(self, parameters: torch.Tensor, rel_std: float = 0.05, min_scale: float = 1e-3,
                    generator: torch.Generator = None) -> torch.Tensor:
        generator = generator if generator is not None else self.rng
        noise = torch.randn(parameters.shape, generator=generator,
                            device=parameters.device, dtype=parameters.dtype)
        scale = parameters.abs().clamp_min(min_scale)
        return parameters + rel_std * scale * noise

    
    def _save_precomputed_binary(self, tag: str, d_mass, loss, n_params: int):
        # float32 on disk, because wang_landau() reads with dtype=np.float32
        # The theta file is NOT written here: the workers already streamed it to disk.
        sweeps, step = loss.shape

        d_mass.detach().cpu().numpy().astype(np.float32, copy=False).tofile(
            os.path.join(self.binary_d_mass_path, f"{tag}_diagonal_mass.bin"))
        loss.detach().cpu().numpy().astype(np.float32, copy=False).tofile(
            os.path.join(self.binary_d_mass_path, f"{tag}_loss.bin"))

        with open(os.path.join(self.binary_d_mass_path, f"{tag}_shape.json"), "w") as fh:
            json.dump({"sweeps": sweeps, "step": step, "n_params": n_params}, fh)

    def _load_precomputed_binary(self, tag: str):
        with open(os.path.join(self.binary_d_mass_path, f"{tag}_shape.json")) as fh:
            shp = json.load(fh)
        sweeps, step, n_params = shp["sweeps"], shp["step"], shp["n_params"]

        dm = np.fromfile(os.path.join(self.binary_d_mass_path, f"{tag}_diagonal_mass.bin"),
                         dtype=np.float32).reshape(sweeps, step, 1)
        loss = np.fromfile(os.path.join(self.binary_d_mass_path, f"{tag}_loss.bin"),
                           dtype=np.float32).reshape(sweeps, step)
        theta_file = os.path.join(self.binary_d_mass_path, f"{tag}_nudged_theta.bin")
        # memmap: nothing is read until you index it, e.g. th[3, 17] reads one row
        th = (np.memmap(theta_file, dtype=np.float32, mode="r", shape=(sweeps, step, n_params))
              if os.path.exists(theta_file) else None)
        return dm, loss, th



    def precompute_d_mass_landscape(self, sweeps: int, step: int, parameters,
                                num_threads: int = 1, tag: str = None, save_name: str = None):
        # num_threads=1 on CPU: PyTorch already uses all cores inside one forward pass.
        if tag is not None:
            print(tag)
        start_time = time.perf_counter()
        fc = sweeps * step
        n_params = parameters.numel()

        ''' 
            Note this can take up lot of disk space and things can get large really fast.
        '''
        
        # (sweeps, step, 1), This is my proposed idea. Here no custom kernel yet but we will take a look into that if required
        nudged_frame = torch.zeros(sweeps, step, 1) # stores the diagonal mass in that tensor
        # (sweeps, step) This holds a loss value
        loss_grid = torch.zeros(sweeps, step)
        # (sweeps, step, paramaters) This holds paramaters that are nudeged and then we can do the math.
        # FIX: this is no longer a torch.zeros(...) tensor in RAM (9.45 GB at 10,000 samples).
        # It is streamed to DISK row by row with positional writes, so nothing is mapped into RAM
        # (an np.memmap keeps every touched page resident until the kernel writes it back).
        theta_fd = None
        if save_name is not None:
            need = fc * n_params * 4
            free = shutil.disk_usage(self.binary_d_mass_path).free
            print(f"theta file: {need / 1e9:.2f} GB on disk, {free / 1e9:.1f} GB free")
            if need > 0.9 * free:
                raise OSError("Not enough disk space for the theta file.")
            theta_fd = os.open(os.path.join(self.binary_d_mass_path, f"{save_name}_nudged_theta.bin"),
                               os.O_RDWR | os.O_CREAT | os.O_TRUNC)
            os.ftruncate(theta_fd, need)
            row_bytes = n_params * 4

        thread_local = threading.local()

        def get_thread_model():
            if not hasattr(thread_local, "model"):
                # since isntance. Not familier with multi threaded opreation in python
                # but looks like this approach keeps my GPU busy. Not to the full capacity because of the
                # cpu is the limiting factor
                m = t.MinimalTransformer(
                    vocab_size=self.cfg.vocab_size, d_model=self.cfg.d_model, n_heads=self.cfg.n_heads,
                    num_layers=self.cfg.num_layers, max_seq_len=self.cfg.seq_len,
                    hidden_mlp=self.cfg.hidden_mlp,
                ).to(self.device)
                m.load_state_dict(self.skeletons_grokked['model_state_dict'])
                m.eval()
                thread_local.model = m
            return thread_local.model

        use_cuda = self.device.type == "cuda"
        streams = [torch.cuda.Stream(device=self.device) for _ in range(num_threads)] if use_cuda else None

        def worker(i: int):
            s, k = divmod(i, step)                    # i = s*step + k
            model = get_thread_model()
            if use_cuda:
                stream = streams[i % num_threads]
                stream.wait_stream(torch.cuda.current_stream(self.device))
                ctx = torch.cuda.stream(stream)
            else:
                ctx = contextlib.nullcontext()

            with ctx:
                with torch.no_grad():
                    gen = torch.Generator(device=self.device)
                    gen.manual_seed(self.seed + i)
                    new_theta = self._nudge_theta(parameters, generator=gen)
                    vector_to_parameters(new_theta, model.parameters())
                    L, loss = a.collect_state_logits(
                        model=model, n=self.cfg.p, device=self.device,
                        batch_size=self.cfg.batch_size, eq_token=self.cfg.p,
                        seq_len=self.cfg.seq_len)
                    S, _, _ = a.structure_factor_from_logits(L)
                    m_diag = a.diagonal_spectral_mass(S)

                    # each worker writes ONLY its own cell, so no lock is needed
                    if theta_fd is not None:
                        # row [s, k] starts at byte (s*step + k) * row_bytes = i * row_bytes; pwrite is thread-safe
                        data = memoryview(new_theta.detach().cpu().numpy().astype(np.float32, copy=False).tobytes())
                        off = i * row_bytes
                        while len(data):
                            n = os.pwrite(theta_fd, data, off)
                            data, off = data[n:], off + n
                    nudged_frame[s, k, 0] = m_diag.item()
                    loss_grid[s, k] = loss.item()

            if use_cuda:
                stream.synchronize()
            return i, m_diag.item()

        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            for count, (i, val) in enumerate(pool.map(worker, range(fc)), start=1):
                if count % 25 == 0 or count == fc:
                    print(f"[{count}/{fc}] Diagonal Spectral Mass: {val:.4f}")

        if theta_fd is not None:
            os.fsync(theta_fd)
            os.close(theta_fd)

        print(f"Execution time: {(time.perf_counter() - start_time) / 60:.2f} Minutes")
        return nudged_frame, loss_grid, None



    """compute_energy

    Pre-computes the energy and returns the following tesnors.

    Args:
        sweeps: int
        step: int
        store_theta: bool (Stores paramaters)
    Returns:
        d_mass: (sweeps, step, 1)
        loss:   (sweeps, step)
        theta:   (sweeps, step, paramaters)
    
    Raises:
        ValueError: If principal or rate is negative.
    """
    def compute_energy(self, sweeps: int, step: int, store_theta: bool = True):
        # We need to compute this for both memorization and generalization phase so we pass in the paramaters two times.
        n_params = self.theta_grokked.numel()
        if store_theta:
            need = 2 * sweeps * step * n_params * 4
            free = shutil.disk_usage(self.binary_d_mass_path).free
            print(f"theta files for both phases: {need / 1e9:.2f} GB, {free / 1e9:.1f} GB free")
            if need > 0.9 * free:
                raise OSError("Not enough disk space for both theta files; reduce sweeps*step.")


        d_mass_grokked, loss_list_grokked, _ = self.precompute_d_mass_landscape(
            sweeps, step, self.theta_grokked, tag="Computing grokked phase",
            save_name="grokked" if store_theta else None)
        self._save_precomputed_binary("grokked", d_mass_grokked, loss_list_grokked, n_params)

        d_mass_mem, loss_list_mem, _ = self.precompute_d_mass_landscape(
            sweeps, step, self.theta_memorized, tag="Computing memorization phase",
            save_name="mem" if store_theta else None)
        self._save_precomputed_binary("mem", d_mass_mem, loss_list_mem, n_params) # number of paramaters is the same here, we are capturing two phases of the same training.
        # we have something like for new theta, we have d_mass thats the energy

    def _loadBin(self, name: str, shape=None, memmap: bool = False):
        pth = os.path.join(self.binary_d_mass_path, name)

        # catches a stale file from a run with different sweeps/step/model size
        if shape is not None:
            expected = int(np.prod(shape)) * 4
            actual = os.path.getsize(pth)
            if actual != expected:
                raise ValueError(f"{name}: file has {actual} bytes, shape {shape} needs {expected}")

        if memmap:
            # here we have to do lazy loading these tensor are huge in size.
            return np.memmap(pth, dtype=np.float32, mode="r", shape=shape)

        data = np.fromfile(pth, dtype=np.float32)
        if shape is not None:
            data = data.reshape(shape)
        return torch.from_numpy(data).to(self.device)

    def load_tensors(self, tag: str, load_theta: bool = False):
        # tag is "grokked" or "mem"
        with open(os.path.join(self.binary_d_mass_path, f"{tag}_shape.json")) as fh:
            shp = json.load(fh)
        sweeps, step, n_params = shp["sweeps"], shp["step"], shp["n_params"]

        d_mass = self._loadBin(f"{tag}_diagonal_mass.bin", (sweeps, step, 1))
        loss = self._loadBin(f"{tag}_loss.bin", (sweeps, step))
        theta = (self._loadBin(f"{tag}_nudged_theta.bin", (sweeps, step, n_params), memmap=True)
                if load_theta else None)
        return d_mass, loss, theta


    """inspect_loss_window

    For inspection of saved tensors. Tensor can get really large in space.

    Args:
        None

    Returns:
        None
    
    Raises:
        ValueError: If principal or rate is negative.
    """
    def inspect_loss_window(self):
        x_star = 0.04
        window = 5e-2  # 0.05

        d_mass_grokked, loss_grokked, theta_grokked = self.load_tensors("grokked", load_theta=True)
        d_mass_mem, loss_mem, theta_mem = self.load_tensors("mem", load_theta=True)
    

    """inspectModels

    For inspection

    Args:
        None

    Returns:
        None
    
    Raises:
        None
    """
    def inspectModels(
            self 
    ):
        # Utility method to plot and inspect two phases 

        history_grokked = self.skeletons_grokked['history']

        t.plot_training_curves(history_grokked, title=f"p={self.cfg.p}, d={self.cfg.d_model} Grokked Model")
        t.plot_spectral_history(history_grokked)

        # for ungrokked model
        history_ungrokked = self.skeleton_memorized['history']
        
        t.plot_training_curves(history_ungrokked, title=f"p={self.cfg.p}, d={self.cfg.d_model} Un-grokked Model")
        t.plot_spectral_history(history_ungrokked)

    def wanglandau(self, tag: str, sweeps: int = 200, steps_per_sweep: int = 50,
                x_star: float = 0.04, window: float = 5e-2, n_bins: int = 20,
                flat: float = 0.8, ln_f_final: float = 1e-6, seed: int = 12345):

        
        d_mass, loss, _ = self.load_tensors(tag, load_theta=False)  
        d_mass = d_mass.reshape(-1).cpu().numpy()    # flat, in sample order i
        loss = loss.reshape(-1).cpu().numpy() # flat strip cont

        # energy shell: only samples whose loss is close to x_star (like in the notes)
        shell = np.flatnonzero(np.abs(loss - x_star) <= window)

        print(f"{shell.size}/{loss.size} samples inside the loss shell")
        if shell.size < 10 * n_bins:
            raise ValueError(f"Only {shell.size} samples in the shell; need >= {10 * n_bins}. "
                            "Widen `window` or change `x_star`.")

        # bin edges come from the data, because the diagonal mass may sit in a very narrow range
        m = d_mass[shell]

        edges = np.linspace(m.min(), m.max(), n_bins + 1)
        bins = np.clip(np.searchsorted(edges, m, side="right") - 1, 0, n_bins - 1)
        counts = np.bincount(bins, minlength=n_bins)
        present = np.flatnonzero(counts)             # bins that can actually be visited

        if present.size < 2:
            raise ValueError("All in-shell samples fall in one bin.")

        ln_g = np.zeros(n_bins)                      # ln g(bin)
        h = np.zeros(n_bins)                         # histogram
        ln_f = 1.0                                   # modification factor, halved when h is flat
        rng = random.Random(seed)

        cur = rng.randrange(shell.size)              # current state = an index into the shell
        x = int(bins[cur])

        for sweep in range(sweeps):
            for k in range(steps_per_sweep):
                prop = rng.randrange(shell.size)     # uniform random proposal
                x_new = int(bins[prop])

                d = ln_g[x] - ln_g[x_new]            # ln[g(x) / g(x_new)] check asgains this ratio
                if d >= 0.0 or rng.random() < math.exp(d):
                    cur, x = prop, x_new

                ln_g[x] += ln_f                      # update the CURRENT bin, accepted or not
                h[x] += 1

            hp = h[present]
            if hp.min() >= flat * hp.mean():         # flat histogram: refine
                ln_f *= 0.5
                h[:] = 0
                if ln_f < ln_f_final:
                    break
        else:
            print(f"WARNING: not converged, ln f = {ln_f:.2e}. Increase sweeps.")

        # normalise so that sum_b g_b = number of in-shell samples
        ln_g[present] -= np.logaddexp.reduce(ln_g[present])
        ln_g[present] += math.log(shell.size)
        ln_g[counts == 0] = -np.inf
        return edges, ln_g, counts                   # S(bin) = ln_g (k_B = 1)

                

if __name__ == "__main__":
    wang = WangLandau("Skeletons/training_checkpoint.pth", "Skeletons/memorization_phase.pth")
    pre_compute_energy = str(input("Pre-compute energy? "))

    if pre_compute_energy == "y":
        wang.compute_energy(200, 50)

    # wang.inspect_loss_window()
    # wang.inspectModels()
    wang.wanglandau("grokked", 200, 50)