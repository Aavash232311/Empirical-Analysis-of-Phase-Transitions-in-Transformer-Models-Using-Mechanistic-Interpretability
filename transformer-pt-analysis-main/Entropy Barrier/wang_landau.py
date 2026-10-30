import time
import contextlib
import threading

import os
import torch
from torch.nn.utils import vector_to_parameters, parameters_to_vector

import analysis as a
import transformer as t
import numpy as np
from concurrent.futures import ThreadPoolExecutor


class WangLandau:

    def __init__(
        self,
        skeleton_path_grokked: str,
        skeleton_path_memorized: str,
        save_path_sk: str = "Binary"

    ) -> None:
        self.skeleton_path_grokked = skeleton_path_grokked
        self.skeleton_path_memorized = skeleton_path_memorized

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cfg = t.TrainConfig

        # Skeletons of grokked models
        self.skeletons_grokked = torch.load(self.skeleton_path_grokked, map_location=self.device)
        self.skeleton_memorized = torch.load(self.skeleton_path_memorized, map_location=self.device)

        # Paramaters of model, between two phases
        state_dict = self.skeletons_grokked['model_state_dict']
        self.theta_grokked = torch.cat([p.flatten() for p in state_dict.values()]).detach().to(self.device)

        state_dict = self.skeleton_memorized['model_state_dict']
        self.theta_memorized = torch.cat([p.flatten() for p in state_dict.values()]).detach().to(self.device)


        # Instance of transformer
        self.model = t.MinimalTransformer(
            vocab_size=self.cfg.vocab_size, d_model=self.cfg.d_model, n_heads=self.cfg.n_heads,
            num_layers=self.cfg.num_layers, max_seq_len=self.cfg.seq_len,
            hidden_mlp=self.cfg.hidden_mlp,
        ).to(self.device)
        self.model.eval()


        self.seed = 42
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(self.seed)

        self.binary_d_mass_path = save_path_sk


    def _nudge_theta(self, parameters: torch.Tensor, rel_std: float = 0.05, min_scale: float = 1e-3,
                    generator: torch.Generator = None) -> torch.Tensor:
        generator = generator if generator is not None else self.rng
        noise = torch.randn(parameters.shape, generator=generator,
                            device=parameters.device, dtype=parameters.dtype)
        scale = parameters.abs().clamp_min(min_scale)
        return parameters + rel_std * scale * noise


    def _save_precomputed_binary(self, tag: str, d_mass, loss, theta):
         # float32 on disk, because wang_landau() reads with dtype=np.float32
        d_mass.cpu().numpy().astype(np.float32).tofile(
            os.path.join(self.binary_d_mass_path, f"{tag}_diagonal_mass.bin"))
        loss.cpu().numpy().astype(np.float32).tofile(
            os.path.join(self.binary_d_mass_path, f"{tag}_loss.bin"))
        theta.detach().cpu().numpy().astype(np.float32).tofile(
            os.path.join(self.binary_d_mass_path, f"{tag}_nudged_theta.bin"))


    def precompute_d_mass_landscape(self, sweeps: int, step: int, parameters, num_threads: int = 8, tag:str = None):

        if tag is not None:
            print(tag)
        start_time = time.perf_counter()
        fc = sweeps * step
        d_mass_list = torch.zeros(fc)
        loss_list = torch.zeros(fc)

        thread_local = threading.local()

        def get_thread_model():
            if not hasattr(thread_local, "model"):
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

            if use_cuda:
                stream.synchronize()
            return i, m_diag.item(), loss.item(), (new_theta if i == fc - 1 else None)

        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            for count, (i, val, loss, new_theta) in enumerate(pool.map(worker, range(fc)), start=1):
                d_mass_list[i] = val
                loss_list[i] = loss
                if count % 25 == 0 or count == fc:
                    print(f"[{count}/{fc}] Diagonal Spectral Mass: {val:.4f}")

        end_time = time.perf_counter()

        print(f"Execution time: {(end_time - start_time) / 60:.2f} Minutes")
        return d_mass_list, loss_list, new_theta



        

    def compute_energy(self, sweeps: int, step: int):
        # We need to compute this for both memorization and generalization phase so we pass in the paramaters two times.
        d_mass_grokked, loss_list_grokked, new_theta_grokked = self.precompute_d_mass_landscape(sweeps, step, self.theta_grokked, "Computing grokked phase")
        d_mass_mem, loss_list_mem, new_theta_mem = self.precompute_d_mass_landscape(sweeps, step, self.theta_memorized, "Computing memorization phase")

        self._save_precomputed_binary("grokked", d_mass_grokked, loss_list_grokked, new_theta_grokked)
        self._save_precomputed_binary("mem", d_mass_mem, loss_list_mem, new_theta_mem)
        # we have something like for new theta, we have d_mass thats the energy 

    

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

if __name__ == "__main__":
    wang = WangLandau("Skeletons/training_checkpoint.pth", "Skeletons/memorization_phase.pth")
    wang.compute_energy(200, 50)